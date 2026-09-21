/// Generic container for storing tensors keyed by an id.
pub mod container;
/// Tensor data type definitions.
pub mod dtype;
/// The order a tensor's dimensions occupy memory in.
pub mod layout;
/// Batched matmul transformation utilities.
pub mod matmul;
/// Quantization data representation.
pub mod quantization;
/// Tensor shape utilities.
pub mod shape;
/// Tensor slicing utilities.
pub mod slice;

pub use dtype::*;
pub use layout::*;
pub use matmul::*;
pub use quantization::*;
pub use shape::*;
pub use slice::*;

pub use cubecl_zspace::indexing::{self, *};
pub use cubecl_zspace::{Strides, metadata::Metadata, strides};

/// Check if the current tensor is contiguous.
///
/// A tensor is considered contiguous if its elements are stored in memory
/// such that the stride at position `k` is equal to the product of the shapes
/// of all dimensions greater than `k`, except for dimensions of size one.
///
/// This means that strides increase as you move from the rightmost to the leftmost dimension,
/// ignoring dimensions of size one.
///
/// Dimensions of size one may have any stride, including zero, since their stride is never applied.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    // A dimension of size one is visited once, so its stride is never applied and can hold
    // anything: a permute leaves whatever the axis carried before, and broadcast views use 0.
    // Comparing it would report a contiguous tensor as strided, as [2, 3, 1] with strides
    // [3, 1, 3] coming out of `permute([0, 2, 1])`.
    for ((&expected, &stride), &dim) in contiguous_strides(shape)
        .iter()
        .zip(strides)
        .zip(shape.iter())
    {
        if dim != 1 && expected != stride {
            return false;
        }
    }

    true
}

/// Check if the current tensor fills its buffer without gaps.
///
/// Unlike [is_contiguous], this holds for any ordering of the dimensions: a permuted tensor is
/// dense, a tensor whose rows were padded by a pitched allocator is not. Dimensions of size one
/// carry no information about the layout and are ignored.
pub fn is_dense(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }

    let mut dims: SmallVec<[(usize, usize); 5]> = shape
        .iter()
        .zip(strides)
        .filter(|&(&dim, _)| dim > 1)
        .map(|(&dim, &stride)| (dim, stride))
        .collect();

    dims.sort_unstable_by_key(|&(_, stride)| stride);

    let mut expected = 1;

    for (dim, stride) in dims {
        if stride != expected {
            return false;
        }

        expected *= dim;
    }

    true
}

/// Computes the strides for a contiguous tensor with the given shape.
///
/// In a contiguous row-major tensor, the stride for each dimension
/// equals the product of all dimension sizes to its right.
pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides = strides![0; shape.len()];
    let mut current = 1;

    for (i, &dim) in shape.iter().enumerate().rev() {
        strides[i] = current;
        current *= dim;
    }

    strides
}

/// The action to take for a reshape operation.
#[derive(Debug)]
pub enum ReshapeAction {
    /// Updating the strides is sufficient to handle the reshape.
    UpdateStrides {
        /// The new strides.
        strides: Strides,
    },
    /// The strides are not compatible, we should recompute the buffer.
    Recompute,
    /// The strides are already correct.
    NoChange,
}

/// The reshape kind.
#[derive(Debug, PartialEq)]
pub enum ReshapeAnalysis {
    /// Original tensor is contiguous, can update the strides.
    IsContiguous,
    /// Original tensor is highly permuted, can't update the strides.
    HighlyPermuted,
    /// Only batch dimensions are added, can update the strides.
    Broadcasted,
    /// Dimensions are only split, can update the strides.
    Split,
    /// Original tensor is bigger than output shape.
    SmallerRank,
    /// New shape is the same.
    NoChange,
}

impl ReshapeAnalysis {
    /// Returns the proper action to take for the current analysis.
    pub fn action(&self, shape: &[usize], strides: &[usize], shape_new: &[usize]) -> ReshapeAction {
        match self {
            ReshapeAnalysis::IsContiguous => ReshapeAction::UpdateStrides {
                strides: contiguous_strides(shape_new),
            },
            ReshapeAnalysis::NoChange => ReshapeAction::NoChange,
            ReshapeAnalysis::HighlyPermuted | ReshapeAnalysis::SmallerRank => {
                ReshapeAction::Recompute
            }
            ReshapeAnalysis::Broadcasted => {
                let shape_rank = shape.len();
                let shape_new_rank = shape_new.len();
                let n_new_batch = shape_new_rank - shape_rank;
                let num_elems = shape.iter().product::<usize>();
                let strides_new = broadcast_strides(n_new_batch, shape_rank, num_elems, strides);

                ReshapeAction::UpdateStrides {
                    strides: strides_new,
                }
            }
            ReshapeAnalysis::Split => {
                let strides_new = split_strides(shape, strides, shape_new);

                ReshapeAction::UpdateStrides {
                    strides: strides_new,
                }
            }
        }
    }
}

/// Returns the proper action to take when reshaping a tensor.
pub fn reshape_action(shape: &Shape, strides: &Strides, shape_new: &Shape) -> ReshapeAction {
    let action =
        reshape_analysis(shape, Some(strides), shape_new).action(shape, strides, shape_new);
    if matches!(action, ReshapeAction::Recompute)
        && let Some(strides) = reshape_view_strides(shape, strides, shape_new)
    {
        return ReshapeAction::UpdateStrides { strides };
    }
    action
}

// A strided tensor is a sequence of physically contiguous runs. A view may
// split or merge dimensions inside a run, but cannot cross a padding or
// permutation boundary. Unit dimensions do not constrain the physical layout.
fn reshape_view_strides(
    shape: &[usize],
    strides: &[usize],
    shape_new: &[usize],
) -> Option<Strides> {
    if shape.is_empty() || shape.contains(&0) || shape_new.contains(&0) {
        return None;
    }
    let mut result = strides![0; shape_new.len()];
    let mut target_axis = shape_new.len();
    let mut base_stride = *strides.last()?;
    let mut source_elements = 1usize;
    let mut target_elements = 1usize;
    for axis in (0..shape.len()).rev() {
        source_elements = source_elements.checked_mul(shape[axis])?;
        let run_ends = axis == 0
            || (shape[axis - 1] != 1
                && strides[axis - 1] != source_elements.checked_mul(base_stride)?);
        if run_ends {
            while target_axis > 0
                && (target_elements < source_elements || shape_new[target_axis - 1] == 1)
            {
                target_axis -= 1;
                result[target_axis] = target_elements.checked_mul(base_stride)?;
                target_elements = target_elements.checked_mul(shape_new[target_axis])?;
            }
            if target_elements != source_elements {
                return None;
            }
            if axis > 0 {
                base_stride = strides[axis - 1];
                source_elements = 1;
                target_elements = 1;
            }
        }
    }
    (target_axis == 0).then_some(result)
}

/// Calculate the new strides given added batch dimensions.
pub fn broadcast_strides(
    n_new_batch: usize,
    rank_prev: usize,
    num_elems: usize,
    strides: &[usize],
) -> Strides {
    let mut strides_new = strides![num_elems; rank_prev + n_new_batch];

    for (i, s) in strides.iter().enumerate() {
        strides_new[i + n_new_batch] = *s;
    }

    strides_new
}

/// Calculate the new strides given added split dimensions.
pub fn split_strides(shape: &[usize], strides: &[usize], shape_new: &[usize]) -> Strides {
    let mut strides_new = strides![1; shape_new.len()];

    // Unit dims in the old shape never anchor a group of new dims, and their
    // stride can be arbitrary (0 for broadcast views, pitched values, ...).
    // Skip them so a real new dim never inherits a unit dim's stride —
    // e.g. reshaping [26, 1] with strides [1, 0] (a `repeat_dim` broadcast
    // view) to [26, 1, 1] must keep stride 1 on dim 0; propagating the 0
    // would make every index along dim 0 alias the first element.
    let skip_unit_dims = |mut idx: usize| {
        while idx > 0 && shape[idx] == 1 {
            idx -= 1;
        }
        idx
    };

    let mut old_idx = skip_unit_dims(shape.len() - 1);
    let mut current_stride = strides[old_idx];
    let mut dim_prod = 1;

    for (i, dim) in shape_new.iter().enumerate().rev() {
        dim_prod *= *dim;
        strides_new[i] = current_stride;
        if *dim == 1 {
            continue;
        } else if dim_prod == shape[old_idx] {
            old_idx = skip_unit_dims(old_idx.saturating_sub(1));
            current_stride = strides[old_idx];
            dim_prod = 1;
        } else {
            current_stride *= *dim;
        }
    }

    strides_new
}

/// Returns the analysis of a reshape operation.
pub fn reshape_analysis(
    shape: &Shape,
    strides: Option<&Strides>,
    shape_new: &Shape,
) -> ReshapeAnalysis {
    let shape_rank = shape.len();
    let shape_new_rank = shape_new.len();

    let is_contiguous = match strides {
        Some(strides) => is_contiguous(shape, strides),
        None => false,
    };

    if is_contiguous {
        return ReshapeAnalysis::IsContiguous;
    }

    if shape_new_rank < shape_rank {
        return ReshapeAnalysis::SmallerRank;
    }

    let n_new_batch = shape_new_rank - shape_rank;

    match n_new_batch > 0 {
        true => {
            if shape.as_ref() == &shape_new[n_new_batch..shape_new_rank]
                && shape_new[0..n_new_batch].iter().all(|it| *it == 1)
            {
                return ReshapeAnalysis::Broadcasted;
            } else {
                let mut dim_prod = 1;
                let mut old_idx = 0;
                for dim in shape_new.iter() {
                    dim_prod *= *dim;

                    // We need to ignore unit dims because they don't affect analysis and break
                    // things because they match the default `dim_prod`. If we don't do this,
                    // reshapes like [2, 3] to [2, 3, 1] will panic from out of bounds access.
                    if *dim == 1 {
                        continue;
                    } else if dim_prod == shape[old_idx] {
                        dim_prod = 1;
                        old_idx += 1;
                    } else if dim_prod > shape[old_idx] {
                        return ReshapeAnalysis::HighlyPermuted;
                    }
                }
                return ReshapeAnalysis::Split;
            }
        }

        false => {
            if shape == shape_new {
                return ReshapeAnalysis::NoChange;
            }
        }
    };

    ReshapeAnalysis::HighlyPermuted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_merges_pitched_batch_dimensions_without_copying() {
        let action = reshape_action(
            &[2048, 64, 960].into(),
            &[65536, 1024, 1].into(),
            &[131072, 960].into(),
        );
        match action {
            ReshapeAction::UpdateStrides { strides } => assert_eq!(&*strides, &[1024, 1]),
            other => panic!("Expected a pitched view, got {other:?}"),
        }
    }

    #[test]
    fn reshape_view_strides_preserve_every_logical_address() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[2, 3, 4], &[24, 8, 1]),
            (&[2, 3, 4], &[1, 8, 2]),
            (&[2, 3, 4], &[0, 4, 1]),
            (&[2, 3, 4], &[0, 0, 0]),
            (&[2, 1, 3], &[6, 999, 2]),
            (&[1, 2, 1], &[777, 3, 999]),
        ];
        let offset = |mut index: usize, shape: &[usize], strides: &[usize]| {
            let mut offset = 0;
            for (&dim, &stride) in shape.iter().zip(strides).rev() {
                offset += (index % dim) * stride;
                index /= dim;
            }
            offset
        };
        for &(shape, strides) in cases {
            let elements: usize = shape.iter().product();
            let mut targets = alloc::vec![alloc::vec![elements]];
            for a in 1..=elements {
                if !elements.is_multiple_of(a) {
                    continue;
                }
                targets.push(alloc::vec![a, elements / a]);
                for b in 1..=elements / a {
                    if (elements / a).is_multiple_of(b) {
                        targets.push(alloc::vec![a, b, elements / a / b]);
                    }
                }
            }
            for target in targets {
                if let Some(new_strides) = reshape_view_strides(shape, strides, &target) {
                    for index in 0..elements {
                        assert_eq!(
                            offset(index, shape, strides),
                            offset(index, &target, &new_strides),
                            "{shape:?}/{strides:?} -> {target:?}/{new_strides:?}"
                        );
                    }
                }
            }
        }
        assert!(reshape_view_strides(&[2, 3], &[1, 2], &[6]).is_none());
        assert!(reshape_view_strides(&[2, 3, 4], &[24, 8, 1], &[24]).is_none());
    }

    #[test]
    fn test_reshape_analysis_is_contiguous() {
        let analysis = reshape_analysis(
            &[32, 1, 1, 1].into(),
            Some(&[1, 1, 1, 1].into()),
            &[1, 1, 32, 1, 1, 1].into(),
        );

        assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
    }

    #[test]
    fn test_reshape_analysis_is_contiguous_2() {
        let analysis = reshape_analysis(
            &[32, 1, 1, 8].into(),
            Some(&[8, 8, 8, 1].into()),
            &[1, 1, 32, 1, 1, 8].into(),
        );

        assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
    }

    // [32, 1, 1, 1] with strides [1, 32, 32, 32] is contiguous: only the first axis is
    // ever indexed, and it steps by one. The trailing strides belong to axes of size one
    // and are never applied, so the analysis short-circuits before reaching the batch and
    // split branches, which the three tests below cover with genuinely strided inputs.
    #[test]
    fn test_reshape_analysis_unit_axes_are_contiguous() {
        for shape_new in [
            [1, 1, 32, 1, 1, 1].as_slice(),
            [32, 1, 1, 1, 1].as_slice(),
            [4, 8, 1, 1, 1].as_slice(),
        ] {
            let analysis = reshape_analysis(
                &[32, 1, 1, 1].into(),
                Some(&[1, 32, 32, 32].into()),
                &shape_new.into(),
            );

            assert_eq!(analysis, ReshapeAnalysis::IsContiguous, "{shape_new:?}")
        }
    }

    #[test]
    fn test_reshape_analysis_broadcasted_batch() {
        let analysis =
            reshape_analysis(&[32, 32].into(), Some(&[1, 32].into()), &[1, 32, 32].into());

        assert_eq!(analysis, ReshapeAnalysis::Broadcasted)
    }

    #[test]
    fn test_reshape_analysis_unsqueeze_split() {
        // Unsqueeze
        let analysis =
            reshape_analysis(&[32, 32].into(), Some(&[1, 32].into()), &[32, 32, 1].into());

        assert_eq!(analysis, ReshapeAnalysis::Split)
    }

    #[test]
    fn test_reshape_analysis_split() {
        let analysis =
            reshape_analysis(&[32, 32].into(), Some(&[1, 32].into()), &[4, 8, 32].into());

        assert_eq!(analysis, ReshapeAnalysis::Split)
    }

    #[test]
    fn test_split_strides_trailing_unit_dim_broadcast_view() {
        // A `repeat_dim` broadcast view: [26, 1] with strides [1, 0],
        // unsqueezed to [26, 1, 1]. Dim 0 must keep stride 1 — propagating
        // the unit dim's stride 0 makes every row alias row 0 (this breaks
        // e.g. `scatter` index tensors built via unsqueeze + repeat).
        let strides = split_strides(&[26, 1], &[1, 0], &[26, 1, 1]);
        assert_eq!(strides.as_ref(), &[1, 1, 1]);
    }

    #[test]
    fn test_split_strides_trailing_unit_dims_arbitrary_strides() {
        // Unit dims can carry arbitrary strides (broadcast 0, pitched
        // values, ...). They must not anchor the stride walk.
        let strides = split_strides(&[32, 1, 1, 1], &[1, 32, 32, 32], &[32, 1, 1, 1, 1]);
        assert_eq!(strides.as_ref(), &[1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_split_strides_split_of_broadcast_dim_keeps_zero() {
        // Splitting a real broadcast (stride 0) dim keeps 0 on the split
        // parts; the leading real dim keeps its stride.
        let strides = split_strides(&[26, 16], &[1, 0], &[26, 4, 4]);
        assert_eq!(strides.as_ref(), &[1, 0, 0]);
    }

    #[test]
    fn test_is_dense_contiguous() {
        assert!(is_dense(&[2, 2, 2, 2], &[8, 4, 2, 1]));
    }

    #[test]
    fn test_is_dense_permuted() {
        assert!(is_dense(&[2, 2, 2, 2], &[8, 1, 4, 2]));
    }

    #[test]
    fn test_is_dense_pitched_row() {
        assert!(!is_dense(&[2, 2, 2, 2], &[16, 8, 4, 1]));
        assert!(!is_dense(&[1, 8, 6, 6], &[384, 48, 8, 1]));
    }

    #[test]
    fn test_is_dense_unit_dims_carry_no_layout() {
        // A unit dim can hold any stride, a broadcast 0 included, without leaving a gap.
        assert!(is_dense(&[1, 4, 1], &[0, 1, 7]));
    }

    #[test]
    fn test_is_dense_rank_mismatch() {
        assert!(!is_dense(&[2, 3], &[1]));
    }

    #[test]
    fn test_split_strides_plain_unsqueeze() {
        let strides = split_strides(&[26, 16], &[16, 1], &[26, 16, 1]);
        assert_eq!(strides.as_ref(), &[16, 1, 1]);
    }
}
