use super::launch_matmul;
use crate::{ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor};
use burn_backend::{DType, cubecl::dtype_to_elem_type};
use burn_std::Metadata;
use cubek::{
    matmul::{
        multi_level::{
            Strategy, components::tile::TileMatmulKind, routines::batch::simple::SimpleArgs,
        },
        routine::BlueprintStrategy,
    },
    reduce::{
        ReduceStrategy,
        components::instructions::ReduceOperationConfig,
        launch::{RoutineStrategy, VectorizationStrategy},
        routines::{BlueprintStrategy as ReduceBlueprint, unit::UnitStrategy},
    },
};

/// Maximum legal partition count, included unanchored in the tuning key.
/// Only split an unbatched FP32 contraction; cap scratch at 128 MiB and keep
/// at least 1024 products per partition. Powers of two divide K exactly.
pub(super) fn partition_limit(lhs: &CubeTensor, rhs: &CubeTensor, out: &CubeTensor) -> usize {
    let a = lhs.meta.shape();
    let b = rhs.meta.shape();
    if out.dtype != DType::F32
        || out.meta.tiling.is_tiled()
        || lhs.dtype != DType::F32
        || rhs.dtype != DType::F32
        || lhs.qparams.is_some()
        || rhs.qparams.is_some()
        || lhs.meta.tiling.is_tiled()
        || rhs.meta.tiling.is_tiled()
        || a[..a.len() - 2].iter().product::<usize>() != 1
        || b[..b.len() - 2].iter().product::<usize>() != 1
    {
        return 1;
    }
    partition_limit_for_shape(a[a.len() - 2], b[b.len() - 1], a[a.len() - 1])
}

fn partition_limit_for_shape(m: usize, n: usize, k: usize) -> usize {
    let Some(elements) = m.checked_mul(n).filter(|&elements| elements != 0) else {
        return 1;
    };
    if k < 4096 {
        return 1;
    }
    let ceiling = 128.min(k / 1024).min((32 * 1024 * 1024) / elements);
    let mut partitions = 1;
    while partitions * 2 <= ceiling && k.is_multiple_of(partitions * 2) {
        partitions *= 2;
    }
    partitions
}

/// Parallel partial matmuls followed by one sum into the caller's output.
/// Operands are metadata-only views, including transposed/strided matrices.
pub(super) fn launch_split_k(
    mut lhs: CubeTensor,
    mut rhs: CubeTensor,
    mut out: CubeTensor,
    requested: usize,
    tile_matmul: TileMatmulKind,
) -> Result<(), String> {
    let partitions = requested.min(partition_limit(&lhs, &rhs, &out));
    if partitions < 2 {
        return Err("Split-K requires an unbatched, divisible long-K FP32 matmul".into());
    }
    let rank_a = lhs.meta.num_dims();
    let rank_b = rhs.meta.num_dims();
    let rank_out = out.meta.num_dims();
    let m = lhs.meta.shape()[rank_a - 2];
    let n = rhs.meta.shape()[rank_b - 1];
    let chunk = lhs.meta.shape()[rank_a - 1] / partitions;
    let a_m = lhs.meta.strides()[rank_a - 2];
    let a_k = lhs.meta.strides()[rank_a - 1];
    let b_k = rhs.meta.strides()[rank_b - 2];
    let b_n = rhs.meta.strides()[rank_b - 1];
    *lhs.meta = Metadata::new([partitions, m, chunk], [chunk * a_k, a_m, a_k]);
    *rhs.meta = Metadata::new([partitions, chunk, n], [chunk * b_k, b_k, b_n]);
    let partial = empty_device_contiguous_dtype(
        out.client.clone(),
        out.device.clone(),
        [partitions, m, n].into(),
        out.dtype,
    );
    let args = BlueprintStrategy::Inferred(SimpleArgs {
        multi_rows: false,
        tile_matmul,
    });
    let strategy = match tile_matmul {
        TileMatmulKind::Cmma => Strategy::SimpleCyclicCmma(args),
        TileMatmulKind::Mma => Strategy::SimpleCyclicMma(args),
        _ => return Err("Split-K requires an accelerated tile matmul".into()),
    };
    launch_matmul(&strategy, lhs, rhs, partial.clone()).map_err(|e| format!("{e:?}"))?;
    let out_m = out.meta.strides()[rank_out - 2];
    let out_n = out.meta.strides()[rank_out - 1];
    *out.meta = Metadata::new([1, m, n], [m * out_m, out_m, out_n]);
    let config = ReduceOperationConfig::Sum;
    let dtypes = config.precision(dtype_to_elem_type(out.dtype), None);
    cubek::reduce::reduce(
        &out.client,
        partial.binding(),
        out.clone().binding(),
        0,
        ReduceStrategy {
            routine: RoutineStrategy::Unit(ReduceBlueprint::Inferred(UnitStrategy)),
            vectorization: VectorizationStrategy {
                parallel_output_vectorization: false,
            },
            autotune_level: Default::default(),
        },
        config,
        dtypes,
    )
    .map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_k_limit_preserves_divisibility_and_bounds_scratch() {
        for (m, n, k) in [
            (192, 576, 131072),
            (768, 3840, 32768),
            (13, 19, 12345),
            (0, 8, 8192),
        ] {
            let p = partition_limit_for_shape(m, n, k);
            assert!(p.is_power_of_two());
            assert_eq!(k % p, 0);
            if p > 1 {
                assert!(p * m * n <= 32 * 1024 * 1024);
                assert!(k / p >= 1024);
            }
        }
        assert_eq!(partition_limit_for_shape(192, 576, 131072), 128);
        assert_eq!(partition_limit_for_shape(13, 19, 12345), 1);
    }

    /// Explicitly opt in on an accelerator. Dyadic inputs make the reference
    /// exact in TF32 too; test transposed and padded input storage, odd output
    /// dimensions, singleton leading dimensions, and every requested split.
    #[test]
    #[ignore]
    fn split_k_matches_host_for_strided_operands() {
        use burn_backend::{TensorData, Tolerance};
        let device = crate::CubeDevice::default();
        let (m, n, k) = (13, 19, 8192);
        let a: Vec<f32> = (0..k * m)
            .map(|i| ((i * 7 % 17) as f32 - 8.0) / 32.0)
            .collect();
        let b: Vec<f32> = (0..k * (n + 3))
            .map(|i| ((i * 3 % 11) as f32 - 5.0) / 64.0)
            .collect();
        let mut lhs = crate::ops::from_data(TensorData::new(a.clone(), [k, m]), &device);
        let a_k = lhs.meta.strides()[0];
        let a_m = lhs.meta.strides()[1];
        *lhs.meta = Metadata::new([1, m, k], [k * a_k, a_m, a_k]);
        let mut rhs = crate::ops::from_data(TensorData::new(b.clone(), [k, n + 3]), &device);
        let b_k = rhs.meta.strides()[0];
        let b_n = rhs.meta.strides()[1];
        *rhs.meta = Metadata::new([1, k, n], [k * b_k, b_k, b_n]);
        let expected: Vec<f32> = (0..m * n)
            .map(|index| {
                let (row, col) = (index / n, index % n);
                (0..k).map(|j| a[j * m + row] * b[j * (n + 3) + col]).sum()
            })
            .collect();
        let expected = TensorData::new(expected, [1, m, n]);
        for partitions in [2, 8, 32, 128] {
            for tile in [TileMatmulKind::Cmma, TileMatmulKind::Mma] {
                let out = empty_device_contiguous_dtype(
                    lhs.client.clone(),
                    device.clone(),
                    [1, m, n].into(),
                    DType::F32,
                );
                launch_split_k(lhs.clone(), rhs.clone(), out.clone(), partitions, tile).unwrap();
                expected.assert_approx_eq::<f32>(
                    &crate::ops::into_data_sync(out),
                    Tolerance::rel_abs(1e-5, 1e-5),
                );
            }
        }
    }
}
