use crate::tensor::CubeTensor;
use burn_backend::DType;
use burn_std::{MatmulTransformAnalysis, MatmulTransformPolicy, Metadata};
use cubecl::cuda::blas::Gemm;

// cuBLAS strided batches need a single physical stride between matrices. A
// permuted batch/head layout can violate that even when each matrix is dense.
fn batch_stride(meta: &Metadata) -> Option<(usize, usize)> {
    let rank = meta.num_dims();
    let mut count = 1usize;
    let mut stride = 0usize;
    for (&dim, &physical) in meta.shape()[..rank - 2]
        .iter()
        .zip(&meta.strides()[..rank - 2])
        .rev()
    {
        if dim == 1 {
            continue;
        }
        if count == 1 {
            stride = physical;
        }
        if physical != count.checked_mul(stride)? {
            return None;
        }
        count = count.checked_mul(dim)?;
    }
    Some((count, stride))
}

fn layout(meta: &Metadata) -> Option<(bool, i32)> {
    let rank = meta.num_dims();
    let shape = meta.shape();
    let strides = meta.strides();
    if strides[rank - 1] == 1 && strides[rank - 2] >= shape[rank - 1] {
        Some((false, strides[rank - 2].try_into().ok()?))
    } else if strides[rank - 2] == 1 && strides[rank - 1] >= shape[rank - 2] {
        Some((true, strides[rank - 1].try_into().ok()?))
    } else {
        None
    }
}

pub(super) fn descriptor(lhs: &CubeTensor, rhs: &CubeTensor, out: &CubeTensor) -> Option<Gemm> {
    if !lhs.client.name().starts_with("cuda-")
        || [lhs, rhs, out].iter().any(|t| {
            t.dtype != DType::F32
                || t.qparams.is_some()
                || t.meta.tiling.is_tiled()
                || t.meta.num_dims() < 2
        })
    {
        return None;
    }
    let mut a = (*lhs.meta).clone();
    let mut c = (*out.meta).clone();
    let action = MatmulTransformPolicy::AlwaysMerge
        .action(&MatmulTransformAnalysis::from_metadata(&a, &rhs.meta, &c));
    action.apply(&mut a);
    action.apply(&mut c);
    let b = &rhs.meta;
    let (ba, sa) = batch_stride(&a)?;
    let (bb, sb) = batch_stride(b)?;
    let (bc, sc) = batch_stride(&c)?;
    // Shared-weight linear layers are folded above. Other broadcast patterns
    // stay with generated kernels rather than constructing device pointer lists.
    if ba != bc || bb != bc || bc == 0 {
        return None;
    }
    let (ta, lda) = layout(&a)?;
    let (tb, ldb) = layout(b)?;
    let (tc, ldc) = layout(&c)?;
    if tc {
        return None;
    }
    let m = a.shape()[a.num_dims() - 2];
    let k = a.shape()[a.num_dims() - 1];
    let n = b.shape()[b.num_dims() - 1];
    if m == 0 || n == 0 || k == 0 {
        return None;
    }
    Some(Gemm {
        m: m.try_into().ok()?,
        n: n.try_into().ok()?,
        k: k.try_into().ok()?,
        batches: bc.try_into().ok()?,
        lhs_transposed: ta,
        rhs_transposed: tb,
        lhs_leading: lda,
        rhs_leading: ldb,
        out_leading: ldc,
        lhs_batch_stride: sa.try_into().ok()?,
        rhs_batch_stride: sb.try_into().ok()?,
        out_batch_stride: sc.try_into().ok()?,
    })
}

pub(super) fn launch(lhs: CubeTensor, rhs: CubeTensor, out: CubeTensor) -> Result<(), String> {
    let gemm = descriptor(&lhs, &rhs, &out).ok_or("Unsupported cuBLAS matmul layout")?;
    // SAFETY: Burn matmul operands/output own valid nonaliasing bindings; the
    // descriptor accepts only plain FP32 matrix layouts and uniform batches.
    unsafe {
        gemm.launch(
            &lhs.client,
            lhs.handle.clone().binding(),
            rhs.handle.binding(),
            out.handle.binding(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vendor_layout_rejects_permuted_batches_and_preserves_padding() {
        assert_eq!(
            batch_stride(&Metadata::new([2, 3, 4, 5], [60, 20, 5, 1])),
            Some((6, 20))
        );
        assert_eq!(
            batch_stride(&Metadata::new([2, 3, 4, 5], [60, 5, 15, 1])),
            None
        );
        assert_eq!(layout(&Metadata::new([7, 5], [8, 1])), Some((false, 8)));
        assert_eq!(layout(&Metadata::new([7, 5], [1, 8])), Some((true, 8)));
        assert_eq!(layout(&Metadata::new([7, 5], [10, 2])), None);
    }
    #[test]
    #[ignore = "requires CUDA and cuBLAS"]
    fn vendor_gemm_merges_shared_weights_and_matches_host() {
        use burn_backend::TensorData;
        let device = crate::CubeDevice::default();
        let (batches, rows, k, n) = (2, 16, 64, 32);
        let a: Vec<f32> = (0..batches * rows * k)
            .map(|i| ((i % 13) as f32 - 6.) / 8.)
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 - 8.) / 8.).collect();
        let lhs = crate::ops::from_data(TensorData::new(a.clone(), [batches, rows, k]), &device);
        let rhs = crate::ops::from_data(TensorData::new(b.clone(), [1, k, n]), &device);
        let out = super::super::init_matmul_output(&lhs, &rhs, DType::F32);
        let desc = descriptor(&lhs, &rhs, &out).expect("shared weights merge");
        assert_eq!(desc.batches, 1);
        assert_eq!(desc.m, (batches * rows) as i32);
        launch(lhs, rhs, out.clone()).unwrap();
        let expected: Vec<f32> = (0..batches * rows * n)
            .map(|i| {
                let row = i / n;
                let col = i % n;
                (0..k).map(|j| a[row * k + j] * b[j * n + col]).sum()
            })
            .collect();
        crate::ops::into_data_sync(out).assert_approx_eq::<f32>(
            &TensorData::new(expected, [batches, rows, n]),
            burn_backend::Tolerance::rel_abs(1e-4, 1e-4),
        );
    }
}
