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
/// Only split an unbatched FP32 contraction into a dense row-major output;
/// cap scratch at 128 MiB and keep at least 1024 products per partition.
/// Powers of two divide K exactly.
///
/// The partial sums are reduced straight into `out`, and that reduction writes
/// a dense layout: into a row-pitched output (e.g. `[320, 24]` stored with a
/// row stride of 32) it scatters rows to the wrong addresses and leaves the
/// rest holding whatever the pool left there. That was Oxi's smolgen.compress
/// weight gradient — sporadic 1e3-1e5 gradient spikes and eventual NaN.
pub(super) fn partition_limit(lhs: &CubeTensor, rhs: &CubeTensor, out: &CubeTensor) -> usize {
    let a = lhs.meta.shape();
    let b = rhs.meta.shape();
    if !is_dense_row_major(out.meta.shape(), out.meta.strides())
        || out.dtype != DType::F32
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

fn is_dense_row_major(shape: &[usize], strides: &[usize]) -> bool {
    let mut expected = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        if dim != 1 && stride != expected {
            return false;
        }
        expected = expected.saturating_mul(dim);
    }
    true
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
    out: CubeTensor,
    requested: usize,
    tile_matmul: TileMatmulKind,
) -> Result<(), String> {
    let partitions = requested.min(partition_limit(&lhs, &rhs, &out));
    if partitions < 2 {
        return Err("Split-K requires an unbatched, divisible long-K FP32 matmul".into());
    }
    let rank_a = lhs.meta.num_dims();
    let rank_b = rhs.meta.num_dims();
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
    launch_split_k_with_scratch(lhs, rhs, out, partitions, tile_matmul, partial)
}

/// `launch_split_k` with the `[partitions, m, n]` partial-product scratch
/// supplied by the caller, so tests can hand it stale contents: the result
/// must not depend on what the pool left in it.
fn launch_split_k_with_scratch(
    lhs: CubeTensor,
    rhs: CubeTensor,
    mut out: CubeTensor,
    partitions: usize,
    tile_matmul: TileMatmulKind,
    partial: CubeTensor,
) -> Result<(), String> {
    let rank_out = out.meta.num_dims();
    let m = lhs.meta.shape()[lhs.meta.num_dims() - 2];
    let n = rhs.meta.shape()[rhs.meta.num_dims() - 1];
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
    fn split_k_declines_pitched_outputs() {
        assert!(is_dense_row_major(&[320, 24], &[24, 1]));
        assert!(is_dense_row_major(&[1, 320, 24], &[999, 24, 1]));
        assert!(!is_dense_row_major(&[320, 24], &[32, 1]));
        assert!(!is_dense_row_major(&[1, 320, 24], &[10240, 32, 1]));
        assert!(!is_dense_row_major(&[320, 24], &[1, 320]));
    }

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

    /// The partial-product scratch comes from the memory pool uninitialized.
    /// Pre-fill it with NaN: any element the partial matmul fails to write
    /// shows up in the result. Shapes are the training loop's weight gradients
    /// (Xᵀ·dY over batch x 64 squares), including smolgen's 24-wide compress.
    #[test]
    #[ignore]
    fn split_k_is_independent_of_stale_scratch() {
        use burn_backend::TensorData;
        let device = crate::CubeDevice::default();
        let mut failures = 0;
        for (m, n, k) in [(13, 19, 8192), (320, 24, 8192), (320, 24, 32768), (448, 24, 32768), (320, 32, 32768), (320, 320, 32768)] {
            let a: Vec<f32> = (0..k * m).map(|i| ((i * 7 % 17) as f32 - 8.0) / 32.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i * 3 % 11) as f32 - 5.0) / 64.0).collect();
            // lhs = Xᵀ as a transposed view of a [k, m] X, like a Linear weight
            // gradient. Allocations may be row-pitched, so read real strides.
            let mut lhs = crate::ops::from_data(TensorData::new(a.clone(), [k, m]), &device);
            let (a_k, a_m) = (lhs.meta.strides()[0], lhs.meta.strides()[1]);
            *lhs.meta = Metadata::new([1, m, k], [k * a_k, a_m, a_k]);
            let mut rhs = crate::ops::from_data(TensorData::new(b.clone(), [k, n]), &device);
            let (b_k, b_n) = (rhs.meta.strides()[0], rhs.meta.strides()[1]);
            *rhs.meta = Metadata::new([1, k, n], [k * b_k, b_k, b_n]);
            let expected: Vec<f64> = (0..m * n)
                .map(|index| {
                    let (row, col) = (index / n, index % n);
                    (0..k).map(|j| a[j * m + row] as f64 * b[j * n + col] as f64).sum()
                })
                .collect();
            for partitions in [8, 32, 128] {
                if k % partitions != 0 || k / partitions < 64 {
                    continue;
                }
                for tile in [TileMatmulKind::Cmma, TileMatmulKind::Mma] {
                    let check = |label: &str, out: CubeTensor| -> usize {
                        let got = crate::ops::into_data_sync(out).to_vec::<f32>().unwrap();
                        let bad: Vec<usize> = got
                            .iter()
                            .zip(&expected)
                            .enumerate()
                            .filter(|(_, (g, e))| {
                                !g.is_finite() || ((**g as f64) - **e).abs() > 1e-3 * (1.0 + e.abs())
                            })
                            .map(|(i, _)| i)
                            .collect();
                        println!(
                            "{label} m={m} n={n} k={k} partitions={partitions} tile={tile:?}: {} of {} wrong{}",
                            bad.len(),
                            m * n,
                            bad.first().map_or(String::new(), |&i| format!(
                                " (first at row {} col {}: {} vs {})",
                                i / n, i % n, got[i], expected[i]
                            ))
                        );
                        bad.len()
                    };
                    let out = empty_device_contiguous_dtype(
                        lhs.client.clone(),
                        device.clone(),
                        [1, m, n].into(),
                        DType::F32,
                    );
                    launch_split_k(lhs.clone(), rhs.clone(), out.clone(), partitions, tile).unwrap();
                    failures += check("pool scratch", out);
                    let mut lhs = lhs.clone();
                    let mut rhs = rhs.clone();
                    let chunk = k / partitions;
                    *lhs.meta = Metadata::new([partitions, m, chunk], [chunk * a_k, a_m, a_k]);
                    *rhs.meta = Metadata::new([partitions, chunk, n], [chunk * b_k, b_k, b_n]);
                    let stale = crate::ops::from_data(
                        TensorData::new(vec![f32::NAN; partitions * m * n], [partitions, m, n]),
                        &device,
                    );
                    let out = empty_device_contiguous_dtype(
                        lhs.client.clone(),
                        device.clone(),
                        [1, m, n].into(),
                        DType::F32,
                    );
                    launch_split_k_with_scratch(lhs, rhs, out.clone(), partitions, tile, stale)
                        .unwrap();
                    failures += check("NaN scratch ", out);
                }
            }
        }
        assert_eq!(failures, 0);
    }

    /// Oxi's smolgen.compress weight gradient — Xᵀ·dY with a 24-wide dY and
    /// K = 32768 — came out wrong in training. Run the same product through
    /// every path autotune can pick and name the one that disagrees with the
    /// host. `cargo test -p burn-cubecl --features burn-cuda/cpp,burn-cuda/cublas
    /// --lib long_k_narrow_n_candidates -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn long_k_narrow_n_candidates_match_host() {
        use crate::kernel::matmul::{MatmulStrategy, matmul};
        use burn_backend::TensorData;
        let device = crate::CubeDevice::default();
        let mut failures = 0;
        for (m, n, k) in [(320, 24, 32768), (320, 24, 8192), (320, 32, 32768), (256, 24, 32768)] {
            let a: Vec<f32> = (0..k * m).map(|i| ((i * 7 % 17) as f32 - 8.0) / 32.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i * 3 % 11) as f32 - 5.0) / 64.0).collect();
            let x = crate::ops::from_data(TensorData::new(a.clone(), [k, m]), &device);
            let mut lhs = x.clone();
            let (x_row, x_col) = (x.meta.strides()[0], x.meta.strides()[1]);
            *lhs.meta = Metadata::new([m, k], [x_col, x_row]);
            let rhs = crate::ops::from_data(TensorData::new(b.clone(), [k, n]), &device);
            println!("m={m} n={n} k={k}: X strides {:?}, dY strides {:?}", x.meta.strides(), rhs.meta.strides());
            let expected: Vec<f64> = (0..m * n)
                .map(|index| {
                    let (row, col) = (index / n, index % n);
                    (0..k).map(|j| a[j * m + row] as f64 * b[j * n + col] as f64).sum()
                })
                .collect();
            let mut check = |label: &str, out: CubeTensor| {
                let got = crate::ops::into_data_sync(out).to_vec::<f32>().unwrap();
                let wrong: Vec<usize> = got
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .filter(|(_, (g, e))| !(((**g as f64) - **e).abs() <= 1e-3 * (1.0 + e.abs())))
                    .map(|(i, _)| i)
                    .collect();
                println!(
                    "  {label:<14} {} of {} wrong{}",
                    wrong.len(),
                    m * n,
                    wrong.first().map_or(String::new(), |&i| format!(
                        " (first row {} col {}: {} vs {})",
                        i / n, i % n, got[i], expected[i]
                    ))
                );
                failures += !wrong.is_empty() as usize;
            };
            let fresh = || empty_device_contiguous_dtype(lhs.client.clone(), device.clone(), [m, n].into(), DType::F32);
            check("autotune", matmul(lhs.clone(), rhs.clone(), None, MatmulStrategy::Autotune, DType::F32).unwrap());
            check("cube default", matmul(lhs.clone(), rhs.clone(), None, MatmulStrategy::Cube, DType::F32).unwrap());
            #[cfg(feature = "cuda-cublas")]
            {
                let out = fresh();
                match super::super::vendor::launch(lhs.clone(), rhs.clone(), out.clone()) {
                    Ok(()) => check("cublas", out),
                    Err(e) => println!("  cublas         declined: {e}"),
                }
            }
            for tile in [TileMatmulKind::Mma, TileMatmulKind::Cmma] {
                let out = fresh();
                match launch_split_k(lhs.clone(), rhs.clone(), out.clone(), 128, tile) {
                    Ok(()) => check(&format!("split_k {tile:?}"), out),
                    Err(e) => println!("  split_k {tile:?} declined: {e}"),
                }
                // The output autotune hands its candidates.
                let out = super::super::init_matmul_output(&lhs, &rhs, DType::F32);
                let strides = out.meta.strides().to_vec();
                match launch_split_k(lhs.clone(), rhs.clone(), out.clone(), 128, tile) {
                    Ok(()) => check(&format!("split_k {tile:?} matmul-output {strides:?}"), out),
                    Err(e) => println!("  split_k {tile:?} matmul-output declined: {e}"),
                }
            }
        }
        assert_eq!(failures, 0);
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
