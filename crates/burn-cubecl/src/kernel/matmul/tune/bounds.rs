use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    client::Client, std::throughput::roofline_bounds, throughput::ThroughputKey, tune::TunableSet,
};
use cubek::matmul::{
    definition::{MatmulCost, MatmulElems, MatmulGlobalElems, compute_key},
    multi_level::definition::adjust_dtypes,
};

use crate::{
    kernel::autotune_bounds,
    kernel::matmul::tune::base::{Inputs, MatmulAutotuneKey},
};

type MatmulTunables<Out> = TunableSet<MatmulAutotuneKey, Inputs, Out>;

/// Registers the performance bounds used for matrix multiplication autotuning.
pub(super) fn with_matmul_bounds<Out: 'static>(set: MatmulTunables<Out>) -> MatmulTunables<Out> {
    autotune_bounds::with_bounds(set, |_key, tensors: &Inputs, thresholds| {
        let client = &tensors.0.client;
        let cost = cost(tensors);

        let mut work = cost.work();
        // Broadcast operands are read once, not once per output batch. Keep the
        // existing storage estimate for quantized/tiled tensors.
        if [&tensors.0, &tensors.1, &tensors.2]
            .iter()
            .all(|tensor| tensor.qparams.is_none() && !tensor.meta.tiling.is_tiled())
        {
            work.bytes = [&tensors.0, &tensors.1, &tensors.2]
                .iter()
                .map(|tensor| {
                    unique_elements(tensor.meta.shape(), tensor.meta.strides())
                        * dtype_to_storage_type(tensor.dtype).size()
                })
                .sum();
        }
        roofline_bounds(
            client,
            accelerated_compute_key(client, &cost.elems),
            work,
            thresholds,
        )
    })
}

fn accelerated_compute_key(client: &Client, globals: &MatmulGlobalElems) -> ThroughputKey {
    let mut elems = MatmulElems::from_globals(globals);
    // Use the same register types as the accelerated candidates. In particular,
    // FP32 storage can execute on TF32 tensor cores, far above scalar FP32 peak.
    adjust_dtypes(client, &mut elems, true);
    compute_key(client, &elems)
}

fn unique_elements(shape: &[usize], strides: &[usize]) -> usize {
    shape
        .iter()
        .zip(strides)
        .map(|(&dim, &stride)| if stride == 0 { 1 } else { dim })
        .product()
}

fn cost((lhs, rhs, out): &Inputs) -> MatmulCost {
    let lhs_shape = lhs.meta.shape();
    let rhs_shape = rhs.meta.shape();
    let ndims = lhs_shape.len();

    MatmulCost {
        batches: out.meta.shape()[..out.meta.shape().len() - 2]
            .iter()
            .product(),
        m: lhs_shape[ndims - 2],
        k: lhs_shape[ndims - 1],
        n: rhs_shape[rhs_shape.len() - 1],
        elems: MatmulGlobalElems {
            lhs: dtype_to_storage_type(lhs.dtype),
            rhs: dtype_to_storage_type(rhs.dtype),
            out: dtype_to_storage_type(out.dtype),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_traffic_counts_shared_storage_once() {
        assert_eq!(unique_elements(&[32, 64, 128], &[0, 128, 1]), 64 * 128);
        assert_eq!(
            unique_elements(&[32, 64, 128], &[8192, 128, 1]),
            32 * 64 * 128
        );
        assert_eq!(unique_elements(&[64, 128], &[1, 80]), 64 * 128);
    }

    #[test]
    #[ignore = "requires a CUDA device with TF32 tensor cores"]
    fn fp32_bounds_use_tensor_core_register_throughput() {
        use burn_backend::TensorData;
        use cubecl::{
            ir::{ElemType, FloatKind},
            throughput::ThroughputMode,
        };
        let tensor = crate::ops::from_data(
            TensorData::new([1.0_f32], [1]),
            &crate::CubeDevice::default(),
        );
        let f32 = ElemType::Float(FloatKind::F32);
        let key = accelerated_compute_key(
            &tensor.client,
            &MatmulGlobalElems {
                lhs: f32,
                rhs: f32,
                out: f32,
            },
        );
        assert_eq!(key.dtype(), ElemType::Float(FloatKind::TF32));
        assert!(matches!(key.mode, ThroughputMode::ComputeCmma { .. }));
    }
}
