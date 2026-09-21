use crate::{ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor};
use burn_backend::DType;
use burn_std::Metadata;
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an accelerator"]
    fn partitioned_sum_preserves_strided_axes() {
        use burn_backend::{TensorData, Tolerance, cubecl::dtype_to_elem_type};
        let device = crate::CubeDevice::default();
        let (rows, cols) = (8192, 7);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 3 % 23) as f32 - 11.0) / 16.0)
            .collect();
        for transposed in [false, true] {
            let mut input =
                crate::ops::from_data(TensorData::new(values.clone(), [rows, cols]), &device);
            let (axis, shape) = if transposed {
                let strides = [input.meta.strides()[1], input.meta.strides()[0]];
                *input.meta = Metadata::new([cols, rows], strides);
                (1, [cols, 1])
            } else {
                (0, [1, cols])
            };
            let output = empty_device_contiguous_dtype(
                input.client.clone(),
                device.clone(),
                shape.into(),
                DType::F32,
            );
            let config = ReduceOperationConfig::Sum;
            launch(
                input,
                output.clone(),
                axis,
                config,
                config.precision(dtype_to_elem_type(DType::F32), None),
            )
            .unwrap();
            let expected: Vec<f32> = (0..cols)
                .map(|col| (0..rows).map(|row| values[row * cols + col]).sum())
                .collect();
            TensorData::new(expected, shape).assert_approx_eq::<f32>(
                &crate::ops::into_data_sync(output),
                Tolerance::rel_abs(1e-4, 1e-4),
            );
        }
    }
}

pub(super) fn limit(
    input: &CubeTensor,
    output: &CubeTensor,
    axis: usize,
    config: ReduceOperationConfig,
) -> usize {
    if !matches!(config, ReduceOperationConfig::Sum)
        || input.dtype != DType::F32
        || output.dtype != DType::F32
        || input.meta.tiling.is_tiled()
        || output.meta.tiling.is_tiled()
        || input.qparams.is_some()
        || output.qparams.is_some()
        || input.meta.shape()[axis] < 8192
        || !input.meta.shape()[axis].is_multiple_of(32)
        || output.meta.num_elements() > 1024 * 1024
    {
        1
    } else {
        32
    }
}

/// Divide a long sum into independent reductions, then combine the partials.
/// Both input and final output are views; only the bounded partials allocate.
pub(super) fn launch(
    mut input: CubeTensor,
    mut output: CubeTensor,
    axis: usize,
    config: ReduceOperationConfig,
    dtypes: ReduceDtypes,
) -> Result<(), String> {
    let partitions = limit(&input, &output, axis, config);
    if partitions == 1 {
        return Err("Sum cannot be partitioned".into());
    }
    let mut shape = input.meta.shape().to_vec();
    let mut strides = input.meta.strides().to_vec();
    let chunk = shape[axis] / partitions;
    shape[axis] = partitions;
    shape.insert(axis + 1, chunk);
    let stride = strides[axis];
    strides[axis] = chunk * stride;
    strides.insert(axis + 1, stride);
    *input.meta = Metadata::new(shape.clone(), strides);
    shape[axis + 1] = 1;
    let partial = empty_device_contiguous_dtype(
        output.client.clone(),
        output.device.clone(),
        shape.into(),
        output.dtype,
    );
    let strategy = |routine| ReduceStrategy {
        routine,
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
        autotune_level: Default::default(),
    };
    cubek::reduce::reduce(
        &output.client,
        input.binding(),
        partial.clone().binding(),
        axis + 1,
        strategy(RoutineStrategy::Plane(BlueprintStrategy::Inferred(
            PlaneStrategy { independent: true },
        ))),
        config,
        dtypes,
    )
    .map_err(|e| e.to_string())?;
    let mut shape = output.meta.shape().to_vec();
    let mut strides = output.meta.strides().to_vec();
    shape.insert(axis + 1, 1);
    strides.insert(axis + 1, strides[axis]);
    *output.meta = Metadata::new(shape, strides);
    cubek::reduce::reduce(
        &output.client,
        partial.binding(),
        output.clone().binding(),
        axis,
        strategy(RoutineStrategy::Unit(BlueprintStrategy::Inferred(
            UnitStrategy,
        ))),
        config,
        dtypes,
    )
    .map_err(|e| e.to_string())
}
