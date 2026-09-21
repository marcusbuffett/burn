use crate::{
    kernel::{untile, utils::address_type},
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::{TensorMetadata, cubecl::dtype_to_storage_type};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::layout::linear::{LinearView, LinearViewMut},
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn forward<F: Float, N: Size>(
    input: LinearView<'_, Vector<F, N>>,
    mut output: LinearViewMut<'_, Vector<F, N>>,
    half: usize,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if !output.is_in_bounds(pos) {
        terminate!();
    }
    let index = (pos / half) * half * 2 + pos % half;
    let gate = input.read(index);
    let up = input.read(index + half);
    let sigmoid = Vector::new(F::new(1.0)) / (Vector::new(F::new(1.0)) + (-gate).exp());
    output.write(pos, gate * sigmoid * up);
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn backward<F: Float, N: Size>(
    input: LinearView<'_, Vector<F, N>>,
    grad: LinearView<'_, Vector<F, N>>,
    mut output: LinearViewMut<'_, Vector<F, N>>,
    half: usize,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if !grad.is_in_bounds(pos) {
        terminate!();
    }
    let index = (pos / half) * half * 2 + pos % half;
    let gate = input.read(index);
    let up = input.read(index + half);
    let dy = grad.read(pos);
    let one = Vector::new(F::new(1.0));
    let sigmoid = one / (one + (-gate).exp());
    let activated = gate * sigmoid;
    output.write(index, dy * up * (sigmoid + activated * (one - sigmoid)));
    output.write(index + half, dy * activated);
}

pub(crate) fn launch(input: CubeTensor, grad: Option<CubeTensor>) -> CubeTensor {
    let input = untile(input);
    let grad = grad.map(untile);
    let mut shape = input.shape();
    let axis = shape
        .num_dims()
        .checked_sub(1)
        .expect("SwiGLU requires at least one axis");
    assert!(
        shape[axis].is_multiple_of(2),
        "Packed SwiGLU requires an even last dimension"
    );
    shape[axis] /= 2;
    if let Some(grad) = &grad {
        assert_eq!(grad.shape(), shape);
        assert_eq!(grad.dtype, input.dtype);
    }
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        if grad.is_some() {
            input.shape()
        } else {
            shape.clone()
        },
        input.dtype,
    );
    let mut vector_size = max_vector_size(&input).min(max_vector_size(&output));
    if let Some(grad) = &grad {
        vector_size = vector_size.min(max_vector_size(grad));
    }
    while !shape[axis].is_multiple_of(vector_size as usize) {
        vector_size /= 2;
    }
    let units = shape.num_elements() / vector_size as usize;
    if units == 0 {
        return output;
    }
    let half = shape[axis] / vector_size as usize;
    let client = input.client.clone();
    let dim = CubeDim::new(&client, units);
    let count = calculate_cube_count_elemwise(&client, units, dim);
    let dtype = dtype_to_storage_type(input.dtype);
    unsafe {
        match grad {
            Some(grad) => backward::launch_unchecked(
                &client,
                count,
                dim,
                address_type!(input, grad, output),
                vector_size,
                input.into_linear_view(),
                grad.into_linear_view(),
                output.clone().into_linear_view(),
                half,
                dtype,
            ),
            None => forward::launch_unchecked(
                &client,
                count,
                dim,
                address_type!(input, output),
                vector_size,
                input.into_linear_view(),
                output.clone().into_linear_view(),
                half,
                dtype,
            ),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{TensorData, Tolerance};

    #[test]
    #[ignore = "requires a GPU"]
    fn packed_swiglu_matches_host_for_vectorized_and_strided_inputs() {
        let device = crate::CubeDevice::default();
        for half in [3, 32, 480] {
            for transpose in [false, true] {
                let rows = 7;
                let width = half * 2;
                let values: Vec<f32> = (0..rows * width)
                    .map(|i| (i % 97) as f32 / 8.0 - 6.0)
                    .collect();
                let grads: Vec<f32> = (0..rows * half)
                    .map(|i| (i % 13) as f32 / 4.0 - 1.0)
                    .collect();
                let make = |values: &[f32], cols| {
                    if transpose {
                        let physical: Vec<f32> = (0..rows * cols)
                            .map(|i| values[(i % rows) * cols + i / rows])
                            .collect();
                        crate::ops::swap_dims(
                            crate::ops::from_data(TensorData::new(physical, [cols, rows]), &device),
                            0,
                            1,
                        )
                    } else {
                        crate::ops::from_data(
                            TensorData::new(values.to_vec(), [rows, cols]),
                            &device,
                        )
                    }
                };
                let input = make(&values, width);
                let grad = make(&grads, half);
                let mut expected = vec![0.0; rows * half];
                let mut derivative = vec![0.0; rows * width];
                for row in 0..rows {
                    for col in 0..half {
                        let i = row * width + col;
                        let g = values[i];
                        let u = values[i + half];
                        let sig = 1.0 / (1.0 + (-g).exp());
                        let dy = grads[row * half + col];
                        expected[row * half + col] = g * sig * u;
                        derivative[i] = dy * u * (sig + g * sig * (1.0 - sig));
                        derivative[i + half] = dy * g * sig;
                    }
                }
                crate::ops::into_data_sync(launch(input.clone(), None)).assert_approx_eq::<f32>(
                    &TensorData::new(expected, [rows, half]),
                    Tolerance::rel_abs(1e-4, 1e-4),
                );
                crate::ops::into_data_sync(launch(input, Some(grad))).assert_approx_eq::<f32>(
                    &TensorData::new(derivative, [rows, width]),
                    Tolerance::rel_abs(1e-4, 1e-4),
                );
            }
        }
    }
}
