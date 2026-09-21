use super::*;
use burn_tensor::module::linear;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_linear_with_bias() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<3>::from_data(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &device,
    )
    .require_grad();
    let weight =
        TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).require_grad();
    let bias = TestTensor::<1>::from_data([0.1, 0.2, 0.3], &device).require_grad();

    let output = linear(x.clone(), weight.clone(), Some(bias.clone()));
    let grads = output.backward();

    let tolerance = Tolerance::default();
    x.grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[[6.0, 15.0], [6.0, 15.0]], [[6.0, 15.0], [6.0, 15.0]]]),
            tolerance,
        );
    weight
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[16.0, 16.0, 16.0], [20.0, 20.0, 20.0]]),
            tolerance,
        );
    bias.grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([4.0, 4.0, 4.0]), tolerance);
}

#[test]
fn should_diff_linear_matches_matmul_decomposition() {
    let device = AutodiffDevice::new();
    let x_data = TensorData::from([[[1.0, -2.0], [3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]]);
    let w_data = TensorData::from([[0.5, -1.0, 2.0], [1.5, 2.5, -0.5]]);

    // Gradients through the linear module op.
    let x = TestTensor::<3>::from_data(x_data.clone(), &device).require_grad();
    let weight = TestTensor::<2>::from_data(w_data.clone(), &device).require_grad();
    let grads = linear(x.clone(), weight.clone(), None).backward();
    let x_grad = x.grad(&grads).unwrap();
    let weight_grad = weight.grad(&grads).unwrap();

    // Gradients through the explicit broadcast matmul decomposition.
    let x_ref = TestTensor::<3>::from_data(x_data, &device).require_grad();
    let weight_ref = TestTensor::<2>::from_data(w_data, &device).require_grad();
    let grads_ref = x_ref
        .clone()
        .matmul(weight_ref.clone().unsqueeze::<3>())
        .backward();

    let tolerance = Tolerance::default();
    x_grad
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
    weight_grad
        .to_data()
        .assert_approx_eq::<FloatElem>(&weight_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
}

#[test]
fn linear_strided_input_weight_only_gradient() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        &device,
    )
    .swap_dims(0, 1);
    let weight = TestTensor::<2>::from_data([[0.5, -1.0], [1.5, 2.0]], &device).require_grad();
    let multiplier = TestTensor::<3>::from_data(
        [
            [[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]],
            [[4.0, 2.0], [5.0, 3.0], [6.0, 4.0]],
        ],
        &device,
    );
    let output = linear(x, weight.clone(), None);
    let grads = (output * multiplier).sum().backward();
    // Sum x[row, in] * multiplier[row, out] over all leading dimensions.
    weight
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[151.0, 79.0], [172.0, 88.0]]),
            Tolerance::default(),
        );
}
