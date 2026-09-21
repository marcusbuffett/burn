use super::*;
use burn_tensor::{DType, Tolerance, module::rms_norm};

#[test]
fn rms_norm_gradients_match_decomposition() {
    let device = AutodiffDevice::new();
    for transposed in [false, true] {
        let x = TestTensor::<3>::from_floats(
            [
                [[1., -2., 3.], [4., 0.5, -1.]],
                [[-0.5, 2., 0.], [0.25, -3., 1.]],
            ],
            &device,
        );
        let x = if transposed { x.swap_dims(0, 1) } else { x };
        let x = x.require_grad();
        let gamma = TestTensor::<1>::from_floats([0.5, 1.25, -2.], &device).require_grad();
        let reference_x = x.clone().detach().require_grad();
        let reference_gamma = gamma.clone().detach().require_grad();
        let actual = rms_norm(x.clone(), gamma.clone(), 1e-5);
        let rms = (reference_x.clone().cast(DType::F32).square().mean_dim(2) + 1e-5).sqrt();
        let expected = reference_x.clone() / rms * reference_gamma.clone().unsqueeze();
        actual
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::rel_abs(1e-4, 1e-4));
        let ga = actual.square().sum().backward();
        let ge = expected.square().sum().backward();
        x.grad(&ga).unwrap().to_data().assert_approx_eq::<f32>(
            &reference_x.grad(&ge).unwrap().to_data(),
            Tolerance::rel_abs(2e-3, 2e-3),
        );
        gamma.grad(&ga).unwrap().to_data().assert_approx_eq::<f32>(
            &reference_gamma.grad(&ge).unwrap().to_data(),
            Tolerance::rel_abs(2e-3, 2e-3),
        );
    }
}
