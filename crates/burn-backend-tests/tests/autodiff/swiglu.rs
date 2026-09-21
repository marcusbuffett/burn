use super::*;
use burn_tensor::{TensorData, Tolerance, activation};

#[test]
fn packed_swiglu_matches_decomposed_forward_and_backward() {
    let device = AutodiffDevice::new();
    let data = TensorData::from([
        [
            [-3.0, 0.0, 2.0, 0.5, -1.0, 4.0],
            [1.0, -2.0, 0.1, -0.7, 2.0, 3.0],
        ],
        [
            [0.2, -0.5, 1.0, 4.0, -3.0, 2.0],
            [-1.0, 3.0, -4.0, 0.1, 0.3, -0.5],
        ],
    ]);
    for transposed in [false, true] {
        let x = TestTensor::<3>::from_data(data.clone(), &device).require_grad();
        let reference = TestTensor::<3>::from_data(data.clone(), &device).require_grad();
        let view = if transposed {
            x.clone().swap_dims(0, 1)
        } else {
            x.clone()
        };
        let reference_view = if transposed {
            reference.clone().swap_dims(0, 1)
        } else {
            reference.clone()
        };
        let actual = activation::swiglu(view);
        let expected = activation::silu(reference_view.clone().slice([0..2, 0..2, 0..3]))
            * reference_view.slice([0..2, 0..2, 3..6]);
        actual.clone().to_data().assert_approx_eq::<FloatElem>(
            &expected.clone().to_data(),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
        let grads = actual.square().mean().backward();
        let reference_grads = expected.square().mean().backward();
        x.grad(&grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<FloatElem>(
                &reference.grad(&reference_grads).unwrap().to_data(),
                Tolerance::rel_abs(1e-4, 1e-4),
            );
    }
}

#[test]
#[ignore = "training-size accelerator benchmark"]
fn packed_swiglu_training_benchmark() {
    let device = AutodiffDevice::new();
    let input = TestTensor::<3>::random(
        [2048, 64, 960],
        burn_tensor::Distribution::Normal(0.0, 0.5),
        &device,
    )
    .require_grad();
    for packed in [false, true] {
        let mut times = Vec::new();
        for iteration in 0..15 {
            let start = std::time::Instant::now();
            let output = if packed {
                activation::swiglu(input.clone())
            } else {
                activation::silu(input.clone().slice([0..2048, 0..64, 0..480]))
                    * input.clone().slice([0..2048, 0..64, 480..960])
            };
            let grads = output.square().mean().backward();
            let checksum = input.grad(&grads).unwrap().sum().into_scalar::<f32>();
            assert!(checksum.is_finite());
            if iteration >= 5 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        println!(
            "swiglu_training packed={packed} ms={:.3}",
            times.iter().sum::<f64>() / times.len() as f64
        );
    }
}
