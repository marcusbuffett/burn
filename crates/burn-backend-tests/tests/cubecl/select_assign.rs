use super::*;
use burn_tensor::Distribution;
use burn_tensor::{Device, IndexingUpdateOp, Tolerance};

#[test]
fn select_add_should_match_reference_2d_dim0() {
    select_add_matches_reference(0, [256, 6]);
}

#[test]
fn select_add_should_match_reference_2d_dim1() {
    select_add_matches_reference(1, [6, 256]);
}

#[test]
fn select_add_should_match_reference_3d_dim0() {
    select_add_matches_reference(0, [256, 6, 6]);
}

#[test]
fn select_add_should_match_reference_3d_dim1() {
    select_add_matches_reference(1, [6, 256, 6]);
}

#[test]
fn select_add_should_match_reference_3d_dim2() {
    select_add_matches_reference(2, [6, 6, 256]);
}

#[test]
fn select_mul_should_match_reference_2d_dim0() {
    select_mul_matches_reference(0, [256, 6]);
}

#[test]
fn select_mul_should_match_reference_2d_dim1() {
    select_mul_matches_reference(1, [6, 256]);
}

fn select_add_matches_reference<const D: usize>(dim: usize, shape: [usize; D]) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let value = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let indices = TestTensorInt::<1>::random(
        [shape[dim]],
        Distribution::Uniform(0., shape[dim] as f64),
        &device,
    );
    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<D>::from_data(value.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<1>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.select_assign(dim, indices, value, IndexingUpdateOp::Add);
    let expected = tensor_ref.select_assign(dim, indices_ref, value_ref, IndexingUpdateOp::Add);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

fn select_mul_matches_reference<const D: usize>(dim: usize, shape: [usize; D]) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let value = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let indices = TestTensorInt::<1>::arange(0..shape[dim] as i64, &device);

    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<D>::from_data(value.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<1>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.select_assign(dim, indices, value, IndexingUpdateOp::Mul);
    let expected = tensor_ref.select_assign(dim, indices_ref, value_ref, IndexingUpdateOp::Mul);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

/// Repeated indices and transposed operands exercise the policy-gather backward
/// workload while checking against an independent CPU implementation.
#[test]
fn select_add_repeated_indices_strided_operands() {
    let device = Device::default();
    let reference = ReferenceDevice::new();
    let indices_data: Vec<i32> = (0..257).map(|i| (i * 17 % 67) as i32).collect();
    let indices = TestTensorInt::<1>::from_data(indices_data.as_slice(), &device);
    let input = TestTensor::<2>::random([67, 19], Distribution::Default, &device).swap_dims(0, 1);
    let values = TestTensor::<2>::random([257, 19], Distribution::Default, &device).swap_dims(0, 1);
    let expected = TestTensor::<2>::from_data(input.to_data(), &reference).select_assign(
        1,
        TestTensorInt::<1>::from_data(indices.to_data(), &reference),
        TestTensor::<2>::from_data(values.to_data(), &reference),
        IndexingUpdateOp::Add,
    );
    // Keep an alias alive: the update must not mutate its shared input.
    let alias = input.clone();
    let original = alias.to_data();
    let actual = input.select_assign(1, indices, values, IndexingUpdateOp::Add);
    expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&actual.to_data(), Tolerance::default());
    original.assert_eq(&alias.to_data(), false);
}

/// Explicitly opt in on an idle accelerator. Transfer after every operation
/// provides completion timing rather than timing lazy graph construction.
#[test]
#[ignore]
fn select_add_policy_backward_benchmark() {
    let device = Device::default();
    let indices_data: Vec<i32> = (0..4864).map(|i| (i * 17 % 4096) as i32).collect();
    let indices = TestTensorInt::<1>::from_data(indices_data.as_slice(), &device);
    let value = TestTensor::<2>::ones([2048, 4864], &device);
    let input = TestTensor::<2>::zeros([2048, 4096], &device);
    let mut timings = Vec::new();
    for iteration in 0..15 {
        let start = std::time::Instant::now();
        let output =
            input
                .clone()
                .select_assign(1, indices.clone(), value.clone(), IndexingUpdateOp::Add);
        let sum = output.sum().into_scalar::<f32>();
        assert_eq!(sum, (2048 * 4864) as f32);
        if iteration >= 5 {
            timings.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    println!(
        "select_add_policy_backward_ms={:.3}",
        timings.iter().sum::<f64>() / timings.len() as f64
    );
}
