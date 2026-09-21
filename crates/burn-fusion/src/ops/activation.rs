use crate::{
    Fusion, FusionBackend,
    stream::{StreamId, execution::Operation},
};
use burn_backend::{TensorMetadata, ops::ActivationOps, tensor::FloatTensor};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use std::marker::PhantomData;

#[derive(Debug)]
struct SwiGlu<B: FusionBackend> {
    desc: CustomOpIr,
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> Operation<B::FusionRuntime> for SwiGlu<B> {
    fn execute(
        &self,
        handles: &mut HandleContainer<B::Handle>,
    ) -> Result<(), crate::ExecutionError> {
        let input = handles.get_float_tensor::<B>(&self.desc.inputs[0]);
        let output = if self.desc.inputs.len() == 2 {
            let grad = handles.get_float_tensor::<B>(&self.desc.inputs[1]);
            B::swiglu_backward(input, grad)
        } else {
            B::swiglu(input)
        };
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
        Ok(())
    }
}

fn register<B: FusionBackend>(
    input: FloatTensor<Fusion<B>>,
    grad: Option<FloatTensor<Fusion<B>>>,
) -> FloatTensor<Fusion<B>> {
    let client = input.client.clone();
    let mut shape = input.shape();
    let axis = shape
        .num_dims()
        .checked_sub(1)
        .expect("SwiGLU requires at least one axis");
    assert!(
        shape[axis].is_multiple_of(2),
        "Packed SwiGLU requires an even last dimension"
    );
    let backward = grad.is_some();
    if !backward {
        shape[axis] /= 2;
    }
    let output = TensorIr::uninit(client.create_empty_handle(), shape, input.dtype);
    let mut inputs = vec![input.into_ir()];
    if let Some(grad) = grad {
        inputs.push(grad.into_ir());
    }
    let desc = CustomOpIr::new(
        if backward {
            "swiglu_backward"
        } else {
            "swiglu"
        },
        &inputs,
        &[output],
    );
    client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            SwiGlu::<B> {
                desc,
                _backend: PhantomData,
            },
        )
        .output()
}

impl<B: FusionBackend> ActivationOps<Self> for Fusion<B> {
    fn swiglu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        register::<B>(tensor, None)
    }
    fn swiglu_backward(tensor: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        register::<B>(tensor, Some(grad))
    }
}
