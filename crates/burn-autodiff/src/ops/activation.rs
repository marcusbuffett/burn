use core::marker::PhantomData;

use crate::{
    Autodiff,
    checkpoint::{
        base::Checkpointer, retro_forward::RetroForward, state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    graph::NodeId,
    ops::{Backward, Ops, OpsKind, unary},
    retro_unary,
};
use burn_backend::{Backend, ops::ActivationOps, tensor::FloatTensor};

impl<B: Backend, C: CheckpointStrategy> ActivationOps<Autodiff<B, C>> for Autodiff<B, C> {
    fn softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        softmax_with_saved_output::<B, C>(tensor, dim, false)
    }

    fn log_softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        softmax_with_saved_output::<B, C>(tensor, dim, true)
    }

    fn swiglu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SwiGlu;
        retro_unary!(RetroSwiGlu, B::swiglu);
        impl<B: Backend> Backward<B, 1> for SwiGlu {
            type State = NodeId;
            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::swiglu_backward(input, grad)
                });
            }
        }
        match SwiGlu
            .prepare::<C>([tensor.node()])
            .memory_bound()
            .retro_forward(RetroSwiGlu::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::swiglu(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::swiglu(tensor.primitive)),
        }
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Gelu;

        retro_unary!(RetroGelu, B::gelu);

        impl<B: Backend> Backward<B, 1> for Gelu {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::gelu_backward(input, grad)
                });
            }
        }

        match Gelu
            .prepare::<C>([tensor.node()])
            .memory_bound()
            .retro_forward(RetroGelu::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::gelu(tensor.primitive.clone()))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::gelu(tensor.primitive)),
        }
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Relu;

        retro_unary!(RetroRelu, B::relu);

        impl<B: Backend> Backward<B, 1> for Relu {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let state = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::relu_backward(state, grad)
                });
            }
        }

        match Relu
            .prepare::<C>([tensor.node()])
            .memory_bound()
            .retro_forward(RetroRelu::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::relu(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::relu(tensor.primitive)),
        }
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sigmoid;

        retro_unary!(RetroSigmoid, B::sigmoid);

        impl<B: Backend> Backward<B, 1> for Sigmoid {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                let output = B::sigmoid(input);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::sigmoid_backward(output, grad)
                });
            }
        }

        match Sigmoid
            .prepare::<C>([tensor.node()])
            .memory_bound()
            .retro_forward(RetroSigmoid::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::sigmoid(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::sigmoid(tensor.primitive)),
        }
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LogSigmoid;

        retro_unary!(RetroLogSigmoid, B::log_sigmoid);

        impl<B: Backend> Backward<B, 1> for LogSigmoid {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::log_sigmoid_backward(input, grad)
                });
            }
        }

        match LogSigmoid
            .prepare::<C>([tensor.node()])
            .memory_bound()
            .retro_forward(RetroLogSigmoid::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::log_sigmoid(tensor.primitive.clone()))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::log_sigmoid(tensor.primitive)),
        }
    }
}

/// Keep only the normalized output. The closed-form derivatives avoid recording
/// and differentiating every operation in the stable forward decomposition.
fn softmax_with_saved_output<B: Backend, C: CheckpointStrategy>(
    tensor: FloatTensor<Autodiff<B, C>>,
    dim: usize,
    logarithmic: bool,
) -> FloatTensor<Autodiff<B, C>> {
    #[derive(Debug)]
    struct Softmax;
    impl<B: Backend> Backward<B, 1> for Softmax {
        type State = (FloatTensor<B>, usize, bool);
        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let (output, dim, logarithmic) = ops.state;
            unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                if logarithmic {
                    let sum = B::float_sum_dim(grad.clone(), dim);
                    B::float_sub(grad, B::float_mul(B::float_exp(output), sum))
                } else {
                    let product = B::float_mul(grad, output.clone());
                    let sum = B::float_sum_dim(product.clone(), dim);
                    B::float_sub(product, B::float_mul(output, sum))
                }
            });
        }
    }
    let prepared = Softmax
        .prepare::<C>([tensor.node()])
        .compute_bound()
        .stateful();
    let output = if logarithmic {
        B::log_softmax(tensor.primitive, dim)
    } else {
        B::softmax(tensor.primitive, dim)
    };
    match prepared {
        OpsKind::Tracked(prep) => prep.finish((output.clone(), dim, logarithmic), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}
