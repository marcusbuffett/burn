use crate::{CubeBackend, tensor::CubeTensor};
use burn_backend::ops::ActivationOps;

impl ActivationOps<Self> for CubeBackend {
    fn swiglu(tensor: CubeTensor) -> CubeTensor {
        crate::kernel::swiglu::launch(tensor, None)
    }
    fn swiglu_backward(tensor: CubeTensor, grad: CubeTensor) -> CubeTensor {
        crate::kernel::swiglu::launch(tensor, Some(grad))
    }
}
