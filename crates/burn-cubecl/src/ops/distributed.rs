use burn_backend::distributed::DistributedOps;

use crate::CubeBackend;

#[cfg(feature = "std")]
use crate::ops::numeric::{self, zeros_client};
#[cfg(feature = "std")]
use burn_backend::{
    DeviceId, TensorMetadata,
    cubecl::{dtype_to_elem_type, reduce_operation_to_cubecl},
    distributed::{CollectiveTensor, ReduceOperation},
    tensor::{Device, FloatTensor},
};

impl DistributedOps<Self> for CubeBackend {
    #[cfg(feature = "std")]
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        let device = &tensor.device.clone();
        // NCCL reduces the handle's byte range. A handle may be padded past
        // the tensor (allocation rounding), by amounts that need not agree
        // across ranks, so the range is trimmed to the tensor below; a handle
        // shorter than the tensor is a view and must be copied.
        let bytes = (tensor.shape().num_elements() * tensor.dtype().size()) as u64;
        let usable = tensor.handle.can_mut()
            && tensor.is_contiguous()
            && tensor.handle.size_in_used() >= bytes;
        let out_tensor = if usable {
            tensor
        } else {
            // A copy here launches work (and may allocate) between collective
            // launches, which can deadlock peers already spinning in NCCL;
            // callers should hand over exclusive buffers.
            static COPIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 || n.is_power_of_two() {
                eprintln!(
                    "all_reduce: copying input #{n} (shared={} contiguous={} bytes={} of {bytes}) shape={:?}",
                    !tensor.handle.can_mut(),
                    tensor.is_contiguous(),
                    tensor.handle.size_in_used(),
                    tensor.shape()
                );
            }
            let zeros_tensor = zeros_client(
                tensor.client.clone(),
                device.clone(),
                tensor.shape(),
                tensor.dtype(),
            );
            numeric::add(zeros_tensor, tensor)
        };
        let padding = out_tensor.handle.size_in_used() - bytes;
        let range = if padding > 0 {
            out_tensor.handle.clone().offset_end(padding)
        } else {
            out_tensor.handle.clone()
        };

        let op = reduce_operation_to_cubecl(op);

        let mut client = device.client();
        client.all_reduce(
            range.clone(),
            range,
            dtype_to_elem_type(out_tensor.dtype),
            device_ids.clone(),
            op,
        );
        CollectiveTensor::new(out_tensor)
    }

    #[cfg(feature = "std")]
    fn sync_collective(device: &Device<Self>) {
        let client = device.client();
        client.sync_collective();
    }
}
