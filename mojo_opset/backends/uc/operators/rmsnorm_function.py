import torch

from mojo_opset.core import MojoRMSNormFunction

from ._utils import _matrix_shape
from ._utils import run_kernel


class UCRMSNormFunction(MojoRMSNormFunction):
    """UC autograd wrapper for RMSNorm forward."""
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        normalized_shape = (input.shape[-1],)

        # Save raw tensors for the inherited torch-native backward.  These
        # are the same arguments parent ``MojoRMSNormFunction.forward``
        # would have saved, so backward semantics are identical.
        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        if input.numel() == 0:
            return torch.empty_like(input)

        if input.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"UCRMSNormFunction supports bf16/fp16 only, got {input.dtype}. "
                "No UC kernel registered for this dtype."
            )

        if not input.is_contiguous() or not weight.is_contiguous():
            raise NotImplementedError("UCRMSNormFunction requires contiguous input and weight.")
        if weight.dtype is not input.dtype:
            raise NotImplementedError("UCRMSNormFunction requires weight dtype to match input dtype.")

        rows, cols = _matrix_shape(input)
        kernel_output = torch.empty_like(input)

        run_kernel(
            "mojo_rmsnorm",
            input.dtype,
            input,
            weight,
            kernel_output,
            rows,
            cols,
            eps,
        )
        return kernel_output.reshape(input.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError(
            "UCRMSNormFunction.backward is not implemented by the current uc-kernel wheel."
        )
