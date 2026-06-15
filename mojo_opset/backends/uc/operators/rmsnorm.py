"""UC wrapper for RMSNorm.

Calls a shape-specialized RMSNorm uc-kernel when available, otherwise the
dynamic ``mojo_rmsnorm_{bf16,fp16}`` kernel.
"""

import torch

from mojo_opset.core import MojoRMSNorm

from ._utils import _matrix_shape, _uc_kernels


_DYNAMIC_API = {
    torch.bfloat16: "mojo_rmsnorm_bf16",
    torch.float16: "mojo_rmsnorm_fp16",
}

_SHAPE_API = {
    2048: {
        torch.bfloat16: "mojo_rmsnorm_n2048_bf16",
        torch.float16: "mojo_rmsnorm_n2048_fp16",
    },
}


def _resolve_api(cols: int, dtype: torch.dtype) -> str:
    kernels = _uc_kernels()
    api = _SHAPE_API.get(cols, {}).get(dtype)
    if api is not None and api in kernels:
        return api
    api = _DYNAMIC_API.get(dtype)
    if api is None or api not in kernels:
        raise NotImplementedError(f"UCRMSNorm has no uc-kernel for dtype {dtype}.")
    return api


class UCRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.numel() == 0:
            return torch.empty_like(hidden_state)
        if not hidden_state.is_contiguous():
            raise NotImplementedError("UCRMSNorm requires contiguous input.")
        if self.weight.dtype is not hidden_state.dtype or not self.weight.is_contiguous():
            raise NotImplementedError("UCRMSNorm requires contiguous weight with input dtype.")

        rows, cols = _matrix_shape(hidden_state)
        api = _resolve_api(cols, hidden_state.dtype)
        output = torch.empty_like(hidden_state)
        _uc_kernels()[api](hidden_state, self.weight, output, rows, cols, self.variance_epsilon)
        return output.reshape(hidden_state.shape)
