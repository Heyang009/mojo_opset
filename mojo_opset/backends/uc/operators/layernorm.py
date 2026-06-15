"""UC wrapper for LayerNorm.

Calls ``mojo_layernorm_{bf16,fp16}`` for contiguous inputs and parameters.
"""

import torch

from mojo_opset.core import MojoLayerNorm

from ._utils import _matrix_shape, _uc_kernels


_API = {
    torch.bfloat16: "mojo_layernorm_bf16",
    torch.float16: "mojo_layernorm_fp16",
}


def _resolve_api(dtype: torch.dtype) -> str:
    api = _API.get(dtype)
    if api is None or api not in _uc_kernels():
        raise NotImplementedError(f"UCLayerNorm has no uc-kernel for dtype {dtype}.")
    return api


def _param(param: torch.Tensor, dtype: torch.dtype, name: str) -> torch.Tensor:
    if param.dtype is not dtype or not param.is_contiguous():
        raise NotImplementedError(f"UCLayerNorm requires contiguous {name} with dtype {dtype}.")
    return param


class UCLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None or not self.elementwise_affine:
            raise NotImplementedError("UCLayerNorm requires affine weight and bias.")
        if hidden_state.numel() == 0:
            return torch.empty_like(hidden_state)
        if not hidden_state.is_contiguous():
            raise NotImplementedError("UCLayerNorm requires contiguous input.")

        api = _resolve_api(hidden_state.dtype)
        rows, cols = _matrix_shape(hidden_state)
        output = torch.empty_like(hidden_state)
        weight = _param(self.weight, hidden_state.dtype, "weight")
        bias = _param(self.bias, hidden_state.dtype, "bias")
        _uc_kernels()[api](hidden_state, weight, bias, output, rows, cols, self.variance_epsilon)
        return output.reshape(hidden_state.shape)
