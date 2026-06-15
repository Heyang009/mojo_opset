"""UC wrapper for residual-add LayerNorm.

Calls ``mojo_residual_add_layernorm_{bf16,fp16}`` or its N=2048 variant for
contiguous inputs and parameters.
"""

import torch

from mojo_opset.core import MojoResidualAddLayerNorm

from ._utils import _matrix_shape, _uc_kernels


_DYNAMIC_API = {
    torch.bfloat16: "mojo_residual_add_layernorm_bf16",
    torch.float16: "mojo_residual_add_layernorm_fp16",
}

_SHAPE_API = {
    2048: {
        torch.bfloat16: "mojo_residual_add_layernorm_n2048_bf16",
        torch.float16: "mojo_residual_add_layernorm_n2048_fp16",
    },
}


def _resolve_api(cols: int, dtype: torch.dtype) -> str:
    kernels = _uc_kernels()
    api = _SHAPE_API.get(cols, {}).get(dtype)
    if api is not None and api in kernels:
        return api
    api = _DYNAMIC_API.get(dtype)
    if api is None or api not in kernels:
        raise NotImplementedError(f"UCResidualAddLayerNorm has no uc-kernel for dtype {dtype}.")
    return api


def _param(param: torch.Tensor, dtype: torch.dtype, name: str) -> torch.Tensor:
    if param.dtype is not dtype or not param.is_contiguous():
        raise NotImplementedError(f"UCResidualAddLayerNorm requires contiguous {name} with dtype {dtype}.")
    return param


class UCResidualAddLayerNorm(MojoResidualAddLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            raise NotImplementedError("UCResidualAddLayerNorm requires residual.")
        if hidden_state.shape != residual.shape or hidden_state.dtype is not residual.dtype:
            raise NotImplementedError("UCResidualAddLayerNorm requires matching input and residual.")
        if self.weight is None or self.bias is None:
            raise NotImplementedError("UCResidualAddLayerNorm requires weight and bias.")
        if hidden_state.numel() == 0:
            empty = torch.empty_like(hidden_state)
            return empty, empty
        if not hidden_state.is_contiguous() or not residual.is_contiguous():
            raise NotImplementedError("UCResidualAddLayerNorm requires contiguous input and residual.")

        rows, cols = _matrix_shape(hidden_state)
        api = _resolve_api(cols, hidden_state.dtype)
        output = torch.empty_like(hidden_state)
        residual_out = torch.empty_like(hidden_state)
        weight = _param(self.weight, hidden_state.dtype, "weight")
        bias = _param(self.bias, hidden_state.dtype, "bias")
        _uc_kernels()[api](
            hidden_state,
            residual,
            weight,
            bias,
            output,
            residual_out,
            rows,
            cols,
            self.variance_epsilon,
        )
        if self.norm_pos == "pre":
            return output.reshape(hidden_state.shape), residual_out.reshape(hidden_state.shape)
        return output.reshape(hidden_state.shape), output.reshape(hidden_state.shape)
