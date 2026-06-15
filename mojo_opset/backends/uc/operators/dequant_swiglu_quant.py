"""UC wrapper for fused dequant, SwiGLU, and dynamic quantization."""

import torch

from mojo_opset.core import MojoDequantSwiGLUQuant

from ._utils import _uc_kernels


_KERNEL_API = "mojo_dequant_swiglu_quant_bf16"
_KERNEL_INNER_Y = 512
_KERNEL_ROW_TILE = 16
_UC_MIN_NUMEL = 64 * 1024


class UCDequantSwiGLUQuant(MojoDequantSwiGLUQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        activation_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        quant_offset: torch.Tensor = None,
        token_count: torch.Tensor = None,
    ):
        if (
            x.dtype is not torch.int8
            or x.dim() != 2
            or x.shape[-1] % 2 != 0
            or self.hidden_size <= _KERNEL_INNER_Y
            or activation_scale is not None
            or bias is not None
            or quant_offset is not None
            or token_count is not None
            or self.activate_left
            or self.quant_dtype is not torch.int8
            or self.quant_mode != 1
            or self.weight_scale.shape[0] != 1
            or self.quant_scale.shape[0] != 1
        ):
            raise NotImplementedError("UCDequantSwiGLUQuant input is outside the uc-kernel contract.")
        if x.numel() < _UC_MIN_NUMEL:
            raise NotImplementedError(f"UCDequantSwiGLUQuant requires at least {_UC_MIN_NUMEL} input elements.")
        if x.shape[0] % _KERNEL_ROW_TILE != 0:
            raise NotImplementedError(f"UCDequantSwiGLUQuant requires rows to be a multiple of {_KERNEL_ROW_TILE}.")
        if not x.is_contiguous():
            raise NotImplementedError("UCDequantSwiGLUQuant requires contiguous input.")
        if self.weight_scale.dtype is not torch.float32 or self.quant_scale.dtype is not torch.float32:
            raise NotImplementedError("UCDequantSwiGLUQuant requires float32 scale tensors.")

        kernels = _uc_kernels()
        if _KERNEL_API not in kernels:
            raise NotImplementedError(f"UCDequantSwiGLUQuant is missing uc-kernel {_KERNEL_API}.")

        rows, cols = x.shape
        half = cols // 2
        weight_scale = self.weight_scale[0]
        quant_scale = self.quant_scale[0]
        if not weight_scale.is_contiguous() or not quant_scale.is_contiguous():
            raise NotImplementedError("UCDequantSwiGLUQuant requires contiguous scale rows.")

        y = torch.empty((rows, half), dtype=torch.int8, device=x.device)
        scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
        kernels[_KERNEL_API](x, weight_scale, quant_scale, y, scale, rows, cols, half)
        return y, scale.unsqueeze(-1)
