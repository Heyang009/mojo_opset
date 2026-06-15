"""UC wrapper for int8 dequantization."""

import torch

from mojo_opset.core import MojoDequant

from ._utils import _uc_kernels


_API_BY_OUTPUT_DTYPE = {
    torch.bfloat16: "mojo_dequant_bf16",
    torch.float16: "mojo_dequant_fp16",
}
_UC_MIN_NUMEL = 64 * 1024


class UCDequant(MojoDequant):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        api = _API_BY_OUTPUT_DTYPE.get(self.output_dtype)
        if api is None or api not in _uc_kernels():
            raise NotImplementedError(f"UCDequant has no uc-kernel for output dtype {self.output_dtype}.")
        if input.dtype is not torch.int8 or input.dim() < scale.dim() or input.dim() == 0:
            raise NotImplementedError("UCDequant requires int8 input and broadcast scale.")
        if input.numel() < _UC_MIN_NUMEL:
            raise NotImplementedError(f"UCDequant requires at least {_UC_MIN_NUMEL} input elements.")
        if scale.numel() == 0 or scale.dtype is not torch.float32:
            raise NotImplementedError("UCDequant requires non-empty float32 scale.")
        if not input.is_contiguous() or not scale.is_contiguous():
            raise NotImplementedError("UCDequant requires contiguous input and scale.")

        scale_shape = tuple(scale.shape)
        if scale_shape and tuple(input.shape[-len(scale_shape):]) != scale_shape:
            raise NotImplementedError("UCDequant scale must match the trailing input shape.")

        cols = scale.numel()
        rows = input.numel() // cols
        input_2d = input.reshape(rows, cols)
        scale_1d = scale.reshape(-1)
        output_2d = torch.empty((rows, cols), dtype=self.output_dtype, device=input.device)
        _uc_kernels()[api](input_2d, scale_1d, output_2d, rows, cols)
        return output_2d.reshape(input.shape)
