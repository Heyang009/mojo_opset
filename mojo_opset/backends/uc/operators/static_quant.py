import torch

from mojo_opset.core import MojoStaticQuant

from ._utils import run_kernel


class UCStaticQuant(MojoStaticQuant):
    supported_platforms_list = ["npu"]

    # P1-G4: small-shape gate. Below this floor the UC kernel is not part of
    # the supported contract.
    _UC_MIN_NUMEL = 64 * 1024

    # The v2 kernel writes a full X=16-row block per program; M must be
    # a multiple of this on entry. Same value as UCDynamicQuant.
    _UC_ROW_TILE = 16

    def forward(self, input: torch.Tensor):
        if self.quant_dtype != torch.int8:
            raise NotImplementedError(f"UCStaticQuant only supports torch.int8, got {self.quant_dtype}.")
        if input.dim() < len(self.input_size):
            raise ValueError(
                f"input must have at least {len(self.input_size)} dims for scale shape "
                f"{self.input_size}, got {tuple(input.shape)}."
            )
        if tuple(input.shape[-len(self.input_size):]) != self.input_size:
            raise ValueError(
                f"input trailing dims {tuple(input.shape[-len(self.input_size):])} must "
                f"match scale shape {self.input_size}."
            )
        if input.numel() == 0:
            return torch.empty_like(input, dtype=self.quant_dtype), self.scale

        if input.numel() < self._UC_MIN_NUMEL:
            raise NotImplementedError(
                "UCStaticQuant does not support sub-64K-element inputs in the current uc-kernel contract."
            )

        if not input.is_contiguous() or not self.scale.is_contiguous():
            raise NotImplementedError("UCStaticQuant requires contiguous input and scale.")
        if self.scale.device != input.device or self.scale.dtype is not torch.float32:
            raise NotImplementedError("UCStaticQuant requires float32 scale on the input device.")

        kernel_input = input
        scale = self.scale
        cols = scale.numel()
        rows = kernel_input.numel() // cols

        if rows % self._UC_ROW_TILE != 0:
            raise NotImplementedError(
                f"UCStaticQuant requires row count to be a multiple of {self._UC_ROW_TILE}, got {rows}."
            )

        kernel_input_2d = kernel_input.reshape(rows, cols)
        scale_1d = scale.reshape(cols)
        kernel_output = torch.empty_like(kernel_input_2d, dtype=self.quant_dtype)

        run_kernel(
            "mojo_static_quant",
            kernel_input.dtype,
            kernel_input_2d,
            scale_1d,
            kernel_output,
            rows,
            cols,
        )
        return kernel_output.reshape(input.shape), self.scale
