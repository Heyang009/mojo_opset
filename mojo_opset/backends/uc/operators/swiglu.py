"""UC wrapper for SwiGLU."""

import torch

from mojo_opset.core import MojoSwiGLU

from ._utils import run_binary_kernel


_UC_MIN_NUMEL = 64 * 1024


class UCSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu"]

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit > 0:
            raise NotImplementedError("UCSwiGLU only supports the unclipped variant.")
        if gate_out.numel() < _UC_MIN_NUMEL:
            raise NotImplementedError(f"UCSwiGLU requires at least {_UC_MIN_NUMEL} elements.")
        return run_binary_kernel("mojo_swiglu", gate_out, up_out)
