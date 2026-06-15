"""UC wrapper for SiLU."""

import torch

from mojo_opset.core import MojoSilu

from ._utils import run_unary_kernel


_UC_MIN_NUMEL = 64 * 1024


class UCSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() < _UC_MIN_NUMEL:
            raise NotImplementedError(f"UCSilu requires at least {_UC_MIN_NUMEL} elements.")
        return run_unary_kernel("mojo_silu", x)
