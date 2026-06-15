"""UC wrapper for vision RoPE."""

from typing import Tuple

import torch

from mojo_opset.core import MojoApplyVisionRoPE2D

from ._utils import _uc_kernels, run_kernel


_KERNEL = "mojo_apply_vision_rope_tnh_d64_r64_costoken_bf16"
_API = "mojo_apply_vision_rope_tnh_d64_r64_costoken"
_HEAD_DIM = 64


class UCApplyVisionRoPE2D(MojoApplyVisionRoPE2D):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 3 or k.ndim != 3 or cos.ndim != 2 or sin.ndim != 2:
            raise NotImplementedError("UCApplyVisionRoPE2D requires q/k rank 3 and cos/sin rank 2.")
        if q.dtype is not torch.bfloat16 or k.dtype is not q.dtype:
            raise NotImplementedError("UCApplyVisionRoPE2D requires bf16 q/k.")
        if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
            raise NotImplementedError("UCApplyVisionRoPE2D requires fp32 cos/sin.")
        if q.shape[0] != k.shape[0] or q.shape[0] != cos.shape[0] or cos.shape != sin.shape:
            raise NotImplementedError("UCApplyVisionRoPE2D shape contract mismatch.")
        if q.shape[-1] != _HEAD_DIM or k.shape[-1] != _HEAD_DIM or cos.shape[-1] != _HEAD_DIM:
            raise NotImplementedError(f"UCApplyVisionRoPE2D requires head_dim={_HEAD_DIM}.")
        if not q.is_contiguous() or not k.is_contiguous() or not cos.is_contiguous() or not sin.is_contiguous():
            raise NotImplementedError("UCApplyVisionRoPE2D requires contiguous q, k, cos, and sin tensors.")
        if _KERNEL not in _uc_kernels():
            raise NotImplementedError(f"UCApplyVisionRoPE2D is missing uc-kernel {_KERNEL}.")

        if q.numel() == 0 or k.numel() == 0:
            return torch.empty_like(q), torch.empty_like(k)

        rows, q_heads, _ = q.shape
        _, k_heads, _ = k.shape
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        run_kernel(_API, q.dtype, q, k, cos, sin, q_out, k_out, rows, q_heads, k_heads)
        return q_out, k_out
