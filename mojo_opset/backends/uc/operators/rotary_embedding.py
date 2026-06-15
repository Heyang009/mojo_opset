from typing import Optional, Tuple

import torch

from mojo_opset.core import MojoRotaryEmbedding

from ._utils import _uc_kernels


_ROTARY_POSIDS_KERNEL = None


def _get_rotary_posids_kernel():
    global _ROTARY_POSIDS_KERNEL
    if _ROTARY_POSIDS_KERNEL is not None:
        return _ROTARY_POSIDS_KERNEL
    api = "mojo_rotary_embedding_position_ids_fp32"
    kernels = _uc_kernels()
    if api not in kernels:
        raise NotImplementedError(
            f"UC kernel {api!r} is not in the loaded uc-kernel wheel. "
            "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
        )
    _ROTARY_POSIDS_KERNEL = kernels[api]
    return _ROTARY_POSIDS_KERNEL


class UCRotaryEmbedding(MojoRotaryEmbedding):
    supported_platforms_list = ["npu"]

    def __init__(
        self,
        rope_theta,
        rope_dim,
        attention_scaling: float = 1.0,
        init_max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(rope_theta, rope_dim, attention_scaling, init_max_length, **kwargs)
        if init_max_length is None:
            raise ValueError("init_max_length must be provided for UCRotaryEmbedding")

    def forward(
        self,
        x: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cu_q_lens is not None:
            assert cu_q_lens.dtype == torch.int32
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        if position_ids is not None:
            assert position_ids.dtype == torch.int32
        assert position_ids is None or cu_q_lens is None, "At most one of cu_q_lens or position_ids should be provided"

        if cu_q_lens is not None:
            raise NotImplementedError(
                "UCRotaryEmbedding does not support cu_q_lens in the current uc-kernel contract."
            )
        elif position_ids is not None:
            assert position_ids.shape == x.shape[:-1], "position_ids must have the same shape as x except the hidden dimension"
            if not position_ids.is_contiguous():
                raise NotImplementedError("UCRotaryEmbedding requires contiguous position_ids.")
        else:
            raise NotImplementedError(
                "UCRotaryEmbedding requires position_ids; the current uc-kernel wheel does not provide "
                "an arange-cache rotary kernel."
            )

        return self._position_ids_cache(position_ids)

    def _position_ids_cache(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_dim = self.cos.shape[-1]
        rows = position_ids.numel()
        output_shape = tuple(position_ids.shape) + (rope_dim,)

        if rows == 0 or rope_dim == 0:
            cos_out = torch.empty(output_shape, device=self.cos.device, dtype=self.cos.dtype)
            sin_out = torch.empty(output_shape, device=self.sin.device, dtype=self.sin.dtype)
            return cos_out, sin_out

        if self.cos.dtype != torch.float32:
            raise NotImplementedError(
                f"UCRotaryEmbedding._position_ids_cache requires fp32 cos/sin cache, got {self.cos.dtype}. "
                "No UC kernel registered for non-fp32 cache dtype."
            )
        if not self.cos.is_contiguous() or not self.sin.is_contiguous():
            raise NotImplementedError("UCRotaryEmbedding requires contiguous cos/sin caches.")
        if rows % 2 != 0:
            raise NotImplementedError("UCRotaryEmbedding requires an even number of position ids.")

        kernel = _get_rotary_posids_kernel()

        cos_out = torch.empty(output_shape, device=self.cos.device, dtype=self.cos.dtype)
        sin_out = torch.empty(output_shape, device=self.sin.device, dtype=self.sin.dtype)

        flat_pids = position_ids.reshape(-1)
        cos_arg = cos_out.reshape(rows, rope_dim)
        sin_arg = sin_out.reshape(rows, rope_dim)

        kernel(
            self.cos,
            self.sin,
            flat_pids,
            cos_arg,
            sin_arg,
            self.cos.shape[0],
            rope_dim,
            rows,
        )

        return cos_out, sin_out
