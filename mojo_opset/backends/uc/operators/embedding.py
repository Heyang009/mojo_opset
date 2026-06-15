"""UC wrapper for embedding lookup."""

from typing import Callable, Optional

import torch

from mojo_opset.core import MojoEmbedding

from ._utils import _uc_kernels


_FIXED_VOCAB = 4096
_FIXED_H = 128
_FIXED_NUM_TOKENS = 64

_DTYPE_TO_API = {
    torch.bfloat16: "mojo_embedding_bf16",
    torch.float16: "mojo_embedding_fp16",
}


def _resolve_api(dtype: torch.dtype) -> Optional[Callable]:
    api = _DTYPE_TO_API.get(dtype)
    if api is None:
        return None
    kernels = _uc_kernels()
    if api not in kernels.keys():
        return None
    return kernels[api]


class UCEmbedding(MojoEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Any deviation from the fixed-shape contract raises below.
        api_fn = _resolve_api(self.weight.dtype)
        if (
            api_fn is None
            or self.num_embeddings != _FIXED_VOCAB
            or self.embedding_dim != _FIXED_H
            or input.numel() != _FIXED_NUM_TOKENS
            or self.padding_idx is not None
            or self.max_norm is not None
        ):
            raise NotImplementedError("UCEmbedding input is outside the uc-kernel contract.")

        if not self.weight.is_contiguous() or not input.is_contiguous():
            raise NotImplementedError("UCEmbedding requires contiguous weight and input tensors.")
        if input.dtype is not torch.int32:
            raise NotImplementedError("UCEmbedding requires int32 input ids.")

        weight = self.weight
        input_ids = input.reshape(-1)

        out_flat = torch.empty(
            (_FIXED_NUM_TOKENS, _FIXED_H),
            dtype=weight.dtype,
            device=weight.device,
        )
        api_fn(weight, input_ids, out_flat)
        return out_flat.reshape(*input.shape, _FIXED_H)
