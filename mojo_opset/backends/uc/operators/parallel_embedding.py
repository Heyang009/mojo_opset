"""UC backend for ``MojoParallelEmbedding``.

``MojoParallelEmbedding`` (``mojo_opset/core/operators/embedding.py``) is a
vocabulary-parallel embedding: the embedding table is sharded along the
``num_embeddings`` (vocab) axis, every rank performs a local lookup that
zeros out-of-range indices, and an HCCL ``all_reduce(SUM)`` assembles the
final result across the TP group.

Strategy for the UC backend (P1-G2 2026-06-11 perf re-baseline)
================================================================

* **Wrapper-level optimisations** (the actual win source):

    - **Single-rank fast-path** (``world_size == 1``): skip the parent's
      ``input - 0`` / range-mask / clamp / ``output * 1`` chain; with no
      TP these all collapse to no-ops but cost 5 extra host->NPU launches
      per ``forward()``. Doing one direct lookup instead of six op
      launches is the dominant single-rank win.

    - **TP / multi-rank path**: shift / clamp / mask on the host (the
      lifter v0.3 ``T.Parallel`` allowlist does not include comparisons,
      so the mask must stay on the host -- see lessons § A.1), then call
      the lookup, then multiply by the boolean range mask and
      ``all_reduce(SUM)``.

* **Lookup primitive** -- two-tier:

    1. **UC kernel** ``mojo_parallel_embedding_h<HT>_<dtype>`` (block
       GATHER with dynamic ``(V, H, N)``, compile-time inner ``HT`` in
       ``{128, 4096}``); enabled by ``_is_kernel_profitable()``.
    2. **`F.embedding` / `aclnnEmbedding`** (NPU vendor primitive) for
       all shapes the UC kernel cannot beat. This is **not** a torch
       fallback in the "wheel-missing" sense -- the wheel ships the
       kernel; ``aclnnEmbedding`` is simply the canonical NPU embedding
       lookup that the TTX / ``torch_npu`` / ``torch`` backends (none of
       which override ``MojoParallelEmbedding``) all dispatch to via the
       parent's ``F.embedding`` call. Picking the faster of the two for
       a given shape is the wrapper's job.

* **Hard guards** (truly unsupported configurations) still raise:
  ``max_norm`` (would mutate weight), unsupported weight dtype, index
  dtype, device mismatch.

P1-G2 perf model
================

After the 2026-06-11 ``perf(uc-kernel/runtime)`` cache fix (commit
``825d888``) which dropped the per-call wrapper floor from ~458 us to
~16 us, the UC parallel-embedding kernel's true device cost emerged
(910B, device 3, ``torch.npu.Event``, bf16 unless noted):

    H     N      UC us  torch us  UC/torch
    128   1      20.5    6.0       3.4x
    128   1024   21.1    6.8       3.1x
    128   8192   92.7    7.8      11.9x
    4096  1      17.6    6.7       2.6x
    4096  32     68.3    5.5      12.4x
    4096  1024   1097.8  8.3     131.8x
    4096  8192   3665.5 25.2     145.7x

The UC kernel's per-DMA setup cost in unic-generated CCE scales linearly
with payload size (P2-16 sweep: HT=128 ~ HT=4096 at same payload, so
amortisation by bigger gathers is impossible), making it dominated by
``ceil(N/48) * H`` for any non-trivial shape. ``aclnnEmbedding`` uses
specialised AscendC gather primitives that we cannot reproduce through
the lifter's "single runtime index dim" block-GATHER recogniser
(``tilelang_uc/uir/lowering/kernel.py`` ``_synthesize_indirect_offset``;
also lessons § A.3).

Conclusion: with the runtime fix in place, the UC kernel still does not
beat ``aclnnEmbedding`` at any measured shape. ``_KERNEL_BUDGET_PRODUCT``
is therefore set to ``0`` -- the gate is kept (rather than ripped) so
that if a future unic / lifter improvement closes the per-DMA-setup gap
(see ``parallel-embedding-p2-16.md`` § 5 cannot-optimize table), bumping
one constant re-enables the kernel without re-plumbing the wrapper.

Wheel ABI: ``mojo_parallel_embedding_h<HT>_<dtype>(weight, indices, out,
V, H, N)`` -- the trailing INT32 scalar order follows the
"first-occurrence in type annotations" rule (lessons § B.1).
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mojo_opset.core import MojoParallelEmbedding

from ._utils import _DTYPE_API_SUFFIX
from ._utils import _uc_kernels


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)

# Inner H-tile sizes the kernel ships, biggest first; the wrapper picks
# the largest one that divides ``H``. Keep in lockstep with
# ``uc-kernel/kernels/mojo_parallel_embedding.py``.
_HT_VARIANTS = (4096, 128)

# Profitability budget for ``_is_kernel_profitable``. With the
# 2026-06-11 runtime cache fix (uc-kernel commit ``825d888``) the UC
# block-GATHER kernel's true cost emerged at ~17 us floor + ~0.4 us /
# token at H=4096, compared to ``aclnnEmbedding``'s ~6-25 us across all
# shapes. **No measured shape sees UC win**, so the budget is set to 0
# (kernel never selected). The gate is *kept rather than ripped* because
# the kernel cost is dominated by unic's per-DMA setup overhead and a
# future compiler fix (per ``parallel-embedding-p2-16.md`` § 5 P0) would
# re-enable a window where UC beats vendor; updating one constant beats
# re-plumbing the wrapper.
_KERNEL_BUDGET_PRODUCT = 0


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _as_long_indices(idx: torch.Tensor) -> torch.Tensor:
    """``F.embedding`` -> ``aclnnEmbedding`` on NPU requires int64 indices.

    The wrapper accepts int32 inputs (the UC kernel path casts to int32
    internally), so we must promote here when we route to ``F.embedding``.
    Skip the device->device copy when the input is already int64.
    """
    return idx if idx.dtype == torch.int64 else idx.long()


def _pick_kernel_api(embedding_dim: int, dtype: torch.dtype) -> Optional[str]:
    """Return the largest-HT registered API whose HT divides ``H``."""
    suffix = _DTYPE_API_SUFFIX.get(dtype)
    if suffix is None:
        return None
    try:
        kernels = _uc_kernels()
    except Exception:
        return None
    for ht in _HT_VARIANTS:
        if embedding_dim % ht != 0:
            continue
        api = f"mojo_parallel_embedding_h{ht}_{suffix}"
        if api in kernels.keys():
            return api
    return None


def _is_kernel_profitable(num_tokens: int, embedding_dim: int) -> bool:
    """Decide whether the dedicated UC kernel is expected to beat
    ``aclnnEmbedding`` for this shape.

    The UC block-GATHER cost grows roughly as ``ceil(N / 48) * H * c``
    where ``c`` is the unic-generated per-DMA setup latency (~0.1 us at
    H=128, ~0.4 us / token at H=4096; P2-16 + P1-G2 measurements). With
    current compiler / lifter limits ``c`` is large enough that
    ``aclnnEmbedding`` (~6-25 us across all measured shapes) wins
    everywhere, so the active budget is ``0``. Keep the predicate as
    structural plumbing for the future compiler-fix scenario.
    """
    if num_tokens <= 0:
        return False
    if _KERNEL_BUDGET_PRODUCT <= 0:
        return False
    programs = 48
    return ((num_tokens + programs - 1) // programs) * embedding_dim <= _KERNEL_BUDGET_PRODUCT


class UCParallelEmbedding(MojoParallelEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------------
        # Hard guards.  Configurations the UC backend genuinely cannot
        # honour (would mutate weight in-place; the wheel cannot compute
        # the right answer for a fundamentally different dtype contract)
        # raise instead of producing a wrong result.
        # ------------------------------------------------------------------
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"UCParallelEmbedding expects a torch.Tensor input, got {type(input)}.")
        if input.dtype not in _SUPPORTED_INDEX_DTYPES:
            raise NotImplementedError(
                f"UCParallelEmbedding supports int32/int64 indices, got {input.dtype}."
            )
        if self.weight.dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError(
                f"UCParallelEmbedding supports bf16/fp16 weights, got {self.weight.dtype}."
            )
        if self.weight.device != input.device:
            raise ValueError(
                f"UCParallelEmbedding requires weight and indices on the same device, "
                f"got weight={self.weight.device} input={input.device}."
            )
        if self.embedding_dim <= 0 or self.local_num_embeddings <= 0:
            raise ValueError(
                f"UCParallelEmbedding has invalid shape: embedding_dim={self.embedding_dim}, "
                f"local_num_embeddings={self.local_num_embeddings}."
            )
        if self.max_norm is not None:
            raise NotImplementedError(
                "UCParallelEmbedding does not implement max_norm (would mutate weight in-place); "
                "the UC kernel path cannot reproduce that semantics."
            )

        # Lookup primitive: dedicated UC kernel iff it's expected to beat
        # ``aclnnEmbedding`` at this shape. See ``_is_kernel_profitable``
        # for the rationale; currently always False, the wrapper just
        # uses ``F.embedding`` which lowers to ``aclnnEmbedding`` on NPU.
        api = _pick_kernel_api(self.embedding_dim, self.weight.dtype)
        use_uc_kernel = api is not None and _is_kernel_profitable(
            input.numel(), self.embedding_dim
        )

        # ------------------------------------------------------------------
        # Single-rank fast-path.  When the global vocab equals the local
        # shard (no TP) and ``torch.distributed`` is not initialised the
        # parent's shift / range-mask / clamp / multiply / all_reduce all
        # collapse to a no-op; running them costs 5 extra host->NPU
        # launches we can skip.
        # ------------------------------------------------------------------
        single_rank = (
            self.vocab_start_index == 0
            and self.local_num_embeddings == self.num_embeddings
            and not _is_dist_initialized()
        )

        if single_rank:
            if use_uc_kernel:
                return self._gather(api, input)
            # ``F.embedding`` -> ``aclnnEmbedding`` on NPU **requires** int64
            # indices; passing int32 silently triggers ``SUSPECT MEM ERROR``
            # 507055 (the vendor lib reads 8 bytes per index off a 4-byte
            # buffer). The UC kernel path (above) does its own int32 cast
            # via ``_gather``, so int32 inputs are only a problem here.
            return F.embedding(_as_long_indices(input), self.weight)

        # ------------------------------------------------------------------
        # TP / multi-rank path: shift / mask / clamp on the host (lifter
        # A.1 forbids comparisons inside the kernel), then look up, then
        # zero out-of-range rows and all_reduce.
        # ------------------------------------------------------------------
        local_input = input - self.vocab_start_index
        in_range = (local_input >= 0) & (local_input < self.local_num_embeddings)
        masked_input = local_input.clamp(0, self.local_num_embeddings - 1)

        if use_uc_kernel:
            output = self._gather(api, masked_input)
        else:
            output = F.embedding(_as_long_indices(masked_input), self.weight)

        # Zero contributions from out-of-range indices.
        output = output * in_range.unsqueeze(-1).to(output.dtype)

        if _is_dist_initialized():
            world_size = dist.get_world_size(group=self.process_group)
            if world_size > 1:
                dist.all_reduce(
                    output, op=dist.ReduceOp.SUM, group=self.process_group
                )
        return output

    # ----------------------------------------------------------------------
    # UC kernel call helper.  Only reached when ``_is_kernel_profitable``
    # has approved the shape, so any missing-API state at this point is a
    # consistency bug (the picker found a matching API but the registry
    # somehow lost it between calls) and is loud-raised.
    # ----------------------------------------------------------------------
    def _gather(self, api: str, indices: torch.Tensor) -> torch.Tensor:
        weight = self.weight.contiguous()
        dtype = weight.dtype

        kernels = _uc_kernels()
        if api not in kernels:
            raise NotImplementedError(
                f"UC kernel {api!r} is not in the loaded uc-kernel wheel. "
                "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
            )
        kernel = kernels[api]

        indices_flat = indices.reshape(-1).to(torch.int32).contiguous()
        rows = indices_flat.numel()
        out_shape = tuple(indices.shape) + (self.embedding_dim,)
        if rows == 0:
            return torch.empty(out_shape, dtype=dtype, device=weight.device)

        flat_out = torch.empty(
            (rows, self.embedding_dim),
            dtype=dtype,
            device=weight.device,
        )
        # Trailing INT32 ABI: (V, H, N) -- "first-occurrence" rule.
        kernel(
            weight,
            indices_flat,
            flat_out,
            self.local_num_embeddings,
            self.embedding_dim,
            rows,
        )

        return flat_out.reshape(*indices.shape, self.embedding_dim)
