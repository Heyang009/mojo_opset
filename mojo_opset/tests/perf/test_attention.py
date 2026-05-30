import math

from typing import Optional

import pytest
import torch

from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoSdpa
from mojo_opset import MojoPagedPrefillSWA
from mojo_opset import MojoPagedDecodeSWA
from mojo_opset import MojoSWA
from mojo_opset.experimental import MojoPagedPrefillGQAWithKVDequant
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


def generate_paged_decode_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)

    total_seq_lens = torch.randint(1, max_seq_len, (batch_size,), dtype=torch.int32)

    max_num_blocks_per_seq = (total_seq_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(total_seq_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = total_seq_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, total_seq_lens, block_tables


test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 8192, 128, torch.bfloat16, "M_BF16_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_decode_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_attn_decode = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    perf(  # noqa: F821
        lambda: paged_attn_decode(
            query,
            k_cache,
            v_cache,
            total_seq_lens,
            block_tables,
            softmax_scale=softmax_scale,
        )
    )


def generate_paged_prefill_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    k_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    v_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = (kv_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        start_loc = cu_total_seq_lens[i].item()

        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

        k_seq = k_unpadded[start_loc : start_loc + seq_len]
        v_seq = v_unpadded[start_loc : start_loc + seq_len]
        for j in range(num_blocks_for_seq):
            physical_block_id = assigned_blocks[j]
            start_pos_in_seq = j * block_size
            tokens_in_block = min(block_size, seq_len - start_pos_in_seq)

            k_slice = k_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)
            v_slice = v_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)

            k_cache[physical_block_id, :, :tokens_in_block, :] = k_slice
            v_cache[physical_block_id, :, :tokens_in_block, :] = v_slice

    cu_total_seq_lens = None if kv_cache_lens is None else torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0).to(torch.int32)]
    )
    return query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens


test_configs_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 4096, 8192, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens",
    [
        pytest.param(
            *generate_paged_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_prefill_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    cu_total_seq_lens: Optional[torch.Tensor],
):
    paged_attn_prefill = MojoPagedPrefillGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: paged_attn_prefill(
            query,
            k_cache,
            v_cache,
            cu_q_lens,
            block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cu_total_seq_lens,
        )
    )


def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
):
    query = torch.randn(bsz, q_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=False)
    key = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=False)
    value = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=False)
    blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool, requires_grad=False)
    return query, key, value, blockwise_diffusion_attn_mask, q_head_num != kv_head_num

def generate_paged_prefill_quant_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    """Generate prefill test data; KV cache kept in float dtype and quantized inside the test."""
    if max_q_len > 0:
        q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
        q_lens = _make_varlen_positive_int32(q_lens, max_q_len)
    else:
        q_lens = torch.randperm(batch_size, dtype=torch.int32)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    if max_kv_computed_len <= 0:
        cu_total_seq_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(
            max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32
        )
        kv_cache_lens = _make_varlen_positive_int32(kv_cache_lens, max_kv_computed_len)
        kv_lens = q_lens + kv_cache_lens
        kv_lens = torch.where(q_lens > 0, kv_lens, torch.zeros_like(kv_lens))
        cu_total_seq_lens = torch.cat(
            [torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)]
        )

    total_q_tokens = cu_q_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = max(1, (kv_lens.max().item() + block_size - 1) // block_size)
    total_blocks_needed = int(torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        if num_blocks_for_seq == 0:
            continue
        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    max_q_lens = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
    max_total_seq_lens = int(kv_lens.max().item()) if kv_lens.numel() > 0 else 0
    return query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_lens, max_total_seq_lens

def _quantize_query(
    query: torch.Tensor,
    query_dtype: torch.Tensor,
):
    if query_dtype == torch.int8:
        int_dtype = torch.int8
        qmax = 2 ** (8 - 1) - 1
        qmin = -(2 ** (8 - 1))
    else:
        assert query_dtype == torch.bfloat16
        return query.to(query_dtype), None
    
    query_f = query.float()
    amax = query_f.abs().amax(dim=-1, keepdim=True)  # -> [num_tokens, num_q_heads, 1]
    qscale = (amax / qmax).clamp(min=1e-5)
    quant = torch.round(query_f / qscale).clamp(qmin, qmax).to(int_dtype)
    return quant, qscale.to(torch.bfloat16)

def _quantize_kv_cache(
    cache: torch.Tensor,  # [n_blocks, n_kv_heads, block_size, head_dim] in float dtype
    context_dtype: torch.dtype,
):
    """Per-channel dynamic quantize a float KV cache along the head_dim axis.

    Returns:
        quant_cache: integer tensor with dtype matching `context_dtype`, same shape as input.
        qscale: per-channel scale of shape (n_kv_heads, head_dim) in the input dtype.
    """
    if context_dtype == torch.int8:
        int_dtype = torch.int8
        qmax = 2 ** (8 - 1) - 1
        qmin = -(2 ** (8 - 1))
    else:
        assert False, f"Context dtype {context_dtype} not supported yet"
    # amax over (n_blocks, block_size) per (head, dim) channel
    cache_f = cache.float()
    amax = cache_f.abs().amax(dim=(0, 2))  # -> [n_kv_heads, head_dim]
    qscale = (amax / qmax).clamp(min=1e-5)
    quant = torch.round(cache_f / qscale.unsqueeze(0).unsqueeze(2)).clamp(qmin, qmax).to(int_dtype)
    return quant, qscale.to(torch.bfloat16)

def _make_varlen_positive_int32(lengths: torch.Tensor, upper_bound: int) -> torch.Tensor:
    """Make per-batch lengths explicitly varlen while keeping them positive."""
    lengths = lengths.to(torch.int32).clone()
    if lengths.numel() <= 1 or upper_bound <= 1:
        return torch.clamp(lengths, min=1)

    offsets = torch.arange(lengths.numel(), dtype=torch.int32)
    span = max(upper_bound - 1, 1)
    lengths = ((lengths + offsets) % span) + 1
    return lengths

@pytest.mark.parametrize(
    "query, key, value, blockwise_diffusion_attn_mask, enable_gqa",
    [
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=8,
                kv_head_num=2,
                head_dim=128,
                seq_length=8192,
            )
        ),
    ],
)
@auto_switch_platform(set_perf=True)
def test_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blockwise_diffusion_attn_mask: torch.Tensor,
    enable_gqa: bool,
):
    diffusion_attn = MojoSdpa(
        scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    perf(lambda: diffusion_attn(query, key, value, blockwise_diffusion_attn_mask))  # noqa: F821


test_configs_swa_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 16, 4, 128, 1024, 8192, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens",
    [
        pytest.param(
            *generate_paged_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_swa_prefill
    ],
)
@pytest.mark.parametrize("gqa_layout, global_window, local_window", [
    ("AABB", 4, 1023),
])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_prefill_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    cu_total_seq_lens: Optional[torch.Tensor],
    global_window: int,
    local_window: int,
):
    swa_prefill = MojoPagedPrefillSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: swa_prefill(
            query,
            k_cache,
            v_cache,
            cu_q_lens,
            block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cu_total_seq_lens,
        )
    )




test_configs_swa_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 16, 4, 128, 8192, 128, torch.bfloat16, "M_BF16_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_swa_decode
    ],
)
@pytest.mark.parametrize("gqa_layout, global_window, local_window", [
    ("AABB", 4, 1023),
])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_decode_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    global_window: int,
    local_window: int,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    swa_decode = MojoPagedDecodeSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    perf(  # noqa: F821
        lambda: swa_decode(
            query,
            k_cache,
            v_cache,
            total_seq_lens,
            block_tables,
            softmax_scale=softmax_scale,
        )
    )


def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    key = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)

    return query, key, value, cu_q_lens, cu_total_seq_lens

test_configs_swa_infer = [
    (2, 16, 4, 128, 1024, 0, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 16, 4, 128, 1024, 8192, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, key, value, cu_q_lens, cu_total_seq_lens",
    [
        pytest.param(
            *generate_sdpa_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, dtype, ID in test_configs_swa_infer
    ],
)
@pytest.mark.parametrize("gqa_layout, global_window, local_window", [
    ("AABB", 4, 1023),
])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_swa_infer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    gqa_layout: str,
    global_window: int,
    local_window: int,
):
    swa_infer = MojoSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: swa_infer(
            query,
            key,
            value,
            cu_q_lens,
            cu_total_seq_lens,
            softmax_scale=softmax_scale,
        )
    )
    
    # ===========================================================================
# MojoPagedPrefillGQAWithKVDequant
# ===========================================================================

test_configs_prefill_gqa_with_kv_dequant = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 512, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 2, 128, 1024, 0, 128, torch.bfloat16, "M_BF16_GROUP"),
    (3, 12, 3, 64, 257, 513, 16, torch.bfloat16, "M_BF16_VARLEN_BLK16_D64"),
    (4, 20, 5, 192, 193, 769, 256, torch.bfloat16, "M_BF16_VARLEN_BLK256_D192"),
    (3, 24, 6, 80, 321, 641, 64, torch.bfloat16, "M_BF16_VARLEN_BLK64_D80"),
    (1, 16, 4, 128, 128, 0, 16, torch.bfloat16, "M_BF16_SMALL_BLK16"),
    (2, 24, 6, 128, 255, 129, 32, torch.bfloat16, "M_BF16_H24_VARLEN"),
    (3, 16, 4, 128, 513, 257, 64, torch.bfloat16, "M_BF16_VARLEN_513"),
    (4, 24, 6, 128, 769, 511, 128, torch.bfloat16, "M_BF16_H24_BLK128"),
    (5, 16, 4, 128, 1025, 333, 256, torch.bfloat16, "M_BF16_ODD_KV_333"),
    (6, 24, 6, 128, 1537, 777, 128, torch.bfloat16, "M_BF16_H24_ODD_777"),
    (5, 16, 4, 128, 2049, 1025, 256, torch.bfloat16, "M_BF16_LONG_ODD"),
    (4, 24, 6, 128, 3073, 1537, 256, torch.bfloat16, "M_BF16_H24_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_lens, max_total_seq_lens",
    [
        pytest.param(
            *generate_paged_prefill_quant_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_gqa_with_kv_dequant
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@pytest.mark.parametrize("query_dtype, context_dtype, compute_dtype", 
    [
        (torch.bfloat16, torch.int8, torch.bfloat16),
        (torch.bfloat16, torch.int8, torch.int8),
    ]
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_prefill_gqa_with_kv_dequant(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    max_q_lens: int,
    max_total_seq_lens: int,
    gqa_layout: str,
    query_dtype: torch.dtype,
    context_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):
    query_q, query_scale = _quantize_query(query, query_dtype)
    k_cache_q, key_scale = _quantize_kv_cache(k_cache, context_dtype)
    v_cache_q, value_scale = _quantize_kv_cache(v_cache, context_dtype)

    op = MojoPagedPrefillGQAWithKVDequant(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedPrefillGQAWithKVDequant._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: op(
        query_q,
        query_scale,
        k_cache_q,
        key_scale,
        v_cache_q,
        value_scale,
        cu_q_lens,
        block_tables,
        softmax_scale=softmax_scale,
        cu_total_seq_lens=cu_total_seq_lens,
        max_q_lens=max_q_lens,
        max_total_seq_lens=max_total_seq_lens,
        )
    )
