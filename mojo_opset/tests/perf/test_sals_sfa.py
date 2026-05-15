# -*- coding: utf-8 -*-
"""Performance: MojoSALSSFA operator-level benchmark.

Only tests MODEL_SPECS scenarios as required.

pytest mojo_opset/tests/perf/test_sals_sfa.py -v
"""
from __future__ import annotations

import pytest
import torch

from mojo_opset import MojoSALSSFA
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


SPARSE_BLOCK_SIZE = 16
HEAD_DIM = 128
DEFAULT_K = 8

MODEL_SPECS = [
    ("new_model_1", 128, 8, 2),
    ("new_model_2", 128, 8, 4),
    ("new_model_3", 128, 16, 4),
    ("new_model_4", 128, 32, 8),
    ("new_model_5", 128, 16, 8),
    ("new_model_6", 128, 32, 16),
    ("M9-23B", 128, 80, 8),
    ("M8-14B", 128, 64, 8),
]


def _generate_sfa_data(num_query_heads, num_kv_heads, head_dim, B_req, q_lens, base_kv_lens):
    device = get_torch_device()
    dtype = torch.float16
    G = B_req
    K = DEFAULT_K
    sbs = SPARSE_BLOCK_SIZE
    softmax_scale = 1.0 / (head_dim**0.5)

    T = sum(q_lens)
    cumsum_q = [0]
    for ql in q_lens:
        cumsum_q.append(cumsum_q[-1] + ql)
    total_kv_lens = [base_kv_lens[b] + q_lens[b] for b in range(B_req)]

    cache_block_size = sbs
    max_blocks_needed = max(
        (tv + cache_block_size - 1) // cache_block_size
        for tv in total_kv_lens
    ) if total_kv_lens else 1
    table_len = max_blocks_needed
    num_blocks = table_len * B_req + 4

    k_cache = torch.randn(
        num_blocks, cache_block_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    v_cache = torch.randn(
        num_blocks, cache_block_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    block_tables = torch.arange(
        0, B_req * table_len, dtype=torch.int32, device=device,
    ).reshape(B_req, table_len)

    q = torch.randn(T, num_query_heads, head_dim, dtype=dtype, device=device)

    group_qid = torch.zeros(G, dtype=torch.int32, device=device)
    group_q_start = torch.zeros(G, dtype=torch.int32, device=device)
    group_q_len_t = torch.zeros(G, dtype=torch.int32, device=device)
    seq_len_flat = torch.zeros(G, dtype=torch.int32, device=device)
    indices_flat = torch.zeros(G, num_kv_heads, K, dtype=torch.int32, device=device)

    for i in range(G):
        qid = i % B_req
        group_qid[i] = qid
        groups_for_req = sum(1 for j in range(G) if j % B_req == qid)
        group_idx = sum(1 for j in range(i) if j % B_req == qid)
        q_len_req = q_lens[qid]
        chunk_size = q_len_req // groups_for_req if groups_for_req > 0 else 0
        q_start_offset = cumsum_q[qid] + group_idx * chunk_size
        if group_idx < groups_for_req - 1:
            q_end_offset = cumsum_q[qid] + (group_idx + 1) * chunk_size
        else:
            q_end_offset = cumsum_q[qid + 1]
        actual_q_len = q_end_offset - q_start_offset
        group_q_start[i] = q_start_offset
        group_q_len_t[i] = actual_q_len

        tv = total_kv_lens[qid]
        max_logical_blocks = (tv + sbs - 1) // sbs
        num_selected = min(K, max(1, max_logical_blocks // 2)) if max_logical_blocks > 0 else 0
        if max_logical_blocks > 0 and num_selected > 0:
            perm = torch.randperm(max_logical_blocks, device=device)[:num_selected]
            actual = perm.shape[0]
            for h in range(num_kv_heads):
                indices_flat[i, h, :actual] = perm
                if actual > 0 and actual < K:
                    indices_flat[i, h, actual:] = perm[-1]
        seq_len_flat[i] = num_selected

    base_kv_len = torch.tensor(base_kv_lens, dtype=torch.int32, device=device)
    cumsum_q_len = torch.tensor(cumsum_q, dtype=torch.int32, device=device)

    return (
        q, k_cache, v_cache,
        None, None,
        block_tables, indices_flat, seq_len_flat,
        group_qid, group_q_start, group_q_len_t,
        cumsum_q_len, base_kv_len, None,
        softmax_scale,
        num_kv_heads, num_query_heads, head_dim, sbs,
    )


@pytest.mark.parametrize(
    "q, k_cache, v_cache, k_scales, v_scales, "
    "block_tables, indices_flat, seq_len_flat, "
    "group_qid, group_q_start, group_q_len, "
    "cumsum_q_len, base_kv_len, group_use_dense, "
    "softmax_scale, "
    "num_kv_heads, num_query_heads, head_dim, sparse_block_size",
    [
        pytest.param(
            *_generate_sfa_data(
                num_query_heads=qh, num_kv_heads=kv, head_dim=hd,
                B_req=2, q_lens=[32, 48], base_kv_lens=[64, 96],
            ),
            id=f"{name}-B2-S32-48",
        )
        for name, hd, qh, kv in MODEL_SPECS
    ] + [
        pytest.param(
            *_generate_sfa_data(
                num_query_heads=qh, num_kv_heads=kv, head_dim=hd,
                B_req=1, q_lens=[64], base_kv_lens=[128],
            ),
            id=f"{name}-B1-S64",
        )
        for name, hd, qh, kv in MODEL_SPECS
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_sfa_perf(
    q, k_cache, v_cache, k_scales, v_scales,
    block_tables, indices_flat, seq_len_flat,
    group_qid, group_q_start, group_q_len,
    cumsum_q_len, base_kv_len, group_use_dense,
    softmax_scale,
    num_kv_heads, num_query_heads, head_dim, sparse_block_size,
):
    op = MojoSALSSFA()
    perf(lambda: op(
        q, k_cache, v_cache, k_scales, v_scales,
        block_tables, indices_flat, seq_len_flat,
        group_qid, group_q_start, group_q_len,
        cumsum_q_len, base_kv_len, group_use_dense,
        softmax_scale,
        num_kv_heads, num_query_heads, head_dim, sparse_block_size,
    ))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
