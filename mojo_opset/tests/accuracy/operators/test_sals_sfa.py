# -*- coding: utf-8 -*-
"""Accuracy: MojoSALSSFA (ttx vs torch reference).

Operator-level test comparing the TTX NPU kernel against the pure-PyTorch
reference for the SALS Sparse Flash Attention operator.

pytest mojo_opset/tests/accuracy/operators/test_sals_sfa.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mojo_opset import MojoSALSSFA
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented


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


def _device():
    return get_torch_device()


def _make_sfa_inputs(
    *,
    num_query_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = HEAD_DIM,
    sparse_block_size: int = SPARSE_BLOCK_SIZE,
    B_req: int = 2,
    q_lens: list[int] | None = None,
    base_kv_lens: list[int] | None = None,
    G: int | None = None,
    K: int = DEFAULT_K,
    dtype: torch.dtype = torch.float16,
    with_scales: bool = False,
    with_dense: bool = False,
) -> dict:
    device = _device()
    assert B_req >= 1

    if q_lens is None:
        q_lens = [32] * B_req
    if base_kv_lens is None:
        base_kv_lens = [64] * B_req
    assert len(q_lens) == B_req
    assert len(base_kv_lens) == B_req

    if G is None:
        G = B_req
    assert G >= 0

    T = sum(q_lens)
    softmax_scale = 1.0 / (head_dim**0.5)

    cumsum_q = [0]
    for ql in q_lens:
        cumsum_q.append(cumsum_q[-1] + ql)

    total_kv_lens = [base_kv_lens[b] + q_lens[b] for b in range(B_req)]

    cache_block_size = sparse_block_size
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

    k_scales: torch.Tensor | None = None
    v_scales: torch.Tensor | None = None
    if with_scales:
        k_scales = torch.randn(
            num_kv_heads, head_dim, dtype=torch.float32, device=device,
        ).abs() + 0.5
        v_scales = torch.randn(
            num_kv_heads, head_dim, dtype=torch.float32, device=device,
        ).abs() + 0.5

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
        max_logical_blocks = (tv + sparse_block_size - 1) // sparse_block_size

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

    group_use_dense: torch.Tensor | None = None
    if with_dense and G > 0:
        group_use_dense = torch.zeros(G, dtype=torch.int32, device=device)
        for i in range(G):
            if i % 3 == 0:
                group_use_dense[i] = 1

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scales=k_scales,
        v_scales=v_scales,
        block_tables=block_tables,
        indices_flat=indices_flat,
        seq_len_flat=seq_len_flat,
        group_qid=group_qid,
        group_q_start=group_q_start,
        group_q_len=group_q_len_t,
        cumsum_q_len=cumsum_q_len,
        base_kv_len=base_kv_len,
        group_use_dense=group_use_dense,
        softmax_scale=softmax_scale,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        head_dim=head_dim,
        sparse_block_size=sparse_block_size,
    )


def _assert_match(torch_out: torch.Tensor, ttx_out: torch.Tensor, *, atol: float = 0.1):
    assert torch_out.shape == ttx_out.shape, (
        f"Shape mismatch: torch {torch_out.shape} vs ttx {ttx_out.shape}"
    )
    nonzero_mask = torch_out.abs() > 0
    if nonzero_mask.any():
        torch_flat = torch_out[nonzero_mask].float()
        ttx_flat = ttx_out[nonzero_mask].float()
        cos_sim = F.cosine_similarity(
            torch_flat.unsqueeze(0), ttx_flat.unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"
    max_err = (torch_out.float() - ttx_out.float()).abs().max().item()
    assert max_err < atol, f"Max abs error {max_err:.6f} >= {atol}"


def _run_operator_vs_reference(kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    op = MojoSALSSFA()
    torch_cls = op._registry.get("torch")
    ttx_cls = op._registry.get("ttx")

    torch_out = torch_cls().forward(**kwargs)
    ttx_out = ttx_cls().forward(**kwargs)
    return torch_out, ttx_out


@pytest.fixture
def assert_ttx_vs_torch():
    def _fn(kwargs, *, atol=0.1):
        torch_out, ttx_out = _run_operator_vs_reference(kwargs)
        _assert_match(torch_out, ttx_out, atol=atol)
    return _fn


# ===== Basic shape tests =====

@pytest.mark.parametrize("num_query_heads,num_kv_heads", [
    (8, 2), (4, 4), (16, 4),
])
@auto_switch_platform()
@bypass_not_implemented
def test_basic(num_query_heads, num_kv_heads, assert_ttx_vs_torch):
    for B_req, q_lens, base_kv_lens in [
        (2, [32, 48], [64, 96]),
        (1, [32], [64]),
        (3, [16, 32, 48], [32, 64, 80]),
    ]:
        kw = _make_sfa_inputs(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            B_req=B_req,
            q_lens=q_lens,
            base_kv_lens=base_kv_lens,
        )
        assert_ttx_vs_torch(kw)


# ===== Edge cases =====

@auto_switch_platform()
@bypass_not_implemented
def test_empty_groups(assert_ttx_vs_torch):
    kw = _make_sfa_inputs(B_req=1, q_lens=[32], base_kv_lens=[64], G=0)
    op = MojoSALSSFA()
    torch_cls = op._registry.get("torch")
    ttx_cls = op._registry.get("ttx")
    torch_out = torch_cls().forward(**kw)
    ttx_out = ttx_cls().forward(**kw)
    assert torch_out.abs().max().item() == 0.0
    assert ttx_out.abs().max().item() == 0.0
    torch.testing.assert_close(torch_out, ttx_out, atol=0, rtol=0)


@auto_switch_platform()
@bypass_not_implemented
def test_with_dense_groups(assert_ttx_vs_torch):
    kw = _make_sfa_inputs(
        B_req=3, q_lens=[32, 48, 32], base_kv_lens=[64, 80, 64],
        G=3, with_dense=True,
    )
    assert_ttx_vs_torch(kw, atol=0.15)


@auto_switch_platform()
@bypass_not_implemented
def test_with_scales(assert_ttx_vs_torch):
    kw = _make_sfa_inputs(
        num_query_heads=8, num_kv_heads=2,
        B_req=2, q_lens=[32, 48], base_kv_lens=[64, 96],
        with_scales=True,
    )
    assert_ttx_vs_torch(kw, atol=0.15)


@auto_switch_platform()
@bypass_not_implemented
def test_single_group(assert_ttx_vs_torch):
    kw = _make_sfa_inputs(
        num_query_heads=4, num_kv_heads=2,
        B_req=1, q_lens=[16], base_kv_lens=[32],
        G=1,
    )
    assert_ttx_vs_torch(kw)


@auto_switch_platform()
@bypass_not_implemented
def test_multiple_groups_per_request(assert_ttx_vs_torch):
    kw = _make_sfa_inputs(
        num_query_heads=8, num_kv_heads=2,
        B_req=2, q_lens=[48, 64], base_kv_lens=[80, 96],
        G=4,
    )
    assert_ttx_vs_torch(kw)


# ===== Dtype tests =====

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@auto_switch_platform()
@bypass_not_implemented
def test_dtype(dtype, assert_ttx_vs_torch):
    kw = _make_sfa_inputs(
        B_req=2, q_lens=[32, 48], base_kv_lens=[64, 96],
        dtype=dtype,
    )
    assert_ttx_vs_torch(kw)


# ===== Determinism =====

@auto_switch_platform()
@bypass_not_implemented
def test_determinism():
    kw = _make_sfa_inputs(B_req=2, q_lens=[32, 48], base_kv_lens=[64, 96])
    op = MojoSALSSFA()
    torch_cls = op._registry.get("torch")
    ttx_cls = op._registry.get("ttx")

    for cls in [torch_cls, ttx_cls]:
        impl = cls()
        a = impl.forward(**kw)
        b = impl.forward(**kw)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


# ===== Critical model spec tests (MUST INCLUDE) =====

@pytest.mark.parametrize("model_name,head_dim,num_query_heads,num_kv_heads", MODEL_SPECS)
@auto_switch_platform()
@bypass_not_implemented
def test_sfa_model_specs(model_name, head_dim, num_query_heads, num_kv_heads, assert_ttx_vs_torch):
    for B_req, q_lens, base_kv_lens in [
        (2, [32, 48], [64, 96]),
        (1, [64], [128]),
    ]:
        kw = _make_sfa_inputs(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sparse_block_size=SPARSE_BLOCK_SIZE,
            B_req=B_req,
            q_lens=q_lens,
            base_kv_lens=base_kv_lens,
        )
        assert_ttx_vs_torch(kw)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
