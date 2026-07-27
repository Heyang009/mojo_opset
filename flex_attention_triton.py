import os

import torch
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import BlockMask, _create_sparse_block_from_block_mask
from typing import NamedTuple, Optional
import triton
import triton.language as tl


Q_BUILD_CHUNK = int(os.environ.get("Q_BUILD_CHUNK", "512"))
APPLY_Q_CHUNK = int(os.environ.get("APPLY_Q_CHUNK", "2048"))
USE_TRITON_BLOCK_SUMMARY = os.environ.get("Z_USE_TRITON_BLOCK_SUMMARY", "0") == "1"

TILE_BLOCK_SIZE = 128

_BITS = 32


def _get_num_aicore():
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not hasattr(npu_mod, "current_device"):
        return 1
    device = npu_mod.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    return max(int(props.get("num_aicore", 1)), 1)


def _persistent_launch_config(num_tasks):
    num_tasks = max(int(num_tasks), 1)
    return (min(_get_num_aicore(), num_tasks),), num_tasks


# ---------------------------------------------------------------------------
# Q-split heuristic for dkdv backward kernel (deterministic load balancing)
# ---------------------------------------------------------------------------
_QSPLIT_ENV = os.environ.get("USE_QSPLIT_DKDV", "auto")
_QSPLIT_MAX_PARTIAL_MB = int(os.environ.get("QSPLIT_MAX_PARTIAL_MB", "64"))
# Backward path: "split" = separate dq kernel + qsplit dkdv + reduce_dkdv
#                "fused" = fused dqdkdv kernel + reduce_dq
_USE_FUSED_BACKWARD = os.environ.get("USE_FUSED_BACKWARD", "0") == "1"


def _compute_qsplit_k(q_num_blks, full_q_num_blks, M, N, Hkv, GQA_SHARED_HEADS):
    """
    Heuristic: return (SPLIT_K, split_start) for dkdv kernel.

    - SPLIT_K: Q-split factor for the REMAINDER tasks (1=no split at all)
    - split_start: base task index where splitting begins.
      Base tasks [0, split_start) are NOT split (process all Q blocks directly).
      Base tasks [split_start, total_base) ARE split into SPLIT_K sub-tasks.

    Optimization strategy (cores-per-remainder-KV):
      total_base = KV_NB * Z * Hkv
      remainder  = total_base % num_core
      If remainder == 0: no split needed (SPLIT_K=1).
      Else:
        - First (total_base - remainder) tasks fill cores evenly → no split
        - Last 'remainder' tasks: each gets (num_core // remainder) cores (floor)
          → SPLIT_K = num_core // remainder

    Note: Hkv is passed explicitly because block_mask's head dim is always 1
          (the sparse mask is shared across heads), not the real KV head count.
    """
    if _QSPLIT_ENV == "0":
        return 1, 0
    if _QSPLIT_ENV == "1":
        return 4, 0

    num_core = _get_num_aicore()
    KV_NB = q_num_blks.shape[-1]
    Z = q_num_blks.shape[0]

    total_base = KV_NB * Z * Hkv
    remainder = total_base % num_core

    if remainder == 0:
        return 1, total_base

    # SPLIT_K: cores allocated per remaining KV block (floor division)
    SPLIT_K = num_core // remainder
    if SPLIT_K < 2:
        return 1, total_base

    # Memory check: partial buffer = SPLIT_K * Z * Hkv * N * D * 4 bytes * 2 (DK+DV)
    D = 128
    max_bytes = _QSPLIT_MAX_PARTIAL_MB * 1024**2
    partial_bytes = SPLIT_K * Z * Hkv * N * D * 4 * 2
    if partial_bytes > max_bytes:
        return 1, total_base

    split_start = total_base - remainder
    return SPLIT_K, split_start


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_partial_p", "stride_partial_m",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_out_z", "stride_out_h",
    ]
)
def flex_attention_kernel(
    Q,
    K,
    V,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    OUT,
    LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_out_z, stride_out_h, stride_out_m, stride_out_k,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_kv_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    """
    Triton kernel for FlexAttention forward pass.

    Args:
        Q: Query tensor [Z, Hq, M, D]
        K: Key tensor [Z, Hkv, N, D]
        V: Value tensor [Z, Hkv, N, Dv]
        KV_NUM_BLKS: Number of KV blocks per Q block
        KV_IDX: Indices of KV blocks per Q block
        FULL_KV_NUM_BLKS: Number of fully unmasked KV blocks per Q block
        FULL_KV_IDX: Indices of fully unmasked KV blocks per Q block
        OUT: Output tensor [Z, Hq, M, Dv]
        LSE: Log-sum-exp output tensor [Z, Hq, M]
        SM_SCALE: Scale factor for softmax
    """
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        out_offset = off_z * stride_out_z + off_hq * stride_out_h
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        OUT_ptr = OUT + out_offset
        LSE_ptr = LSE + lse_offset

        # m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),     #   & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0
        )

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        partial_mask_offset = tl.load(PARTIAL_MASK_OFFSETS + q_sparse_idx * stride_partial_offset_m)
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        for start_n in range(0, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

            offs_n_load = kv_start + tl.arange(0, BLOCK_N)
            if USE_PACKED_PARTIAL_MASK:
                partial_block_idx = partial_mask_offset + blk_idx_in_list
                offs_m_in_block = offs_m - q_sparse_base
                offs_n_in_block = (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = load_packed_partial_mask(
                    PARTIAL_MASK_PACKED,
                    stride_partial_p,
                    stride_partial_m,
                    stride_partial_n,
                    partial_block_idx,
                    offs_m_in_block,
                    offs_n_in_block,
                    SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                )
            else:
                mask = load_dense_mask(
                    DENSE_MASK,
                    stride_mask_m,
                    stride_mask_n,
                    offs_m,
                    offs_n_load,
                    Q_LEN=Q_LEN,
                    KV_LEN=KV_LEN,
                )

            k = tl.load(
                K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n_load[:, None] < KV_LEN),   # & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0
            )
            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN),   # & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0
            )
            k = tl.trans(k)

            qk = tl.dot(q, k, input_precision="ieee")
            qk *= SM_SCALE

            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)

            alpha = tl.math.exp(m_i - m_ij_masked)
            p = tl.math.exp(qk - m_ij_masked[:, None])

            pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        if HAS_FULL_BLOCKS:
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

            for start_n in range(0, block_n_end):
                blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
                kv_block = tl.load(kv_indices + blk_idx_in_list)
                kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

                offs_n_load = kv_start + tl.arange(0, BLOCK_N)
                k = tl.load(
                    K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n_load[:, None] < KV_LEN), # & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0
                )
                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN), # & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0
                )
                k = tl.trans(k)

                qk = tl.dot(q, k, input_precision="ieee")
                qk *= SM_SCALE

                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

                alpha = tl.math.exp(m_i - m_ij)
                p = tl.math.exp(qk - m_ij[:, None])

                pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + pv
                m_i = m_ij
        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]

        out_mask = (offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM)
        tl.store(
            OUT_ptr + offs_m[:, None] * stride_out_m + offs_v[None, :] * stride_out_k,
            acc,
            mask=out_mask
        )

        lse = m_i + tl.math.log(l_i)
        tl.store(LSE_ptr + offs_m * stride_lse_m, lse, mask=offs_m < Q_LEN)


@triton.jit
def load_dense_mask(
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    offs_m,
    offs_n,
    Q_LEN,
    KV_LEN,
):
    stride_mask_m = stride_mask_m.to(tl.int64)
    # stride_mask_n = stride_mask_n.to(tl.int64)
    ptrs = DENSE_MASK + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    valid = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def load_packed_partial_mask(
    PARTIAL_MASK_PACKED,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    partial_block_idx,
    offs_m_in_block,
    offs_n_in_block,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
):
    ptrs = (
        PARTIAL_MASK_PACKED
        + partial_block_idx * stride_partial_p
        + offs_m_in_block[:, None] * stride_partial_m
        + offs_n_in_block[None, :] * stride_partial_n
    )
    valid = (
        (offs_m_in_block[:, None] < SPARSE_Q_BLOCK_SIZE)
        & (offs_n_in_block[None, :] < SPARSE_KV_BLOCK_SIZE)
    )
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def bwd_dq_block_mn(
    q, do, lse, delta,
    K_ptr, V_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    Q_LEN, KV_LEN,
    offs_m, offs_n, offs_k, offs_v,
    q_sparse_idx, kv_block, kv_sub, q_sparse_base,
    stride_kn, stride_kk, stride_vn, stride_vk,
    MATMUL_PRECISION,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
):
    k = tl.load(
        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
        mask=(offs_n[:, None] < KV_LEN),    # & (offs_k[None, :] < QK_HEAD_DIM)
        other=0.0,
    )
    v = tl.load(
        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
        mask=(offs_n[:, None] < KV_LEN),    # & (offs_v[None, :] < V_HEAD_DIM)
        other=0.0,
    )

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    mask = True
    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_block * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            offs_m_in_block = offs_m - q_sparse_base
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask & (offs_n[None, :] < KV_LEN), qk, float("-inf"))
    else:
        qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

    p = tl.math.exp(qk - lse[:, None])
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = p * (dp - delta[:, None])

    # if not IS_FULL_BLOCKS:
    #     ds = tl.where(mask, ds, 0.0)

    dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
    return dq


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dqz", "stride_dqh",
    ]
)
def flex_attention_backward_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_kv_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS
    sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE
    MATMUL_PRECISION = Q.dtype.element_ty

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        do_offset = off_z * stride_doz + off_hq * stride_doh
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
        delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h
        dq_offset = off_z * stride_dqz + off_hq * stride_dqh

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DO_ptr = DO + do_offset
        LSE_ptr = LSE + lse_offset
        DELTA_ptr = DELTA + delta_offset
        DQ_ptr = DQ + dq_offset

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),     # & (offs_k[None, :] < QK_HEAD_DIM)
            other=0.0,
        )
        do = tl.load(
            DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
            mask=(offs_m[:, None] < Q_LEN),     # & (offs_v[None, :] < V_HEAD_DIM)
            other=0.0,
        )

        lse = tl.load(LSE_ptr + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
        delta = tl.load(DELTA_ptr + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
        lse = tl.where(lse == float("-inf"), 0.0, lse)

        dq = tl.zeros([BLOCK_M, QK_HEAD_DIM], dtype=tl.float32)

        q_sparse_idx = q_start // sparse_q_multiple
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        for blk_idx_in_list in range(0, kv_num_blocks):
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

            for kv_sub in range(NUM_KV_SUB_BLOCKS):
                start_n = kv_start_full + kv_sub * BLOCK_N
                offs_n = start_n + tl.arange(0, BLOCK_N)

                k = tl.load(
                    K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n[:, None] < KV_LEN),    # & (offs_k[None, :] < QK_HEAD_DIM)
                    other=0.0,
                )
                v = tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n[:, None] < KV_LEN),    # & (offs_v[None, :] < V_HEAD_DIM)
                    other=0.0,
                )

                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                qk *= SM_SCALE

                if USE_PACKED_PARTIAL_MASK:
                    partial_block_idx = tl.load(
                        PARTIAL_BLOCK_TABLE
                        + q_sparse_idx * stride_partial_table_m
                        + kv_block * stride_partial_table_n
                    )
                    safe_partial_block_idx = tl.maximum(partial_block_idx, 0, propagate_nan=True)
                    offs_m_in_block = offs_m - q_sparse_base
                    offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask = load_packed_partial_mask(
                        PARTIAL_MASK_PACKED,
                        stride_partial_p,
                        stride_partial_m,
                        stride_partial_n,
                        safe_partial_block_idx,
                        offs_m_in_block,
                        offs_n_in_block,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                    )
                    mask = mask & (partial_block_idx >= 0)
                else:
                    mask = load_dense_mask(
                        DENSE_MASK,
                        stride_mask_m,
                        stride_mask_n,
                        offs_m,
                        offs_n,
                        Q_LEN=Q_LEN,
                        KV_LEN=KV_LEN,
                    )
                qk = tl.where(mask, qk, float("-inf"))  # & (offs_n[None, :] < KV_LEN)

                p = tl.math.exp(qk - lse[:, None])
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta[:, None])
                ds *= SM_SCALE
                # ds = tl.where(mask, ds, 0.0)
                dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        if HAS_FULL_BLOCKS:
            kv_indices_f = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks_f = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            for blk_idx_in_list in range(0, kv_num_blocks_f):
                kv_block = tl.load(kv_indices_f + blk_idx_in_list)
                kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

                for kv_sub in range(NUM_KV_SUB_BLOCKS):
                    start_n = kv_start_full + kv_sub * BLOCK_N
                    offs_n = start_n + tl.arange(0, BLOCK_N)

                    k = tl.load(
                        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                        mask=(offs_n[:, None] < KV_LEN),    # & (offs_k[None, :] < QK_HEAD_DIM)
                        other=0.0,
                    )
                    v = tl.load(
                        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=(offs_n[:, None] < KV_LEN),    #  & (offs_v[None, :] < V_HEAD_DIM)
                        other=0.0,
                    )

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE
                    # qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])
                    ds *= SM_SCALE
                    dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        # dq *= SM_SCALE
        tl.store(
            DQ_ptr + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )



@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                q_offset = off_z * stride_qz + off_hq * stride_qh
                do_offset = off_z * stride_doz + off_hq * stride_doh
                dq_offset = off_z * stride_qz + off_hq * stride_qh
                lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_h = Q + q_offset
                DQ_h = DQ + dq_offset
                DO_h = DO + do_offset
                LSE_h = LSE + lse_offset
                DELTA_h = DELTA + delta_offset

                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )
                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_dkdv_block_mn(
                        Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        COMPUTE_DQ=False,
                    )

                if HAS_FULL_BLOCKS:
                    q_indices = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end = tl.minimum(
                        q_num_blocks * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                    )

                    for start_m in range(0, block_m_end):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)

                        bwd_dkdv_block_mn(
                            Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                            COMPUTE_DQ=False,
                        )


# ===========================================================================
# Fused dq+dk+dv backward kernel
#
# Task decomposition: same as dkdv — task = (kv_block, z, hkv).
# For each Q block:
#   dk/dv: atomic_add to DK/DV (accumulate over Q, same as dkdv)
#   dq:    atomic_add to DQ_PARTIAL[kv_block] (disjoint per kv_block → no cross-task contention)
# A separate reduce_dq_kernel sums DQ_PARTIAL over kv_blocks → DQ.
# ===========================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
        "stride_dqp_kv",
    ]
)
def flex_attention_backward_dqdkdv_fused_kernel(
    Q, K, V, DO, LSE, DELTA,
    Q_NUM_BLKS, Q_IDX, FULL_Q_NUM_BLKS, FULL_Q_IDX,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, PARTIAL_MASK_OFFSETS, PARTIAL_BLOCK_TABLE,
    stride_partial_p, stride_partial_m, stride_partial_n,
    stride_partial_offset_m, stride_partial_table_m, stride_partial_table_n,
    DK, DV, DQ_PARTIAL,
    stride_dqp_kv,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                q_offset = off_z * stride_qz + off_hq * stride_qh
                do_offset = off_z * stride_doz + off_hq * stride_doh
                lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_h = Q + q_offset
                DO_h = DO + do_offset
                LSE_h = LSE + lse_offset
                DELTA_h = DELTA + delta_offset
                # dq_partial base for this kv_block, z, hq
                DQ_PARTIAL_h = DQ_PARTIAL + kv_start_block * stride_dqp_kv + q_offset

                # ---- Partial Q-blocks ----
                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL,
                )
                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_fused_block_mn(
                        Q_h, DO_h, DQ_PARTIAL_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                    )

                # ---- Full Q-blocks ----
                if HAS_FULL_BLOCKS:
                    q_indices_f = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks_f = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end_f = tl.minimum(
                        q_num_blocks_f * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL,
                    )
                    for start_m in range(0, block_m_end_f):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices_f + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)

                        bwd_fused_block_mn(
                            Q_h, DO_h, DQ_PARTIAL_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        )


@triton.jit
def bwd_fused_block_mn(
    Q, DO, DQ_PARTIAL_h, DK_ptr, DELTA, LSE, DV_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
    stride_qm, stride_qk, stride_dom, stride_dok,
    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
    MATMUL_PRECISION,
    SM_SCALE: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
):
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        other=0.0,
    )
    do = tl.load(
        DO + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    lse = tl.load(LSE + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == float("-inf"), 0.0, lse)

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    mask = True
    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_sparse_idx * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
            offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask, qk, float("-inf"))

    p = tl.math.exp(qk - lse[:, None])

    # DV
    dv = tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do, input_precision="ieee")
    tl.atomic_add(
        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
        dv,
        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
    )

    # DS
    Di = tl.load(DELTA + offs_m, mask=offs_m < Q_LEN, other=0.0)
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = (p * (dp - Di[:, None]))
    ds *= SM_SCALE

    # DQ → dq_partial (atomic_add, disjoint per kv_block)
    dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
    tl.atomic_add(
        DQ_PARTIAL_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        dq,
        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )

    # DK
    dk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
    tl.atomic_add(
        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk,
        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )


# ===========================================================================
# Reduce kernel: DQ = sum(DQ_PARTIAL[0..NUM_KV_BLOCKS-1])
# Each task reduces one [z, hq, M-tile] slice over all kv_blocks.
# ===========================================================================
@triton.jit(
    do_not_specialize=[
        "NUM_TASKS", "Q_LEN", "NUM_KV_BLOCKS",
        "stride_dqp_kv",
        "stride_dqz", "stride_dqh",
    ]
)
def reduce_dq_kernel(
    DQ,
    DQ_PARTIAL,
    stride_dqp_kv,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    NUM_TASKS, Q_LEN, NUM_KV_BLOCKS,
    Q_HEAD,
    BLOCK_M: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    offs_k = tl.arange(0, QK_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        m_tile = task_id % tl.cdiv(Q_LEN, BLOCK_M)
        zhq = task_id // tl.cdiv(Q_LEN, BLOCK_M)
        off_z = (zhq // Q_HEAD).to(tl.int64)
        off_hq = (zhq % Q_HEAD).to(tl.int64)

        dq_base = off_z * stride_dqz + off_hq * stride_dqh
        offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < Q_LEN

        dq_sum = tl.zeros([BLOCK_M, QK_HEAD_DIM], dtype=tl.float32)
        for kv in range(NUM_KV_BLOCKS):
            dq_sum += tl.load(
                DQ_PARTIAL + kv * stride_dqp_kv + dq_base
                + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
                mask=m_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
        tl.store(
            DQ + dq_base + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
            dq_sum,
            mask=m_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
        )
#
# Splits each KV-block task into K_SPLIT sub-tasks along the Q-block axis.
# Uses atomic_add to disjoint partial buffer slots (one per sub_id).
# Deterministic: each (kv_block, sub_id) task runs on exactly one core,
# GQA heads are sequential, and different tasks write to disjoint memory.
# A separate reduce_dkdv_kernel sums the K partials in fixed order.
# ===========================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
        "stride_dkp_k", "stride_dvp_k",
        "SPLIT_START",
    ]
)
def flex_attention_backward_dkdv_kernel_qsplit(
    Q, K, V, DO, LSE, DELTA,
    Q_NUM_BLKS, Q_IDX, FULL_Q_NUM_BLKS, FULL_Q_IDX,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, PARTIAL_MASK_OFFSETS, PARTIAL_BLOCK_TABLE,
    stride_partial_p, stride_partial_m, stride_partial_n,
    stride_partial_offset_m, stride_partial_table_m, stride_partial_table_n,
    DK_PARTIAL, DV_PARTIAL,
    stride_dkp_k, stride_dvp_k,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    K_SPLIT: tl.constexpr,
    SPLIT_START,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        # ---- Task decomposition: direct vs split ----
        # Direct tasks [0, SPLIT_START): process ALL Q blocks, write directly to DK/DV
        # Split tasks [SPLIT_START, NUM_TASKS): process 1/K_SPLIT Q blocks, write to DK_PARTIAL[sub_id]
        if task_id < SPLIT_START:
            base_task = task_id
            sub_id = 0
        else:
            split_task = task_id - SPLIT_START
            sub_id = split_task % K_SPLIT
            base_task = SPLIT_START + split_task // K_SPLIT

        kv_start_block = base_task % NUM_KV_BLOCKS
        off_z = (base_task // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (base_task // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset

        # Output pointer: DK_PARTIAL[sub_id, z, h, ...] — disjoint per sub_id
        # Note: dk_partial[0] is aliased to dk (view), so direct tasks (sub_id=0) write to dk directly
        DK_OUT_ptr = DK_PARTIAL + sub_id * stride_dkp_k + dk_offset
        DV_OUT_ptr = DV_PARTIAL + sub_id * stride_dvp_k + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < KV_LEN

            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                Q_h = Q + off_z * stride_qz + off_hq * stride_qh
                DO_h = DO + off_z * stride_doz + off_hq * stride_doh
                LSE_h = LSE + off_z * stride_lse_z + off_hq * stride_lse_h
                DELTA_h = DELTA + off_z * stride_delta_z + off_hq * stride_delta_h

                # ---- Partial Q-blocks (this sub-task's slice) ----
                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end_p = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL,
                )
                if task_id < SPLIT_START:
                    q_start_p = 0
                    q_end_p = block_m_end_p
                else:
                    q_start_p = sub_id * block_m_end_p // K_SPLIT
                    q_end_p = (sub_id + 1) * block_m_end_p // K_SPLIT

                for start_m in range(q_start_p, q_end_p):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_dkdv_block_mn(
                            Q_h, DO_h, DK_OUT_ptr, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        COMPUTE_DQ=False,
                    )

                # ---- Full Q-blocks (this sub-task's slice) ----
                if HAS_FULL_BLOCKS:
                    q_indices_f = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks_f = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end_f = tl.minimum(
                        q_num_blocks_f * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                        propagate_nan=tl.PropagateNan.ALL,
                    )
                    if task_id < SPLIT_START:
                        q_start_f = 0
                        q_end_f = block_m_end_f
                    else:
                        q_start_f = sub_id * block_m_end_f // K_SPLIT
                        q_end_f = (sub_id + 1) * block_m_end_f // K_SPLIT

                    for start_m in range(q_start_f, q_end_f):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices_f + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)
                        q_sparse_idx = q_block

                        bwd_dkdv_block_mn(
                        Q_h, DO_h, DK_OUT_ptr, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                            COMPUTE_DQ=False,
                        )


# ===========================================================================
# Reduce kernel: only for SPLIT KV blocks — DK = sum(DK_PARTIAL[0..K-1])
# Direct KV blocks already wrote to DK/DV in the dkdv kernel.
# Deterministic: fixed iteration order s=0,1,...,K-1
# ===========================================================================
@triton.jit(
    do_not_specialize=[
        "NUM_TASKS", "KV_LEN", "SPLIT_START",
        "NUM_KV_BLOCKS",
        "stride_dkp_k", "stride_dvp_k",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def reduce_dkdv_kernel(
    DK, DV,
    DK_PARTIAL, DV_PARTIAL,
    stride_dkp_k, stride_dvp_k,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    NUM_TASKS, KV_LEN, SPLIT_START, NUM_KV_BLOCKS,
    KV_HEAD,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    NUM_N_TILES_PER_KV: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        # Each task handles one N-tile of one split KV block
        split_base_id = task_id // NUM_N_TILES_PER_KV
        n_tile_in_kv = task_id % NUM_N_TILES_PER_KV

        base_task = SPLIT_START + split_base_id
        kv_block = base_task % NUM_KV_BLOCKS
        zhkv = base_task // NUM_KV_BLOCKS
        off_z = (zhkv // KV_HEAD).to(tl.int64)
        off_hkv = (zhkv % KV_HEAD).to(tl.int64)

        start_n = kv_block * SPARSE_KV_BLOCK_SIZE + n_tile_in_kv * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < KV_LEN

        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        # Sum K partials for DK (fixed order -> deterministic)
        dk_sum = tl.zeros([BLOCK_N, QK_HEAD_DIM], dtype=tl.float32)
        for s in range(K):
            dk_sum += tl.load(
                DK_PARTIAL + s * stride_dkp_k + dk_offset
                + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
        tl.store(
            DK + dk_offset + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
            dk_sum,
            mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
        )

        # Sum K partials for DV
        dv_sum = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)
        for s in range(K):
            dv_sum += tl.load(
                DV_PARTIAL + s * stride_dvp_k + dv_offset
                + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )
        tl.store(
            DV + dv_offset + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
            dv_sum,
            mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
        )


# ===========================================================================
# Fused qsplit dkdv kernel (single-stage, lock-based ordered accumulation)
#
# Replaces the two-stage (dkdv_qsplit + reduce_dkdv) pipeline.
# - Direct tasks [0, SPLIT_START): single core per (z, hkv, kv_block, kv_sub),
#   atomic_add directly to DK/DV (pre-zeroed).
# - Split tasks [SPLIT_START, NUM_TASKS): K_SPLIT cores collaborate per KV
#   block. Each core computes its Q-segment's dk/dv into a register accumulator
#   (intra-core reduction over GQA heads + Q blocks), then acquires a per-
#   (split_base_id, kv_sub) spinlock in sub_id order and atomic_adds its
#   partial to DK/DV. Ordering guarantees deterministic accumulation
#   DK = ((dk_0 + dk_1) + dk_2) + ..., matching the original reduce kernel.
# - DK/DV are pre-zeroed externally; LOCKS buffer is pre-zeroed (one int32 per
#   (split_base_id, kv_sub) slot, value encodes which sub_id holds the lock).
# ===========================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
        "NUM_LOCKS",
        "SPLIT_START",
    ]
)
def flex_attention_backward_dkdv_kernel_qsplit_fused(
    Q, K, V, DO, LSE, DELTA,
    Q_NUM_BLKS, Q_IDX, FULL_Q_NUM_BLKS, FULL_Q_IDX,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, PARTIAL_MASK_OFFSETS, PARTIAL_BLOCK_TABLE,
    stride_partial_p, stride_partial_m, stride_partial_n,
    stride_partial_offset_m, stride_partial_table_m, stride_partial_table_n,
    # Final output buffers (pre-zeroed externally); replaces DK_PARTIAL/DV_PARTIAL
    DK, DV,
    # Spinlock buffer: one int32 per (split_base_id, kv_sub), pre-zeroed.
    # LOCKS[slot] == sub_id  =>  sub_id holds the lock (initially 0 means sub_id=0)
    LOCKS,
    NUM_LOCKS,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    K_SPLIT: tl.constexpr,
    SPLIT_START,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

    for task_id in range(pid, NUM_TASKS, num_core):
        # ---- Task decomposition: direct vs split ----
        # Direct tasks [0, SPLIT_START): process ALL Q blocks, atomic_add to DK/DV
        # Split tasks [SPLIT_START, NUM_TASKS): process 1/K_SPLIT Q blocks,
        #   acquire lock in sub_id order, then atomic_add to DK/DV
        if task_id < SPLIT_START:
            base_task = task_id
            sub_id = 0
            is_split = False
        else:
            split_task = task_id - SPLIT_START
            sub_id = split_task % K_SPLIT
            base_task = SPLIT_START + split_task // K_SPLIT
            is_split = True

        kv_start_block = base_task % NUM_KV_BLOCKS
        off_z = (base_task // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (base_task // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < KV_LEN

            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            # ---- Intra-core accumulators: aggregate GQA heads + Q blocks for this sub_id ----
            dk_acc = tl.zeros([BLOCK_N, QK_HEAD_DIM], dtype=tl.float32)
            dv_acc = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                Q_h = Q + off_z * stride_qz + off_hq * stride_qh
                DO_h = DO + off_z * stride_doz + off_hq * stride_doh
                LSE_h = LSE + off_z * stride_lse_z + off_hq * stride_lse_h
                DELTA_h = DELTA + off_z * stride_delta_z + off_hq * stride_delta_h

                # ---- Partial Q-blocks (this sub-task's slice) ----
                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end_p = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL,
                )
                if not is_split:
                    q_start_p = 0
                    q_end_p = block_m_end_p
                else:
                    q_start_p = sub_id * block_m_end_p // K_SPLIT
                    q_end_p = (sub_id + 1) * block_m_end_p // K_SPLIT

                for start_m in range(q_start_p, q_end_p):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    # ===== Inlined bwd_dkdv_block_mn (partial branch), accumulate to dk_acc/dv_acc =====
                    q = tl.load(
                        Q_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                        other=0.0,
                    )
                    do = tl.load(
                        DO_h + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                        other=0.0,
                    )
                    lse = tl.load(LSE_h + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
                    lse = tl.where(lse == float("-inf"), 0.0, lse)

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE

                    if USE_PACKED_PARTIAL_MASK:
                        partial_block_idx = tl.load(
                            PARTIAL_BLOCK_TABLE
                            + q_sparse_idx * stride_partial_table_m
                            + kv_sparse_idx * stride_partial_table_n
                        )
                        safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
                        offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
                        offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
                        mask = load_packed_partial_mask(
                            PARTIAL_MASK_PACKED,
                            stride_partial_p, stride_partial_m, stride_partial_n,
                            safe_partial_block_idx,
                            offs_m_in_block, offs_n_in_block,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        )
                        mask = mask & (partial_block_idx >= 0)
                    else:
                        mask = load_dense_mask(
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            offs_m, offs_n, Q_LEN=Q_LEN, KV_LEN=KV_LEN,
                        )
                    qk = tl.where(mask, qk, float("-inf"))

                    p = tl.math.exp(qk - lse[:, None])

                    dv_blk = tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do, input_precision="ieee")
                    dv_acc += dv_blk

                    Di = tl.load(DELTA_h + offs_m, mask=offs_m < Q_LEN, other=0.0)
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = (p * (dp - Di[:, None])) * SM_SCALE

                    dk_blk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
                    dk_acc += dk_blk

                # ---- Full Q-blocks (this sub-task's slice) ----
                if HAS_FULL_BLOCKS:
                    q_indices_f = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks_f = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end_f = tl.minimum(
                        q_num_blocks_f * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                        propagate_nan=tl.PropagateNan.ALL,
                    )
                    if not is_split:
                        q_start_f = 0
                        q_end_f = block_m_end_f
                    else:
                        q_start_f = sub_id * block_m_end_f // K_SPLIT
                        q_end_f = (sub_id + 1) * block_m_end_f // K_SPLIT

                    for start_m in range(q_start_f, q_end_f):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices_f + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)
                        q_sparse_idx = q_block

                        # ===== Inlined bwd_dkdv_block_mn (full branch, no mask) =====
                        q = tl.load(
                            Q_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                            other=0.0,
                        )
                        do = tl.load(
                            DO_h + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                            mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                            other=0.0,
                        )
                        lse = tl.load(LSE_h + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
                        lse = tl.where(lse == float("-inf"), 0.0, lse)

                        qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                        qk *= SM_SCALE
                        p = tl.math.exp(qk - lse[:, None])

                        dv_blk = tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do, input_precision="ieee")
                        dv_acc += dv_blk

                        Di = tl.load(DELTA_h + offs_m, mask=offs_m < Q_LEN, other=0.0)
                        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                        ds = (p * (dp - Di[:, None])) * SM_SCALE

                        dk_blk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
                        dk_acc += dk_blk

            # ==================== Writeback stage ====================
            dk_ptr = DK + dk_offset + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
            dv_ptr = DV + dv_offset + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk
            store_mask_dk = n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM)
            store_mask_dv = n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM)

            if not is_split:
                # ---- Direct task: single core owns this (z, hkv, kv_block, kv_sub) ----
                # DK/DV pre-zeroed, atomic_add is equivalent to store.
                tl.atomic_add(dk_ptr, dk_acc, mask=store_mask_dk)
                tl.atomic_add(dv_ptr, dv_acc, mask=store_mask_dv)
            else:
                # ---- Split task: spinlock-ordered accumulation ----
                # One lock per (split_base_id, kv_sub). LOCKS[slot] == sub_id means
                # it's sub_id's turn. Initial 0 => sub_id=0 goes first.
                split_base_id = base_task - SPLIT_START
                my_idx = split_base_id * NUM_KV_SUB_BLOCKS + kv_sub
                LOCK_ptr = LOCKS + my_idx

                # Spin-wait until it's my turn (atomic_load provides acquire semantics)
                while tl.atomic_load(LOCK_ptr) != sub_id:
                    pass

                # Critical section: atomic_add my partial (DK/DV pre-zeroed, no
                # load+add+store branch needed)
                tl.atomic_add(dk_ptr, dk_acc, mask=store_mask_dk)
                tl.atomic_add(dv_ptr, dv_acc, mask=store_mask_dv)

                # Release lock: advance turn to next sub_id (wraps to 0 after last)
                tl.atomic_xchg(LOCK_ptr, (sub_id + 1) % K_SPLIT)


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dqdkdv_kernel_b128(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        offs_n = kv_start_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        k = tl.load(
            K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
            mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
            mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < V_HEAD_DIM),
            other=0.0,
        )

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq = off_hkv * GQA_SHARED_HEADS + off_g
            off_hq = off_hq.to(tl.int64)

            q_offset = off_z * stride_qz + off_hq * stride_qh
            do_offset = off_z * stride_doz + off_hq * stride_doh
            lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
            delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

            Q_h = Q + q_offset
            DQ_h = DQ + q_offset
            DO_h = DO + do_offset
            LSE_h = LSE + lse_offset
            DELTA_h = DELTA + delta_offset

            q_indices = Q_IDX + sparse_q_idx_offset
            q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
            block_m_end = tl.minimum(
                q_num_blocks * sparse_q_multiple,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
            )
            for start_m in range(0, block_m_end):
                blk_idx_in_list = start_m // sparse_q_multiple
                q_block = tl.load(q_indices + blk_idx_in_list)
                q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                offs_m = q_start + tl.arange(0, BLOCK_M)
                q_sparse_idx = q_block

                q = tl.load(
                    Q_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0,
                )
                do = tl.load(
                    DO_h + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0,
                )
                lse = tl.load(LSE_h + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
                delta = tl.load(DELTA_h + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0).to(
                    V.dtype.element_ty)
                lse = tl.where(lse == float("-inf"), 0.0, lse)

                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                qk *= SM_SCALE

                if USE_PACKED_PARTIAL_MASK:
                    partial_block_idx = tl.load(
                        PARTIAL_BLOCK_TABLE
                        + q_sparse_idx * stride_partial_table_m
                        + kv_sparse_idx * stride_partial_table_n
                    )
                    safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
                    offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
                    offs_n_in_block = offs_n - kv_sparse_idx * SPARSE_KV_BLOCK_SIZE
                    mask = load_packed_partial_mask(
                        PARTIAL_MASK_PACKED,
                        stride_partial_p,
                        stride_partial_m,
                        stride_partial_n,
                        safe_partial_block_idx,
                        offs_m_in_block,
                        offs_n_in_block,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                    )
                    mask = mask & (partial_block_idx >= 0)
                else:
                    mask = load_dense_mask(
                        DENSE_MASK,
                        stride_mask_m,
                        stride_mask_n,
                        offs_m,
                        offs_n,
                        Q_LEN=Q_LEN,
                        KV_LEN=KV_LEN,
                    )
                qk = tl.where(mask, qk, float("-inf"))
                qk = tl.where(offs_n[:, None] < KV_LEN, qk, float("-inf"))

                p = tl.math.exp(qk - lse[:, None]).to(V.dtype.element_ty)
                dp = tl.dot(do, tl.trans(v), input_precision="ieee").to(V.dtype.element_ty)
                ds = (p * (dp - delta[:, None])).to(V.dtype.element_ty)
                ds = tl.where(mask, ds, 0.0)

                dq = tl.dot(ds.to(Q.dtype.element_ty), k, input_precision="ieee")
                tl.atomic_add(
                    DQ_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    dq,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                )

                dk = tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q, input_precision="ieee")
                tl.atomic_add(
                    DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                    dk,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                )

                dv = tl.dot(tl.trans(p).to(V.dtype.element_ty), do, input_precision="ieee")
                tl.atomic_add(
                    DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                    dv,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                )

            if HAS_FULL_BLOCKS:
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )

                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)

                    q = tl.load(
                        Q_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                        other=0.0,
                    )
                    do = tl.load(
                        DO_h + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                        other=0.0,
                    )
                    lse = tl.load(LSE_h + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
                    delta = tl.load(DELTA_h + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
                    lse = tl.where(lse == float("-inf"), 0.0, lse)

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE
                    qk = tl.where(offs_n[:, None] < KV_LEN, qk, float("-inf"))

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])

                    dq = tl.dot(ds.to(Q.dtype.element_ty), k, input_precision="ieee")
                    tl.atomic_add(
                        DQ_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                        dq,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    )

                    dk = tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q, input_precision="ieee")
                    tl.atomic_add(
                        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                        dk,
                        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    )

                    dv = tl.dot(tl.trans(p).to(V.dtype.element_ty), do, input_precision="ieee")
                    tl.atomic_add(
                        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                        dv,
                        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    )


@triton.jit
def bwd_dkdv_block_mn(
    Q, DO, DQ, DK_ptr, DELTA, LSE, DV_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
    stride_qm, stride_qk, stride_dom, stride_dok, stride_dqm, stride_dqd,
    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
    MATMUL_PRECISION,
    SM_SCALE: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
    COMPUTE_DQ: tl.constexpr = True,
):
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        other=0.0,
    )
    do = tl.load(
        DO + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    lse = tl.load(LSE + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == float("-inf"), 0.0, lse)

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    mask = True
    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_sparse_idx * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
            offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask, qk, float("-inf"))      # & (offs_n[None, :] < KV_LEN)
    # else:
    #     qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

    p = tl.math.exp(qk - lse[:, None])

    dv = tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do, input_precision="ieee")
    tl.atomic_add(
        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
        dv,
        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
    )

    Di = tl.load(DELTA + offs_m, mask=offs_m < Q_LEN, other=0.0)
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = (p * (dp - Di[:, None]))
    ds *= SM_SCALE

    # if not IS_FULL_BLOCKS:
    #     ds = tl.where(mask, ds, 0.0)

    if COMPUTE_DQ:
        dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
        # dq *= SM_SCALE
        tl.atomic_add(
            DQ + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqd,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )

    dk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
    # dk *= SM_SCALE
    tl.atomic_add(
        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk,
        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dqdkdv_kernel_b64(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = False,
    PERSISTENT_MODE: tl.constexpr = True,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                q_offset = off_z * stride_qz + off_hq * stride_qh
                do_offset = off_z * stride_doz + off_hq * stride_doh
                dq_offset = off_z * stride_qz + off_hq * stride_qh
                lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_h = Q + q_offset
                DQ_h = DQ + dq_offset
                DO_h = DO + do_offset
                LSE_h = LSE + lse_offset
                DELTA_h = DELTA + delta_offset

                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )
                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_dkdv_block_mn(
                        Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                    )

                if HAS_FULL_BLOCKS:
                    q_indices = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end = tl.minimum(
                        q_num_blocks * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                    )

                    for start_m in range(0, block_m_end):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)

                        bwd_dkdv_block_mn(
                            Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        )


class FlexAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        block_mask=None,
        score_mod=None,
        mask_type="full",
        doc_start=None,
        sliding_window=None,
        global_window=None
    ):
        # assert q.dim() == 4, "Q must be 4D tensor"
        # assert k.dim() == 4, "K must be 4D tensor"
        # assert v.dim() == 4, "V must be 4D tensor"
        # del score_mod

        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape

        GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
        # assert k.shape == v.shape, "K and V must have same shape"

        SM_SCALE = 1.0 / (D ** 0.5)

        SPARSE_Q_BLOCK_SIZE = 128
        SPARSE_KV_BLOCK_SIZE = 128
        BLOCK_M = SPARSE_Q_BLOCK_SIZE
        BLOCK_N = SPARSE_KV_BLOCK_SIZE

        num_q_blocks = (M + SPARSE_Q_BLOCK_SIZE - 1) // SPARSE_Q_BLOCK_SIZE

        output = torch.empty_like(q)
        lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

        kv_num_blks = block_mask.kv_num_blocks
        kv_idx = block_mask.kv_indices
        full_kv_num_blks = getattr(block_mask, "full_kv_num_blocks", torch.zeros_like(kv_num_blks))
        full_kv_idx = getattr(block_mask, "full_kv_indices", torch.zeros_like(kv_idx))

        q_num_blks = getattr(block_mask, "q_num_blocks", None)
        q_idx = getattr(block_mask, "q_indices", None)
        # assert q_num_blks is not None, "q_num_blocks and q_indices must be provided"
        # assert q_idx is not None, "q_indices must be provided"
        full_q_num_blks = getattr(block_mask, "full_q_num_blocks", torch.zeros_like(q_num_blks))
        full_q_idx = getattr(block_mask, "full_q_indices", torch.zeros_like(q_idx))

        # q = q.contiguous()
        # k = k.contiguous()
        # v = v.contiguous()

        # kv_num_blks = kv_num_blks.contiguous()
        # kv_idx = kv_idx.contiguous()
        # full_kv_num_blks = full_kv_num_blks.contiguous()
        # full_kv_idx = full_kv_idx.contiguous()

        # q_num_blks = q_num_blks.contiguous()
        # q_idx = q_idx.contiguous()
        # full_q_num_blks = full_q_num_blks.contiguous()
        # full_q_idx = full_q_idx.contiguous()

        dense_mask = getattr(block_mask, "dense_mask", None)
        packed_partial_mask = getattr(block_mask, "packed_partial_mask", None)
        partial_mask_offsets = getattr(block_mask, "partial_mask_offsets", None)
        partial_block_table = getattr(block_mask, "partial_block_table", None)
        use_packed_partial_mask = (
            packed_partial_mask is not None and
            partial_mask_offsets is not None and
            partial_block_table is not None
        )

        num_tasks = num_q_blocks * Z * Hq
        grid, num_tasks = _persistent_launch_config(num_tasks)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{grid=},{num_tasks=}")

        # kv_num_blks = kv_num_blks.to(torch.int32)
        # kv_idx = kv_idx.to(torch.int32)
        # full_kv_num_blks = full_kv_num_blks.to(torch.int32)
        # full_kv_idx = full_kv_idx.to(torch.int32)
        # q_num_blks = q_num_blks.to(torch.int32)
        # q_idx = q_idx.to(torch.int32)
        # full_q_num_blks = full_q_num_blks.to(torch.int32)
        # full_q_idx = full_q_idx.to(torch.int32)

        if dense_mask is None:
            dense_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=q.device)
        # dense_mask = dense_mask.contiguous()

        # if use_packed_partial_mask:
        #     packed_partial_mask = packed_partial_mask.contiguous()
        #     # partial_mask_offsets = partial_mask_offsets.to(torch.int32).contiguous()
        #     # partial_block_table = partial_block_table.to(torch.int32).contiguous()
        #     partial_mask_offsets = partial_mask_offsets.contiguous()
        #     partial_block_table = partial_block_table.contiguous()
        # else:
        #     packed_partial_mask = torch.zeros(
        #         (1, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE),
        #         dtype=torch.bool,
        #         device=q.device,
        #     )
        #     partial_mask_offsets = torch.zeros(
        #         (1, 1, max(num_q_blocks, 1)),
        #         dtype=torch.int32,
        #         device=q.device,
        #     )
        #     partial_block_table = torch.full(
        #         (max(num_q_blocks, 1), max((N + SPARSE_KV_BLOCK_SIZE - 1) // SPARSE_KV_BLOCK_SIZE, 1)),
        #         -1,
        #         dtype=torch.int32,
        #         device=q.device,
        #     )

        flex_attention_kernel[grid](
            q, k, v,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            dense_mask, dense_mask.stride(2), dense_mask.stride(3),
            packed_partial_mask, partial_mask_offsets,
            packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
            partial_mask_offsets.stride(2),
            output, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            kv_idx.stride(2),
            SM_SCALE = SM_SCALE,
            QK_HEAD_DIM = D,
            V_HEAD_DIM = Dv,
            BLOCK_M = BLOCK_M,
            BLOCK_N = BLOCK_N,
            NUM_TASKS = num_tasks,
            NUM_Q_BLOCKS = num_q_blocks,
            Q_HEAD = Hq,
            SPARSE_Q_BLOCK_SIZE = SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE = SPARSE_KV_BLOCK_SIZE,
            Q_LEN = M,
            KV_LEN = N,
            GQA_SHARED_HEADS = GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS = True,
            USE_PACKED_PARTIAL_MASK = use_packed_partial_mask,
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            intra_cache_num=3,
            inter_cache_num=2,
            enable_cross_if_fusion=True,
            enable_buffer_insert_optimization=True,
            enable_ub_refine_opt = True, 
            # ------------------------
            # enable_dynamic_cv_pipeline=False,
            # enable_preload=True,
            # hfusion_enable_multiple_consumer_fusion=True,
        )

        # if use_packed_partial_mask:
        #     dense_mask_for_save = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=q.device)
        # else:
        #     dense_mask_for_save = dense_mask

        ctx.save_for_backward(
            q, k, v, output, lse,
            dense_mask, packed_partial_mask, partial_mask_offsets, partial_block_table,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx
        )
        ctx.mask_type = mask_type
        ctx.sliding_window = sliding_window
        ctx.global_window = global_window
        ctx.gqa_shared_heads = GQA_SHARED_HEADS
        ctx.sm_scale = SM_SCALE
        ctx.sparse_q_block_size = SPARSE_Q_BLOCK_SIZE
        ctx.sparse_kv_block_size = SPARSE_KV_BLOCK_SIZE
        ctx.has_full_blocks = True
        ctx.use_packed_partial_mask = use_packed_partial_mask

        # Pre-compute Q-split K to avoid D2H sync in backward
        ctx.qsplit_k, ctx.qsplit_start = _compute_qsplit_k(q_num_blks, full_q_num_blks, M, N, Hkv, GQA_SHARED_HEADS)

        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse=None):
        (
            q, k, v, output, lse,
            dense_mask, packed_partial_mask, partial_mask_offsets, partial_block_table,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx
        ) = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        delta = (output * grad_output).sum(dim=-1).to(torch.float32).contiguous()

        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape
        GQA_SHARED_HEADS = ctx.gqa_shared_heads
        persistent_kernel = True
        if persistent_kernel:
            dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
            dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)

            BLOCK_M_DKDV = ctx.sparse_q_block_size
            BLOCK_N_DKDV = ctx.sparse_q_block_size
            NUM_KV_SUB_BLOCKS_VAL = ctx.sparse_kv_block_size // BLOCK_N_DKDV
            num_kv_blocks = triton.cdiv(N, ctx.sparse_kv_block_size)

            if _USE_FUSED_BACKWARD:
                # ============================================================
                # Path B: Fused dq+dk+dv kernel + reduce_dq
                # ============================================================
                dq_partial = torch.zeros(
                    (num_kv_blocks, Z, Hq, M, D), dtype=torch.float32, device=q.device,
                )

                num_tasks_fused = num_kv_blocks * Z * Hkv
                grid_fused, _ = _persistent_launch_config(num_tasks_fused)
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>fused>>>>>>>>>{grid_fused=},{num_tasks_fused=}")
                flex_attention_backward_dqdkdv_fused_kernel[grid_fused](
                    q, k, v, grad_output, lse, delta,
                    q_num_blks, q_idx, full_q_num_blks, full_q_idx,
                    dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                    packed_partial_mask, partial_mask_offsets, partial_block_table,
                    packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                    partial_mask_offsets.stride(2),
                    partial_block_table.stride(0), partial_block_table.stride(1),
                    dk, dv, dq_partial,
                    dq_partial.stride(0),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    delta.stride(0), delta.stride(1), delta.stride(2),
                    q_idx.stride(2),
                    SM_SCALE=ctx.sm_scale,
                    QK_HEAD_DIM=D,
                    V_HEAD_DIM=Dv,
                    BLOCK_M=BLOCK_M_DKDV,
                    BLOCK_N=BLOCK_N_DKDV,
                    NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
                    NUM_TASKS=num_tasks_fused,
                    NUM_KV_BLOCKS=num_kv_blocks,
                    KV_HEAD=Hkv,
                    SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                    SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                    Q_LEN=M,
                    KV_LEN=N,
                    GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                    HAS_FULL_BLOCKS=ctx.has_full_blocks,
                    USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                    limit_auto_multi_buffer_buffer="no-limit",
                    hfusion_enable_multiple_consumer_fusion=True,
                    intra_cache_num=2,
                    inter_cache_num=1,
                )

                # Reduce dq: sum dq_partial over kv_blocks → dq
                BLOCK_M_REDUCE = 64
                num_m_tiles = triton.cdiv(M, BLOCK_M_REDUCE)
                num_dq_reduce_tasks = num_m_tiles * Z * Hq
                grid_dq_reduce, _ = _persistent_launch_config(num_dq_reduce_tasks)
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>reduce_dq>>>>>>>>>{grid_dq_reduce=},{num_dq_reduce_tasks=}")
                dq = torch.empty_like(q)
                reduce_dq_kernel[grid_dq_reduce](
                    dq,
                    dq_partial,
                    dq_partial.stride(0),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    num_dq_reduce_tasks, M, num_kv_blocks,
                    Q_HEAD=Hq,
                    BLOCK_M=BLOCK_M_REDUCE,
                    QK_HEAD_DIM=D,
                )
                del dq_partial

            else:
                # ============================================================
                # Path A: Separate dq kernel + qsplit dkdv + reduce_dkdv
                # ============================================================
                # ---- 1. dq kernel ----
                dq = torch.empty_like(q)
                BLOCK_M_DQ = ctx.sparse_q_block_size
                BLOCK_N_DQ = ctx.sparse_q_block_size
                NUM_KV_SUB_BLOCKS_DQ = ctx.sparse_kv_block_size // BLOCK_N_DQ
                num_q_blocks = triton.cdiv(M, BLOCK_M_DQ)
                grid_dq, num_tasks_dq = _persistent_launch_config(num_q_blocks * Z * Hq)
                flex_attention_backward_dq_kernel[grid_dq](
                    q, k, v, grad_output, lse, delta,
                    kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
                    dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                    packed_partial_mask, partial_mask_offsets, partial_block_table,
                    packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                    partial_mask_offsets.stride(2),
                    partial_block_table.stride(0), partial_block_table.stride(1),
                    dq,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    delta.stride(0), delta.stride(1), delta.stride(2),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    kv_idx.stride(2),
                    SM_SCALE=ctx.sm_scale,
                    QK_HEAD_DIM=D,
                    V_HEAD_DIM=Dv,
                    BLOCK_M=BLOCK_M_DQ,
                    BLOCK_N=BLOCK_N_DQ,
                    NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_DQ,
                    NUM_TASKS=num_tasks_dq,
                    NUM_Q_BLOCKS=num_q_blocks,
                    Q_HEAD=Hq,
                    SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                    SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                    Q_LEN=M,
                    KV_LEN=N,
                    GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                    HAS_FULL_BLOCKS=ctx.has_full_blocks,
                    USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                    limit_auto_multi_buffer_buffer="no-limit",
                    hfusion_enable_multiple_consumer_fusion=True,
                    intra_cache_num=3,
                    inter_cache_num=2,
                    enable_cross_if_fusion=True,
                    enable_buffer_insert_optimization=True,
                    enable_ub_refine_opt=True,
                )

                # ---- 2. dkdv kernel (qsplit fused: lock-based ordered accumulation) ----
                # Replaces two-stage (dkdv_qsplit + reduce_dkdv) with single fused kernel.
                # DK/DV are pre-zeroed (outer scope); split tasks use spinlock to
                # accumulate in sub_id order, deterministic with original reduce kernel.
                K_SPLIT = ctx.qsplit_k
                SPLIT_START = ctx.qsplit_start

                if K_SPLIT > 1:
                    # dk/dv are already zeroed in outer scope (line ~2654)
                    total_base = num_kv_blocks * Z * Hkv
                    num_split_base = total_base - SPLIT_START

                    # Spinlock buffer: one int32 per (split_base_id, kv_sub), pre-zeroed.
                    # LOCKS[slot] == sub_id means it's sub_id's turn (initial 0 => sub_id=0 first).
                    num_locks = num_split_base * NUM_KV_SUB_BLOCKS_VAL
                    locks = torch.zeros(num_locks, dtype=torch.int32, device=k.device)

                    num_tasks_dkv = SPLIT_START + num_split_base * K_SPLIT
                    grid_dkv, _ = _persistent_launch_config(num_tasks_dkv)
                    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>dkdv_fused>>>>>>>>>{grid_dkv=},{num_tasks_dkv=}, K_SPLIT={K_SPLIT}, SPLIT_START={SPLIT_START}, total_base={total_base}, num_locks={num_locks}")
                    flex_attention_backward_dkdv_kernel_qsplit_fused[grid_dkv](
                        q, k, v, grad_output, lse, delta,
                        q_num_blks, q_idx, full_q_num_blks, full_q_idx,
                        dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                        packed_partial_mask, partial_mask_offsets, partial_block_table,
                        packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                        partial_mask_offsets.stride(2),
                        partial_block_table.stride(0), partial_block_table.stride(1),
                        dk, dv,
                        locks,
                        num_locks,
                        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                        lse.stride(0), lse.stride(1), lse.stride(2),
                        delta.stride(0), delta.stride(1), delta.stride(2),
                        q_idx.stride(2),
                        SM_SCALE=ctx.sm_scale,
                        QK_HEAD_DIM=D,
                        V_HEAD_DIM=Dv,
                        BLOCK_M=BLOCK_M_DKDV,
                        BLOCK_N=BLOCK_N_DKDV,
                        NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
                        NUM_TASKS=num_tasks_dkv,
                        NUM_KV_BLOCKS=num_kv_blocks,
                        KV_HEAD=Hkv,
                        SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                        SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                        Q_LEN=M,
                        KV_LEN=N,
                        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                        K_SPLIT=K_SPLIT,
                        SPLIT_START=SPLIT_START,
                        HAS_FULL_BLOCKS=ctx.has_full_blocks,
                        USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                        limit_auto_multi_buffer_buffer="no-limit",
                        hfusion_enable_multiple_consumer_fusion=True,
                        limit_auto_multi_buffer_of_local_buffer="no-l0c",
                        intra_cache_num=2,
                        inter_cache_num=1,
                    )
                    # reduce_dkdv_kernel is no longer needed: accumulation is fused.

                else:
                    grid_dkv, num_tasks_dkv = _persistent_launch_config(num_kv_blocks * Z * Hkv)
                    flex_attention_backward_dkdv_kernel[grid_dkv](
                        q, k, v, grad_output, lse, delta,
                        q_num_blks, q_idx, full_q_num_blks, full_q_idx,
                        dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                        packed_partial_mask, partial_mask_offsets, partial_block_table,
                        packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                        partial_mask_offsets.stride(2),
                        partial_block_table.stride(0), partial_block_table.stride(1),
                        dq, dk, dv,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                        lse.stride(0), lse.stride(1), lse.stride(2),
                        delta.stride(0), delta.stride(1), delta.stride(2),
                        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                        q_idx.stride(2),
                        SM_SCALE=ctx.sm_scale,
                        QK_HEAD_DIM=D,
                        V_HEAD_DIM=Dv,
                        BLOCK_M=BLOCK_M_DKDV,
                        BLOCK_N=BLOCK_N_DKDV,
                        NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
                        NUM_TASKS=num_tasks_dkv,
                        NUM_KV_BLOCKS=num_kv_blocks,
                        KV_HEAD=Hkv,
                        SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                        SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                        Q_LEN=M,
                        KV_LEN=N,
                        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                        HAS_FULL_BLOCKS=ctx.has_full_blocks,
                        USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                        limit_auto_multi_buffer_buffer="no-limit",
                        hfusion_enable_multiple_consumer_fusion=True,
                        unit_flag=True,
                        limit_auto_multi_buffer_of_local_buffer="no-l0c",
                        intra_cache_num=2,
                        inter_cache_num=1,
                    )

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None


def flex_attention(
    q,
    k,
    v,
    block_mask=None,
    score_mod=None,
    return_lse=None,
    mask_type="full",
    doc_start=None,
    sliding_window=None,
    global_window=None,
):
    """
    Args:
        q: Query tensor [Z, Hq, M, D]
        k: Key tensor [Z, Hkv, N, D]
        v: Value tensor [Z, Hkv, N, Dv]
        block_mask: Block mask from torch.nn.attention.flex_attention
        score_mod: Optional score modification function

    Returns:
        Output tensor [Z, Hq, N, Dv]
    """
    output, lse = FlexAttentionFunc.apply(
        q,
        k,
        v,
        block_mask,
        score_mod,
        mask_type,
        doc_start,
        sliding_window,
        global_window,
    )
    if return_lse:
        return output, lse
    return output