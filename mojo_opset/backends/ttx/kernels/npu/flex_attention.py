from typing import Optional
from typing import Tuple

import math

import torch

import triton
import triton.language as tl

from .utils import get_num_cores


TILE_BLOCK_SIZE = 128


def _get_num_aicore():
    try:
        return max(get_num_cores(op_type="cube"), 1)
    except Exception:
        return 1


def _persistent_launch_config(num_tasks):
    num_tasks = max(int(num_tasks), 1)
    return (min(_get_num_aicore(), num_tasks),), num_tasks


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

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),
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
                mask=(offs_n_load[:, None] < KV_LEN),
                other=0.0
            )
            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN),
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
                    mask=(offs_n_load[:, None] < KV_LEN),
                    other=0.0
                )
                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN),
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
        mask=(offs_n[:, None] < KV_LEN),
        other=0.0,
    )
    v = tl.load(
        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
        mask=(offs_n[:, None] < KV_LEN),
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
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0,
        )
        do = tl.load(
            DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
            mask=(offs_m[:, None] < Q_LEN),
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
                    mask=(offs_n[:, None] < KV_LEN),
                    other=0.0,
                )
                v = tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n[:, None] < KV_LEN),
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
                qk = tl.where(mask, qk, float("-inf"))

                p = tl.math.exp(qk - lse[:, None])
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta[:, None])
                ds *= SM_SCALE
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
                        mask=(offs_n[:, None] < KV_LEN),
                        other=0.0,
                    )
                    v = tl.load(
                        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=(offs_n[:, None] < KV_LEN),
                        other=0.0,
                    )

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])
                    ds *= SM_SCALE
                    dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

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


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS", "NUM_CORES",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dkdv_kernel_ordered(
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
    TASK_KV,
    TASK_START_ORDER,
    TASK_END_ORDER,
    TASK_IS_SPLIT,
    TASK_CORE_START,
    COUNT,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    NUM_CORES,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    """优化版 dkdv kernel：混合调度。

    非拆分 task：kv_sub 外层循环，K/V 复用（像原 kernel），无需 COUNT
    拆分 task：q_block 外层循环，COUNT 保序累加
    """
    pid = tl.program_id(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

    # 读取本核的 task 范围
    task_start = tl.load(TASK_CORE_START + pid)
    task_end = tl.load(TASK_CORE_START + pid + 1)

    for task_idx in range(task_start, task_end):
        kv_task = tl.load(TASK_KV + task_idx)
        start_order = tl.load(TASK_START_ORDER + task_idx)
        end_order = tl.load(TASK_END_ORDER + task_idx)
        is_split = tl.load(TASK_IS_SPLIT + task_idx)

        # 拆解 kv_task -> (z, hkv, kv_block_idx)
        kv_block_idx = kv_task % NUM_KV_BLOCKS
        off_hkv = (kv_task // NUM_KV_BLOCKS) % KV_HEAD
        off_z = kv_task // (NUM_KV_BLOCKS * KV_HEAD)

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

        start_n_full = kv_block_idx * KV_BLOCK_SIZE
        kv_sparse_idx = kv_block_idx // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        count_offset = off_z * (KV_HEAD * NUM_KV_BLOCKS) + off_hkv * NUM_KV_BLOCKS + kv_block_idx
        count_offset = count_offset.to(tl.int64)

        q_num_blocks_sparse = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)

        if is_split != 0:
            # ====== 拆分路径：q_block 外层循环 + COUNT 保序 ======
            my_order = start_order
            while my_order < end_order:
                # 自旋等待轮到自己
                while tl.load(COUNT + count_offset) != my_order:
                    pass

                is_full = my_order >= q_num_blocks_sparse

                if is_full:
                    if not HAS_FULL_BLOCKS:
                        tl.atomic_add(COUNT + count_offset, 1)
                        my_order = my_order + 1
                        continue
                    order_in_full = my_order - q_num_blocks_sparse
                    q_indices = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end = tl.minimum(
                        q_num_blocks * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                    )
                    if order_in_full >= block_m_end:
                        tl.atomic_add(COUNT + count_offset, 1)
                        my_order = my_order + 1
                        continue
                    start_m = order_in_full
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)

                    for kv_sub in range(NUM_KV_SUB_BLOCKS):
                        start_n = start_n_full + kv_sub * BLOCK_N
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

                            bwd_dkdv_block_mn_ordered(
                                Q_h, DO_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                                DENSE_MASK, stride_mask_m, stride_mask_n,
                                PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                                PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                                k, v, Q_LEN, KV_LEN,
                                off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
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
                else:
                    # sparse 段
                    q_indices = Q_IDX + sparse_q_idx_offset
                    start_m = my_order
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    for kv_sub in range(NUM_KV_SUB_BLOCKS):
                        start_n = start_n_full + kv_sub * BLOCK_N
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

                            bwd_dkdv_block_mn_ordered(
                                Q_h, DO_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                                DENSE_MASK, stride_mask_m, stride_mask_n,
                                PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                                PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                                k, v, Q_LEN, KV_LEN,
                                off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
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

                # 通知下一个 order
                tl.atomic_add(COUNT + count_offset, 1)
                my_order = my_order + 1
        else:
            # ====== 非拆分路径：kv_sub 外层循环，K/V 复用（像原 kernel）======
            for kv_sub in range(NUM_KV_SUB_BLOCKS):
                start_n = start_n_full + kv_sub * BLOCK_N
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

                    # sparse q_blocks
                    q_indices = Q_IDX + sparse_q_idx_offset
                    q_num_blocks = q_num_blocks_sparse
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

                        bwd_dkdv_block_mn_ordered(
                            Q_h, DO_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
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

                    # full q_blocks
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

                            bwd_dkdv_block_mn_ordered(
                                Q_h, DO_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                                DENSE_MASK, stride_mask_m, stride_mask_n,
                                PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                                PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                                k, v, Q_LEN, KV_LEN,
                                off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
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

    if COMPUTE_DQ:
        dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
        tl.atomic_add(
            DQ + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqd,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )

    dk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
    tl.atomic_add(
        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk,
        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )


@triton.jit
def bwd_dkdv_block_mn_ordered(
    Q, DO, DK_ptr, DELTA, LSE, DV_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
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
    """Step 4: dkdv block 计算（方案 Y：保序锁在 kernel 层，block 内用 atomic_add）。

    与 bwd_dkdv_block_mn 的差异：
      - 去掉 DQ/COMPUTE_DQ（dkdv 不算 dq）
      - DK/DV 累加用 atomic_add（同一 q_block 的不同 off_g 写同一位置）
    保序由 kernel 层的自旋锁 + atomic_add(COUNT, 1) 保证（q_block 之间严格按 order 顺序）。
    """
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

    dk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
    tl.atomic_add(
        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk,
        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )


def _prepare_block_mask_attrs(block_mask, q, num_q_blocks, sparse_q_block_size, sparse_kv_block_size):
    N = q.shape[0] if q.dim() == 4 else q.shape[2]
    kv_num_blks = block_mask.kv_num_blocks
    kv_idx = block_mask.kv_indices
    full_kv_num_blks = getattr(block_mask, "full_kv_num_blocks", torch.zeros_like(kv_num_blks))
    full_kv_idx = getattr(block_mask, "full_kv_indices", torch.zeros_like(kv_idx))

    q_num_blks = getattr(block_mask, "q_num_blocks", None)
    q_idx = getattr(block_mask, "q_indices", None)
    assert q_num_blks is not None, "q_num_blocks and q_indices must be provided"
    assert q_idx is not None, "q_indices must be provided"
    full_q_num_blks = getattr(block_mask, "full_q_num_blocks", torch.zeros_like(q_num_blks))
    full_q_idx = getattr(block_mask, "full_q_indices", torch.zeros_like(q_idx))

    kv_num_blks = kv_num_blks.to(torch.int32).contiguous()
    kv_idx = kv_idx.to(torch.int32).contiguous()
    full_kv_num_blks = full_kv_num_blks.to(torch.int32).contiguous()
    full_kv_idx = full_kv_idx.to(torch.int32).contiguous()
    q_num_blks = q_num_blks.to(torch.int32).contiguous()
    q_idx = q_idx.to(torch.int32).contiguous()
    full_q_num_blks = full_q_num_blks.to(torch.int32).contiguous()
    full_q_idx = full_q_idx.to(torch.int32).contiguous()

    dense_mask = getattr(block_mask, "dense_mask", None)
    packed_partial_mask = getattr(block_mask, "packed_partial_mask", None)
    partial_mask_offsets = getattr(block_mask, "partial_mask_offsets", None)
    partial_block_table = getattr(block_mask, "partial_block_table", None)
    use_packed_partial_mask = (
        packed_partial_mask is not None
        and partial_mask_offsets is not None
        and partial_block_table is not None
    )

    if dense_mask is None:
        dense_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=q.device)
    dense_mask = dense_mask.contiguous()

    if use_packed_partial_mask:
        packed_partial_mask = packed_partial_mask.contiguous()
        partial_mask_offsets = partial_mask_offsets.to(torch.int32).contiguous()
        partial_block_table = partial_block_table.to(torch.int32).contiguous()
    else:
        packed_partial_mask = torch.zeros(
            (1, sparse_q_block_size, sparse_kv_block_size),
            dtype=torch.bool,
            device=q.device,
        )
        partial_mask_offsets = torch.zeros(
            (1, 1, max(num_q_blocks, 1)),
            dtype=torch.int32,
            device=q.device,
        )
        partial_block_table = torch.full(
            (max(num_q_blocks, 1), max((N + sparse_kv_block_size - 1) // sparse_kv_block_size, 1)),
            -1,
            dtype=torch.int32,
            device=q.device,
        )

    return {
        "kv_num_blks": kv_num_blks,
        "kv_idx": kv_idx,
        "full_kv_num_blks": full_kv_num_blks,
        "full_kv_idx": full_kv_idx,
        "q_num_blks": q_num_blks,
        "q_idx": q_idx,
        "full_q_num_blks": full_q_num_blks,
        "full_q_idx": full_q_idx,
        "dense_mask": dense_mask,
        "packed_partial_mask": packed_partial_mask,
        "partial_mask_offsets": partial_mask_offsets,
        "partial_block_table": partial_block_table,
        "use_packed_partial_mask": use_packed_partial_mask,
    }


def compute_k_count(bm, num_kv_blocks):
    """Step 1: 生成 K 轴累加次数矩阵 K_COUNT[Z, Hkv, NUM_KV_BLOCKS]。

    每个元素 = 该结果位置 (z, hkv, kv_block) 的有效 Q block 数
    （= sparse/partial Q block 数 + full Q block 数）。
    语义等价于扫描 BLOCK_FLAGS 统计非空 block（flag=1 partial + flag=2 full）。
    后续用作保序累加计数器的初值依据，以及负载判定的输入。
    """
    q_num_blks = bm["q_num_blks"]
    full_q_num_blks = bm["full_q_num_blks"]
    k_count = (q_num_blks + full_q_num_blks).to(torch.int32).contiguous()
    return k_count


def is_load_imbalanced(k_count, num_cores, threshold=0.2):
    """Step 2: 判断按 KV 方向分核后是否负载不均。

    判定逻辑：把每个 (z, hkv) 视作一组，组内 NUM_KV_BLOCKS 个任务的计算量
    以 K_COUNT（有效 Q block 数）衡量。静态调度下每个核被分配
    ceil(num_kv_blocks / num_cores) 个任务，取各核任务量上限 max_load 与
    各任务量平均 avg_load，若 (max_load - avg_load) / avg_load > threshold
    则判为不均，走新算子。

    Args:
        k_count: [Z, 1, NUM_KV_BLOCKS] int32，每个 kv_block 的有效 Q block 数
        num_cores: 物理核数（KV 方向分核数）
        threshold: 计算量差异阈值，默认 0.2（20%）
    Returns:
        bool: True 表示负载不均，应走新算子
    """
    if num_cores <= 1:
        return False

    k_count_flat = k_count.reshape(-1, k_count.shape[-1])  # [Z*Hkv, NUM_KV_BLOCKS]
    num_kv_blocks = k_count_flat.shape[-1]
    if num_kv_blocks == 0:
        return False

    for row in range(k_count_flat.shape[0]):
        loads = k_count_flat[row].to(torch.float32)
        avg_load = loads.mean().item()
        if avg_load <= 0:
            continue

        # 静态调度：前 num_kv_blocks % num_cores 个核多分 1 个任务，
        # 取最大核负载 = max(前余数核 * (ceil+1), 其余核 * ceil)
        per_core = (num_kv_blocks + num_cores - 1) // num_cores
        remainder = num_kv_blocks % num_cores
        if remainder == 0:
            max_blocks = per_core
        else:
            max_blocks = per_core  # ceil 即最大

        # 最大核负载：贪心取最大的 max_blocks 个任务量之和
        sorted_loads, _ = loads.sort(descending=True)
        max_load = sorted_loads[:max_blocks].sum().item()
        if (max_load - avg_load) / avg_load > threshold:
            return True
    return False


def build_ordered_task_list(k_count, num_kv_blocks, num_kv_heads, num_cores=1, threshold=0.2):
    """Step 3 (优化版): 混合调度任务列表。

    策略：
      1. 尽量将完整 KV block 分配给单个核（保留 KV 数据复用，减少读取次数）
      2. 贪心装箱：每次把 KV block 分给当前负载最轻的核
      3. 对于过重的 KV block（K_COUNT > target_per_core * (1+threshold)），
         拆分到多核，使用计数保序累加
      4. 非拆分 task：单核处理整个 KV block 的所有 q_block，无需 COUNT
         拆分 task：多核处理同一 KV block 的不同 q_block 段，需 COUNT 保序

    Returns:
        task_kv: [TOTAL_TASKS] int32 - kv 任务 id
        task_start_order: [TOTAL_TASKS] int32 - 起始 order（inclusive）
        task_end_order: [TOTAL_TASKS] int32 - 结束 order（exclusive）
        task_is_split: [TOTAL_TASKS] int32 - 1=拆分（需计数保序），0=单核
        task_core_start: [num_cores+1] int32 - 每个核在 task 表中的起止位置
    """
    Z = k_count.shape[0]
    Hkv = num_kv_heads
    device = k_count.device

    k_count_expanded = k_count.expand(Z, Hkv, num_kv_blocks).contiguous()

    # 构建 (kv_task_id, k_count) 列表
    kv_tasks = []
    for z in range(Z):
        for hkv in range(Hkv):
            for kv_blk in range(num_kv_blocks):
                kv_task_id = z * (Hkv * num_kv_blocks) + hkv * num_kv_blocks + kv_blk
                kc = int(k_count_expanded[z, hkv, kv_blk].item())
                if kc > 0:
                    kv_tasks.append((kv_task_id, kc))

    # 按 k_count 降序排列（先处理重任务，利于装箱均衡）
    kv_tasks.sort(key=lambda x: x[1], reverse=True)

    total_tasks = sum(kc for _, kc in kv_tasks)
    if total_tasks == 0 or num_cores <= 0:
        num_cores = max(num_cores, 1)
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        empty_core = torch.zeros(num_cores + 1, dtype=torch.int32, device=device)
        return empty, empty, empty, empty, empty_core

    target_per_core = max(total_tasks / num_cores, 1.0)
    split_threshold = target_per_core * (1.0 + threshold)

    # 贪心装箱
    core_loads = [0] * num_cores
    core_assignments = [[] for _ in range(num_cores)]

    for kv_task_id, kc in kv_tasks:
        if kc > split_threshold:
            # 过重 KV block：拆分到多核
            num_splits = max(2, (kc + int(target_per_core) - 1) // int(target_per_core))
            orders_per_split = (kc + num_splits - 1) // num_splits
            for start in range(0, kc, orders_per_split):
                end = min(start + orders_per_split, kc)
                min_core = min(range(num_cores), key=lambda c: core_loads[c])
                core_assignments[min_core].append((kv_task_id, start, end, 1))
                core_loads[min_core] += (end - start)
        else:
            # 完整 KV block 分配给最轻的核
            min_core = min(range(num_cores), key=lambda c: core_loads[c])
            core_assignments[min_core].append((kv_task_id, 0, kc, 0))
            core_loads[min_core] += kc

    # 展平为 task 表（按 core 顺序排列）
    task_kv_list = []
    task_start_order_list = []
    task_end_order_list = []
    task_is_split_list = []
    task_core_start = [0] * (num_cores + 1)

    for core_id in range(num_cores):
        task_core_start[core_id] = len(task_kv_list)
        for kv_task_id, start_order, end_order, is_split in core_assignments[core_id]:
            task_kv_list.append(kv_task_id)
            task_start_order_list.append(start_order)
            task_end_order_list.append(end_order)
            task_is_split_list.append(is_split)
    task_core_start[num_cores] = len(task_kv_list)

    task_kv = torch.tensor(task_kv_list, dtype=torch.int32, device=device)
    task_start_order = torch.tensor(task_start_order_list, dtype=torch.int32, device=device)
    task_end_order = torch.tensor(task_end_order_list, dtype=torch.int32, device=device)
    task_is_split = torch.tensor(task_is_split_list, dtype=torch.int32, device=device)
    task_core_start = torch.tensor(task_core_start, dtype=torch.int32, device=device)

    return task_kv, task_start_order, task_end_order, task_is_split, task_core_start


def flex_attention_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape

    GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = TILE_BLOCK_SIZE
    BLOCK_N = TILE_BLOCK_SIZE
    SPARSE_Q_BLOCK_SIZE = BLOCK_M
    SPARSE_KV_BLOCK_SIZE = BLOCK_N

    num_q_blocks = (M + SPARSE_Q_BLOCK_SIZE - 1) // SPARSE_Q_BLOCK_SIZE

    output = torch.empty_like(q)
    lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    bm = _prepare_block_mask_attrs(block_mask, q, num_q_blocks, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)

    num_tasks = num_q_blocks * Z * Hq
    grid, num_tasks = _persistent_launch_config(num_tasks)

    flex_attention_kernel[grid](
        q, k, v,
        bm["kv_num_blks"], bm["kv_idx"], bm["full_kv_num_blks"], bm["full_kv_idx"],
        bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
        bm["packed_partial_mask"], bm["partial_mask_offsets"],
        bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
        bm["partial_mask_offsets"].stride(2),
        output, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        bm["kv_idx"].stride(2),
        SM_SCALE=sm_scale,
        QK_HEAD_DIM=D,
        V_HEAD_DIM=Dv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_TASKS=num_tasks,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_HEAD=Hq,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        Q_LEN=M,
        KV_LEN=N,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=True,
        USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True,
        intra_cache_num=3,
        inter_cache_num=2,
        enable_cross_if_fusion=True,
        enable_buffer_insert_optimization=True,
        enable_ub_refine_opt = True,
    )

    return output, lse


def flex_attention_bwd_impl(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape
    GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    grad_output = grad_output.contiguous()
    delta = (output * grad_output).sum(dim=-1).to(torch.float32).contiguous()

    SPARSE_Q_BLOCK_SIZE = TILE_BLOCK_SIZE
    SPARSE_KV_BLOCK_SIZE = TILE_BLOCK_SIZE
    num_q_blocks = triton.cdiv(M, SPARSE_Q_BLOCK_SIZE)

    bm = _prepare_block_mask_attrs(block_mask, q, num_q_blocks, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)

    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)

    BLOCK_M_DQ = TILE_BLOCK_SIZE
    BLOCK_N_DQ = TILE_BLOCK_SIZE
    NUM_KV_SUB_BLOCKS_VAL = SPARSE_KV_BLOCK_SIZE // BLOCK_N_DQ
    grid_dq, num_tasks_dq = _persistent_launch_config(num_q_blocks * Z * Hq)
    flex_attention_backward_dq_kernel[grid_dq](
        q, k, v, grad_output, lse, delta,
        bm["kv_num_blks"], bm["kv_idx"], bm["full_kv_num_blks"], bm["full_kv_idx"],
        bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
        bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
        bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
        bm["partial_mask_offsets"].stride(2),
        bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
        dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        bm["kv_idx"].stride(2),
        SM_SCALE=sm_scale,
        QK_HEAD_DIM=D,
        V_HEAD_DIM=Dv,
        BLOCK_M=BLOCK_M_DQ,
        BLOCK_N=BLOCK_N_DQ,
        NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
        NUM_TASKS=num_tasks_dq,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_HEAD=Hq,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        Q_LEN=M,
        KV_LEN=N,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=True,
        USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True,
        enable_select_analysis=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
        intra_cache_num=3,
        inter_cache_num=2,
    )

    BLOCK_M_DKDV = TILE_BLOCK_SIZE
    BLOCK_N_DKDV = TILE_BLOCK_SIZE
    NUM_KV_SUB_BLOCKS_VAL = SPARSE_KV_BLOCK_SIZE // BLOCK_N_DKDV
    num_kv_blocks = triton.cdiv(N, SPARSE_KV_BLOCK_SIZE)

    # Step 6: 负载判定，决定走原算子还是新算子
    k_count = compute_k_count(bm, num_kv_blocks)
    num_cores_dkdv = _get_num_aicore()
    use_ordered_kernel = is_load_imbalanced(k_count, num_cores_dkdv)

    if use_ordered_kernel:
        # 新算子：混合调度（非拆分 K/V 复用 + 拆分保序累加）
        task_kv, task_start_order, task_end_order, task_is_split, task_core_start = \
            build_ordered_task_list(k_count, num_kv_blocks, Hkv, num_cores_dkdv)
        num_tasks_ordered = task_kv.shape[0]
        total_count_slots = Z * Hkv * num_kv_blocks
        count_buffer = torch.zeros(total_count_slots, dtype=torch.int32, device=q.device)
        grid_dkv_ord = (num_cores_dkdv,)

        flex_attention_backward_dkdv_kernel_ordered[grid_dkv_ord](
            q, k, v, grad_output, lse, delta,
            bm["q_num_blks"], bm["q_idx"], bm["full_q_num_blks"], bm["full_q_idx"],
            bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
            bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
            bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
            bm["partial_mask_offsets"].stride(2),
            bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
            dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            bm["q_idx"].stride(2),
            task_kv, task_start_order, task_end_order, task_is_split, task_core_start,
            count_buffer,
            SM_SCALE=sm_scale,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
            NUM_TASKS=num_tasks_ordered,
            NUM_KV_BLOCKS=num_kv_blocks,
            NUM_CORES=num_cores_dkdv,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=True,
            USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=2,
            inter_cache_num=1,
        )
    else:
        # 原算子：静态调度
        grid_dkv, num_tasks_dkv = _persistent_launch_config(num_kv_blocks * Z * Hkv)
        flex_attention_backward_dkdv_kernel[grid_dkv](
            q, k, v, grad_output, lse, delta,
            bm["q_num_blks"], bm["q_idx"], bm["full_q_num_blks"], bm["full_q_idx"],
            bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
            bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
            bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
            bm["partial_mask_offsets"].stride(2),
            bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            bm["q_idx"].stride(2),
            SM_SCALE=sm_scale,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
            NUM_TASKS=num_tasks_dkv,
            NUM_KV_BLOCKS=num_kv_blocks,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=True,
            USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            #unit_flag=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=2,
            inter_cache_num=1,
        )

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)