import os
import torch
import pytest

from flex_attention_triton import APPLY_Q_CHUNK
from flex_attention_triton import flex_attention
from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask, create_mask
from torch.nn.attention.flex_attention import flex_attention as flex_attention_native
from torch.nn.attention.flex_attention import create_block_mask as create_block_mask_native
from torch.nn.attention import flex_attention as _fa_module
from typing import Optional
import triton
import triton.language as tl

# NPU 设备上 torch 原生 flex_attention 默认拒绝非 CUDA/CPU 设备,
# monkey-patch _validate_device 以允许 NPU (dynamo 在 NPU 上可用, eager 模式正常工作)
_fa_module._validate_device = lambda q, k, v: None


# ============================================================================
# 全局配置
# ============================================================================
# 模型参数
DTYPE = torch.bfloat16
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
SLIDING_WINDOW = 1024
GLOBAL_WINDOW = 4

# 数据布局: 2个样本, 每个样本 [text, image_gen, text]
# DATA_LENGTH = [[512, 1024, 512], [512, 1024, 512]]
# DATA_LENGTH = [[500, 1200, 300], [200, 1400, 300]]
# DATA_LENGTH = [[200, 6000, 300], [200, 6000, 300]]
# DATA_LENGTH = [[200, 16000, 300], [200, 16000, 300]]
# DATA_LENGTH = [[2000, 22000, 2000], [2000, 22000, 2000]]

# =======================131K 变长===========================
# DATA_LENGTH = [
#     [1500,  6000, 1500],    # 9000
#     [1300,  8400, 1300],    # 11000
#     [1100, 10800, 1100],    # 13000
#     [ 900, 13200,  900],    # 15000
#     [ 700, 15600,  700],    # 17000
#     [ 600, 17800,  600],    # 19000
#     [ 800, 21400,  800],    # 23000
#     [ 600, 22800,  600],    # 24000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"]
#                    ]
# RETURN_GRID = torch.tensor(1310000, dtype=DTYPE, device=torch.device("npu"))

# =================================64K==========================
# DATA_LENGTH = [
#     [1200,  5600, 1200],    # 8000
#     [1000,  7000, 1000],    # 9000
#     [ 800,  8400,  800],    # 10000
#     [ 600,  9800,  600],    # 11000
#     [ 700, 10600,  700],    # 12000
#     [ 700, 12600,  700],    # 14000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ]
# RETURN_GRID = torch.tensor(640000, dtype=DTYPE, device=torch.device("npu"))

# ==================================32K===========================
# DATA_LENGTH = [
#     [600, 2800, 600],    # 4000
#     [500, 3500, 500],    # 4500
#     [400, 4200, 400],    # 5000
#     [600, 4300, 600],    # 5500
#     [500, 5000, 500],    # 6000
#     [600, 5800, 600],    # 7000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ]
# RETURN_GRID = torch.tensor(320000, dtype=DTYPE, device=torch.device("npu"))

# ===================================16K=============================
# DATA_LENGTH = [
#     [500, 3000, 500],    # 4000
#     [600, 3800, 600],    # 5000
#     [800, 5400, 800],    # 7000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ]
# RETURN_GRID = torch.tensor(160000, dtype=DTYPE, device=torch.device("npu"))

# ===================================8K=============================
# DATA_LENGTH = [
#     [400, 2200, 400],    # 3000
#     [600, 3800, 600],    # 5000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ["text", "image_gen", "text"],
#                    ]
# RETURN_GRID = torch.tensor(80000, dtype=DTYPE, device=torch.device("npu"))

# ===================================4K=============================
# DATA_LENGTH = [
#     [500, 3000, 500],    # 4000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ]
# RETURN_GRID = torch.tensor(40000, dtype=DTYPE, device=torch.device("npu"))

# ===================================2K=============================
# DATA_LENGTH = [
#     [90, 1400, 510],    # 2000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ]
# RETURN_GRID = torch.tensor(20000, dtype=DTYPE, device=torch.device("npu"))

# ===================================1K=============================
# DATA_LENGTH = [
#     [90, 800, 110],    # 1000
# ]
# DATA_INPUT_TYPE = [["text", "image_gen", "text"],
#                    ]
# RETURN_GRID = torch.tensor(10000, dtype=DTYPE, device=torch.device("npu"))



FULL_MASK_MODALITIES = ("image_gen", "image_vae")

# zhichuang-small-case
DATA_LENGTH = [[90, 800, 110]]
RETURN_GRID = torch.tensor(1024, dtype=DTYPE, device=torch.device("npu"))
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
DATA_INPUT_TYPE = [["text", "image_gen", "text"]]
DATA_LENGTH_1 = [[72, 4056, 987]]

# Video帧布局: 2个样本, 每样本4个video, 每样本总长26000
# DATA_LENGTH_VIDEO 段分割与 VIDEO_FRAME_LENGTH 的 video 边界对齐 (每段=1个video=6500)
# 帧长取非128倍数 (如3000/2000/1500/6500), 使帧边界落在block中间产生partial blocks
DATA_LENGTH_VIDEO = [
    [6500, 6500, 6500, 6500],   # sample 0: 4 videos, total 26000
    [6500, 6500, 6500, 6500],   # sample 1: 4 videos, total 26000
]
VIDEO_FRAME_LENGTH = [
    # sample 0: 4 videos, 帧数 3/2/4/1
    [[3000, 2000, 1500], [4000, 2500], [1500, 1500, 1500, 2000], [6500]],
    # sample 1: 4 videos, 帧数 2/4/3/1
    [[3500, 3000], [1000, 2000, 1500, 2000], [2000, 2500, 2000], [6500]],
]

SEED = 0
CORRECTNESS_TOL = 2e-2
PROFILING = True
# native验证开关: flex_attention_native 在 NPU 上回退到 math attention,
# 长序列下 Q@K^T 物化 [B,H,S,S] float32 scores 矩阵 (52K序列需161GiB) → OOM.
# 设为 True 启用 torch native 基线对比; False 跳过 (仅对比 triton vs ascendc).
ENABLE_NATIVE = False
# native 运行设备: "npu" 直接在 NPU 上跑 (长序列易 OOM); "cpu" 搬到 CPU 跑 (慢但不会 OOM)
NATIVE_DEVICE = "cpu"

# backward梯度缩放因子(避免bf16下溢)

# RETURN_GRID = torch.tensor(520000, dtype=DTYPE, device=torch.device("npu"))
# RETURN_GRID = torch.tensor(24000, dtype=DTYPE, device=torch.device("npu"))
# RETURN_GRID = torch.tensor(5120, dtype=DTYPE, device=torch.device("npu"))
# RETURN_GRID = torch.tensor(1.0, dtype=DTYPE, device=torch.device("npu"))


# 性能测试配置
_WARMUP = 1
_ITERS = 3
_MB = 1024 ** 2

# Block划分参数 — 决定attention的块大小和mask的粒度
Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 128


def _get_num_vector_core():
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not hasattr(npu_mod, "current_device"):
        return 1
    device = npu_mod.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    return max(int(props.get("num_vectorcore", 1)), 1)


# ============================================================================
# Kernel 1: bool_count_nonzero_kernel
# 功能: 统计每个block中True的数量, 用于_convert_mask_to_block_mask的旧路径
# 输入: dense_mask [B,H,Q_NB,KV_NB,Q_BS,KV_BS] bool
# 输出: counts [B,H,Q_NB,KV_NB] int32
# ============================================================================
@triton.jit
def bool_count_nonzero_kernel(
    MASK,
    OUT,
    stride_mb,
    stride_mh,
    stride_mqb,
    stride_mkb,
    stride_mqi,
    stride_mki,
    stride_ob,
    stride_oh,
    stride_oqb,
    stride_okb,
    NUM_TASKS,
    H: tl.constexpr,
    Q_NUM_BLOCKS,
    KV_NUM_BLOCKS,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    num_blocks_per_bh = Q_NUM_BLOCKS * KV_NUM_BLOCKS

    for task_id in range(pid, NUM_TASKS, num_core):
        off_bh = task_id // num_blocks_per_bh
        off_inner = task_id % num_blocks_per_bh
        off_b = (off_bh // H).to(tl.int64)
        off_h = (off_bh % H).to(tl.int64)
        off_qb = (off_inner // KV_NUM_BLOCKS).to(tl.int64)
        off_kb = (off_inner % KV_NUM_BLOCKS).to(tl.int64)

        mask_base = (
            MASK
            + off_b * stride_mb
            + off_h * stride_mh
            + off_qb * stride_mqb
            + off_kb * stride_mkb
        )

        q_inner = tl.arange(0, Q_BLOCK_SIZE)[:, None]
        kv_inner = tl.arange(0, KV_BLOCK_SIZE)[None, :]
        ptrs = mask_base + q_inner * stride_mqi + kv_inner * stride_mki
        vals = tl.load(ptrs)
        count = tl.sum(tl.sum(vals.to(tl.int32), axis=1), axis=0)

        out_ptr = (
            OUT
            + off_b * stride_ob
            + off_h * stride_oh
            + off_qb * stride_oqb
            + off_kb * stride_okb
        )
        tl.store(out_ptr, count)


def triton_count_nonzero_last(new_mask):
    if new_mask.dtype != torch.bool:
        raise TypeError(f"new_mask 必须是 bool tensor, 当前为 {new_mask.dtype}")
    if new_mask.ndim != 6:
        raise ValueError(f"new_mask 必须是 6 维, 当前shape={tuple(new_mask.shape)}")

    new_mask = new_mask.contiguous()
    B, H, Q_NUM_BLOCKS, KV_NUM_BLOCKS, Q_BLOCK_SIZE, KV_BLOCK_SIZE = new_mask.shape
    out_i32 = torch.empty(
        (B, H, Q_NUM_BLOCKS, KV_NUM_BLOCKS),
        device=new_mask.device,
        dtype=torch.int32,
    )

    num_tasks = B * H * Q_NUM_BLOCKS * KV_NUM_BLOCKS
    grid = (min(_get_num_vector_core(), max(num_tasks, 1)),)
    bool_count_nonzero_kernel[grid](
        new_mask,
        out_i32,
        new_mask.stride(0),
        new_mask.stride(1),
        new_mask.stride(2),
        new_mask.stride(3),
        new_mask.stride(4),
        new_mask.stride(5),
        out_i32.stride(0),
        out_i32.stride(1),
        out_i32.stride(2),
        out_i32.stride(3),
        NUM_TASKS=num_tasks,
        H=H,
        Q_NUM_BLOCKS=Q_NUM_BLOCKS,
        KV_NUM_BLOCKS=KV_NUM_BLOCKS,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )

    return out_i32


def _round_up_to_multiple(x, multiple):
    return (x + multiple - 1) // multiple * multiple


# ============================================================================
# Kernel 2: create_mask_kernel
# 功能: 直接从1D lookup表构建dense_mask [1,1,Q,KV] bool
# 替代torch的vmap-based create_mask, 避免vmap带来的28GB瞬时显存峰值
#
# 设计: 每个program处理一个TILE×TILE的子区域, 通过查表+比较直接计算mask值
# MASK_TYPE: 编译期常量, 决定mask逻辑:
#   0=sparse(跨样本SWA+同图双向), 1=stair(同视频帧因果),
#   2=video_stair(同视频帧内全可见+帧间因果), 3=cross_causal_video_bidir(全局因果+同视频双向)
# TABLE1/2/3: 根据MASK_TYPE传入不同的1D索引表
# ============================================================================
@triton.jit
def create_mask_kernel(
    OUT,
    stride_ob, stride_oh, stride_oq, stride_ok,
    TABLE1, stride_t1,
    TABLE2, stride_t2,
    TABLE3, stride_t3,
    Q_LEN, KV_LEN,
    W, G,
    MASK_TYPE: tl.constexpr,
    TILE: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int32)
    pid_k = tl.program_id(1).to(tl.int32)
    q_off = pid_q * TILE + tl.arange(0, TILE)
    k_off = pid_k * TILE + tl.arange(0, TILE)
    q_idx = q_off[:, None]
    k_idx = k_off[None, :]

    if MASK_TYPE == 0:
        # sparse: same_doc & (swa_window | global_window) | same_image
        # TABLE1=segment_ids, TABLE2=doc_start, TABLE3=modality
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=0)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-1)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        window = causal & ((q_idx - k_idx) <= W)
        ds_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        glob = causal & (k_idx >= ds_q) & (k_idx < ds_q + G)
        sparse = same_doc & (window | glob)
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = sparse | same_img
    elif MASK_TYPE == 1:
        # stair: same_video & frame_causal
        # TABLE1=video_ids, TABLE2=frame_ids
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        frame_causal = fid_q >= fid_k
        result = same_doc & frame_causal
    elif MASK_TYPE == 2:
        # video_stair: same_video & (same_frame | prev_frame)
        # TABLE1=video_ids, TABLE2=frame_ids
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_video = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        same_frame = fid_q == fid_k
        prev_frame = fid_q > fid_k
        result = same_video & (same_frame | prev_frame)
    elif MASK_TYPE == 3:
        # cross_sample_causal_video_bidir: causal | same_video
        # TABLE1=modality
        causal = q_idx >= k_idx
        mod_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        is_video = mod_q > 0
        same_video = is_video & (mod_q == mod_k)
        result = causal | same_video
    elif MASK_TYPE == 4:
        # full: samedoc_causal | same_image
        # TABLE1=segment_ids, TABLE3=modality
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        samedoc_causal = same_doc & causal
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = samedoc_causal | same_img
    elif MASK_TYPE == 5:
        # causal_middle_full: global causal | (both q and kv in middle segment)
        # TABLE1 = middle_flag (1=middle segment, 0=other)
        causal = q_idx >= k_idx
        mid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=0)
        mid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=0)
        middle_full = (mid_q == 1) & (mid_k == 1)
        result = causal | middle_full
    else:
        result = tl.full([TILE, TILE], False, tl.int1)

    out_base = OUT
    q_mask = q_idx < Q_LEN
    k_mask = k_idx < KV_LEN
    valid = q_mask & k_mask
    ptrs = out_base + q_idx * stride_oq + k_idx * stride_ok
    tl.store(ptrs, result, mask=valid)


_MASK_TYPE_MAP = {
    "sparse": 0,
    "stair": 1,
    "video_stair": 2,
    "cross_sample_causal_video_bidir": 3,
    "full": 4,
    "causal_middle_full": 5,
}


def triton_create_mask(problem, mask_type: str, tile_size: int = 128):
    """
    用Triton kernel构建dense_mask, 替代torch的vmap-based create_mask.
    优势: 峰值显存仅dense_mask本身(~2.7GB for 52K×52K bool),
    而vmap版因4层嵌套展开产生~28GB瞬时峰值.
    """
    SEQ_LEN = problem["total_s"]
    device = problem["q"].device
    out = torch.empty(1, 1, SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=device)

    mt = _MASK_TYPE_MAP[mask_type]
    t1 = t2 = t3 = torch.empty(0, device=device)
    s1 = s2 = s3 = 0
    W_val = 0
    G_val = 0

    if mt == 0:
        # sparse: TABLE1=segment_ids, TABLE2=doc_start, TABLE3=modality
        t1, t2, t3 = problem["segment_ids"], problem["doc_start"], problem["modality"]
        s1, s2, s3 = t1.stride(0), t2.stride(0), t3.stride(0)
        W_val = problem["sliding_window"]
        G_val = problem["global_window"]
    elif mt == 1:
        # stair: TABLE1=video_ids, TABLE2=frame_ids
        t1, t2 = problem["video_ids"], problem["frame_ids"]
        s1, s2 = t1.stride(0), t2.stride(0)
        t3 = torch.empty(0, device=device)
    elif mt == 2:
        # video_stair: TABLE1=video_ids, TABLE2=frame_ids
        t1, t2 = problem["video_ids"], problem["frame_ids"]
        s1, s2 = t1.stride(0), t2.stride(0)
        t3 = torch.empty(0, device=device)
    elif mt == 3:
        # cross_sample_causal_video_bidir: TABLE1=modality
        t1 = problem["modality"]
        s1 = t1.stride(0)
    elif mt == 4:
        # full: TABLE1=segment_ids, TABLE3=modality
        t1, t3 = problem["segment_ids"], problem["modality"]
        s1, s3 = t1.stride(0), t3.stride(0)
    elif mt == 5:
        # causal_middle_full: TABLE1=middle_flag
        t1 = problem["middle_flag"]
        s1 = t1.stride(0)

    n_tiles_q = (SEQ_LEN + tile_size - 1) // tile_size
    n_tiles_k = (SEQ_LEN + tile_size - 1) // tile_size
    grid = (n_tiles_q, n_tiles_k)

    create_mask_kernel[grid](
        out,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        t1, s1,
        t2, s2,
        t3, s3,
        SEQ_LEN, SEQ_LEN,
        W_val, G_val,
        MASK_TYPE=mt,
        TILE=tile_size,
    )
    return out


# ============================================================================
# Kernel 3: block_classify_kernel
# 功能: 将dense_mask [B,H,Q,KV] bool 划分为 Q_BLOCK_SIZE×KV_BLOCK_SIZE 的block,
#        每个block分类为: 0=empty(全False), 1=partial(混合), 2=full(全True)
# 输入: dense_mask [B,H,Q,KV] bool
# 输出: block_flags [B,H,Q_NB,KV_NB] int8
# ============================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mq", "stride_mk",
        "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "NUM_TASKS",
    ]
)
def block_classify_kernel(
    DENSE_MASK,
    stride_mb,
    stride_mh,
    stride_mq,
    stride_mk,
    BLOCK_FLAGS,
    stride_fb,
    stride_fh,
    stride_fqb,
    stride_fkb,
    Q_LEN,
    KV_LEN,
    NUM_TASKS,
    H: tl.constexpr,
    Q_NUM_BLOCKS,
    KV_NUM_BLOCKS,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    num_blocks_per_bh = Q_NUM_BLOCKS * KV_NUM_BLOCKS
    TILE_M: tl.constexpr = 64
    TILE_N: tl.constexpr = 64

    for task_id in range(pid, NUM_TASKS, num_core):
        off_bh = task_id // num_blocks_per_bh
        off_inner = task_id % num_blocks_per_bh
        off_b = (off_bh // H).to(tl.int64)
        off_h = (off_bh % H).to(tl.int64)
        off_qb = (off_inner // KV_NUM_BLOCKS).to(tl.int64)
        off_kb = (off_inner % KV_NUM_BLOCKS).to(tl.int64)

        has_one = tl.full((), 0, dtype=tl.int32)
        all_one = tl.full((), 1, dtype=tl.int32)

        mask_base = DENSE_MASK + off_b * stride_mb + off_h * stride_mh

        for m0 in range(0, Q_BLOCK_SIZE, TILE_M):
            offs_m = off_qb * Q_BLOCK_SIZE + m0 + tl.arange(0, TILE_M)
            valid_m = offs_m < Q_LEN
            for n0 in range(0, KV_BLOCK_SIZE, TILE_N):
                offs_n = off_kb * KV_BLOCK_SIZE + n0 + tl.arange(0, TILE_N)
                valid_n = offs_n < KV_LEN
                valid = valid_m[:, None] & valid_n[None, :]

                ptrs = (
                    mask_base
                    + offs_m[:, None] * stride_mq
                    + offs_n[None, :] * stride_mk
                )
                vals = tl.load(ptrs, mask=valid, other=0).to(tl.int32)

                tile_any = tl.max(tl.max(tl.where(valid, vals, 0), axis=1), axis=0)
                tile_all = tl.min(tl.min(tl.where(valid, vals, 0), axis=1), axis=0)
                has_one = tl.where(tile_any != 0, 1, has_one)
                all_one = tl.where(tile_all == 0, 0, all_one)

        partial = (has_one == 1) & (all_one == 0)
        full = all_one == 1
        flag = tl.where(full, 2, tl.where(partial, 1, 0))

        out_ptr = (
            BLOCK_FLAGS
            + off_b * stride_fb
            + off_h * stride_fh
            + off_qb * stride_fqb
            + off_kb * stride_fkb
        )
        tl.store(out_ptr, flag.to(tl.int8))


def classify_mask_blocks(dense_mask, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Step2: 对dense_mask做block级分类, 输出block_flags [B,H,Q_NB,KV_NB]
    每个block: 0=empty, 1=partial, 2=full
    """
    if dense_mask.dtype != torch.bool:
        raise TypeError(f"dense_mask must be bool, got {dense_mask.dtype}")
    if dense_mask.ndim != 4:
        raise ValueError(f"dense_mask must be 4D [B,H,Q,KV], got shape {tuple(dense_mask.shape)}")

    dense_mask = dense_mask.contiguous()
    B, H, Q_PAD, KV_PAD = dense_mask.shape
    Q_NUM_BLOCKS = _round_up_to_multiple(Q_LEN, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_NUM_BLOCKS = _round_up_to_multiple(KV_LEN, KV_BLOCK_SIZE) // KV_BLOCK_SIZE

    block_flags = torch.zeros(
        (B, H, Q_NUM_BLOCKS, KV_NUM_BLOCKS),
        device=dense_mask.device,
        dtype=torch.int8,
    )

    num_tasks = B * H * Q_NUM_BLOCKS * KV_NUM_BLOCKS
    grid = (min(_get_num_vector_core(), max(num_tasks, 1)),)
    block_classify_kernel[grid](
        dense_mask,
        dense_mask.stride(0),
        dense_mask.stride(1),
        dense_mask.stride(2),
        dense_mask.stride(3),
        block_flags,
        block_flags.stride(0),
        block_flags.stride(1),
        block_flags.stride(2),
        block_flags.stride(3),
        Q_LEN,
        KV_LEN,
        NUM_TASKS=num_tasks,
        H=H,
        Q_NUM_BLOCKS=Q_NUM_BLOCKS,
        KV_NUM_BLOCKS=KV_NUM_BLOCKS,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )
    return block_flags


def _convert_mask_to_block_mask(
    mask,
    Q_BLOCK_SIZE=128,
    KV_BLOCK_SIZE=128,
    separate_full_blocks=False
):
    """
    旧路径: 通过reshape+count_nonzero做block分类, 替代Triton版block_classify_kernel.
    当separate_full_blocks=True时, 返回(partial_blocks, full_blocks) int8
    否则返回(partial_blocks, None), 其中partial=非空block
    """
    assert mask.dtype == torch.bool

    def padding_needed_for_multiple(x, multiple):
        return _round_up_to_multiple(x, multiple) - x

    mask_pad = torch.nn.functional.pad(
        mask,
        (
            0,
            padding_needed_for_multiple(mask.shape[-1], KV_BLOCK_SIZE),
            0,
            padding_needed_for_multiple(mask.shape[-2], Q_BLOCK_SIZE),
        ),
    )

    B, H, Q, KV = mask_pad.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0
    mask_pad = mask_pad.view(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )

    new_mask = mask_pad.permute(
        0, 1, 2, 4, 3, 5
    )

    mask_block_sum_triton = triton_count_nonzero_last(new_mask)

    del new_mask, mask_pad

    mask_block_sum = mask_block_sum_triton
    if separate_full_blocks:
        full_block_sum = Q_BLOCK_SIZE * KV_BLOCK_SIZE
        full_blocks = mask_block_sum == full_block_sum
        partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        full_blocks = full_blocks.to(dtype=torch.int8)
        return partial_blocks, full_blocks
    else:
        partial_blocks = mask_block_sum > 0
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        return partial_blocks, None


def _compute_partial_offsets(block_flags):
    """
    Step3: 从block_flags计算partial block的偏移表
    - A: 每个partial block在其所在行中的局部序号 (cumsum)
    - B: 每行的partial block数量 (max of A)
    - C: 每行的起始偏移量 (cumsum of B - B)

    返回:
    - partial_mask_offsets: [B,H,Q_NB] int32, 每行的起始偏移
    - local_idx: [B,H,Q_NB,KV_NB] int32, 每个partial block的行内序号
    - total_partial: int, 总partial block数
    """
    flags = (block_flags == 1).to(torch.int32)      # [B, H, Q_NB, KV_NB]
    A = flags.cumsum(dim=-1)                        # [B, H, Q_NB, KV_NB] 行内局部索引
    B = A.max(dim=-1).values                        # [B, H, Q_NB] 每行partial数
    C = B.cumsum(dim=-1) - B                        # [B, H, Q_NB] 行起始偏移
    total_partial = int(B.sum().item())
    partial_mask_offsets = C.contiguous().to(torch.int32) # [B, H, Q_NB]
    return partial_mask_offsets, A, total_partial


# ============================================================================
# Kernel 4: pack_partial_blocks_kernel
# 功能: 将dense_mask中的partial block拷贝到紧凑的packed_partial_mask中,
#        同时构建partial_block_table记录每个partial block在packed中的位置.
#
# 输入:
#   - DENSE_MASK: [B,H,Q,KV] bool — 原始全量mask
#   - BLOCK_FLAGS: [Q_NB,KV_NB] int8 — block分类(0/1/2)
#   - PARTIAL_OFFSETS: [Q_NB] int32 — 每行的起始偏移
#   - LOCAL_IDX: [Q_NB,KV_NB] int32 — 每个partial block的行内序号
# 输出:
#   - PACKED_MASK: [total_partial, Q_BS, KV_BS] bool — 紧凑存储的partial block数据
#   - BLOCK_TABLE: [Q_NB, KV_NB] int32 — 每个block在packed中的索引, -1表示非partial
# ============================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mq", "stride_mk",
        "stride_offset_q",
        "stride_local_q", "stride_local_k",
        "stride_flag_q", "stride_flag_k",
        "stride_table_q", "stride_table_k",
        "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "TOTAL_PARTIAL",
    ]
)
def pack_partial_blocks_kernel(
    DENSE_MASK,
    stride_mb,
    stride_mh,
    stride_mq,
    stride_mk,
    BLOCK_FLAGS,
    stride_flag_q,
    stride_flag_k,
    PARTIAL_OFFSETS,
    stride_offset_q,
    LOCAL_IDX,
    stride_local_q,
    stride_local_k,
    PACKED_MASK,
    stride_packed_p,
    stride_packed_m,
    stride_packed_n,
    BLOCK_TABLE,
    stride_table_q,
    stride_table_k,
    Q_LEN,
    KV_LEN,
    Q_NUM_BLOCKS,
    KV_NUM_BLOCKS,
    TOTAL_PARTIAL,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)

    if pid_q >= Q_NUM_BLOCKS:
        return

    row_offset = tl.load(PARTIAL_OFFSETS + pid_q * stride_offset_q).to(tl.int32)
    offs_m_local = tl.arange(0, Q_BLOCK_SIZE)[:, None].to(tl.int64)
    offs_n_local = tl.arange(0, KV_BLOCK_SIZE)[None, :].to(tl.int64)

    for kv_idx in range(KV_NUM_BLOCKS):
        flag = tl.load(BLOCK_FLAGS + pid_q * stride_flag_q + kv_idx * stride_flag_k).to(tl.int32)
        is_partial = flag == 1

        if is_partial:
            local_idx = tl.load(
                LOCAL_IDX + pid_q * stride_local_q + kv_idx * stride_local_k
            ).to(tl.int32)
            packed_idx = (row_offset + local_idx - 1).to(tl.int64)

            offs_m = (pid_q * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE))[:, None].to(tl.int64)
            offs_n = (kv_idx * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE))[None, :].to(tl.int64)
            valid_src = (offs_m < Q_LEN) & (offs_n < KV_LEN)

            src_ptrs = DENSE_MASK + offs_m * stride_mq + offs_n * stride_mk
            block = tl.load(src_ptrs, mask=valid_src, other=0)

            dst_ptrs = (
                PACKED_MASK
                + packed_idx * stride_packed_p
                + offs_m_local * stride_packed_m
                + offs_n_local * stride_packed_n
            )
            tl.store(dst_ptrs, block)

            tl.store(
                BLOCK_TABLE + pid_q * stride_table_q + kv_idx * stride_table_k,
                packed_idx.to(tl.int32),
            )


def pack_partial_blocks(dense_mask, block_flags, partial_mask_offsets, local_idx,
                        total_partial, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Step4: 打包partial block数据 + 构建block table
    返回:
    - packed_partial_mask: [total_partial, Q_BS, KV_BS] bool
    - partial_block_table: [Q_NB, KV_NB] int32, -1 for non-partial
    - partial_mask_offsets_3d: [B, H, Q_NB] int32
    """
    dense_mask = dense_mask.contiguous()
    B, H, Q_NB, KV_NB = block_flags.shape
    device = dense_mask.device

    if total_partial == 0:
        packed_partial_mask = torch.zeros(
            (0, Q_BLOCK_SIZE, KV_BLOCK_SIZE),
            dtype=torch.bool, device=device,
        )
        partial_block_table = torch.full(
            (Q_NB, KV_NB), -1, dtype=torch.int32, device=device,
        )
        partial_mask_offsets_3d = torch.zeros(
            (B, H, Q_NB), dtype=torch.int32, device=device,
        )
        return packed_partial_mask, partial_block_table, partial_mask_offsets_3d

    packed_partial_mask = torch.zeros(
        (total_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE),
        dtype=torch.bool, device=device,
    )
    partial_block_table = torch.full(
        (Q_NB, KV_NB), -1, dtype=torch.int32, device=device,
    )

    # B=H=1, 降维到2D传给kernel
    offsets_1d = partial_mask_offsets[0, 0].contiguous()
    local_idx_2d = local_idx[0, 0].contiguous()
    flags_2d = block_flags[0, 0].contiguous()

    pack_grid = (Q_NB,)
    pack_partial_blocks_kernel[pack_grid](
        dense_mask,
        dense_mask.stride(0),
        dense_mask.stride(1),
        dense_mask.stride(2),
        dense_mask.stride(3),
        flags_2d,
        flags_2d.stride(0),
        flags_2d.stride(1),
        offsets_1d,
        offsets_1d.stride(0),
        local_idx_2d,
        local_idx_2d.stride(0),
        local_idx_2d.stride(1),
        packed_partial_mask,
        packed_partial_mask.stride(0),
        packed_partial_mask.stride(1),
        packed_partial_mask.stride(2),
        partial_block_table,
        partial_block_table.stride(0),
        partial_block_table.stride(1),
        Q_LEN,
        KV_LEN,
        Q_NUM_BLOCKS=Q_NB,
        KV_NUM_BLOCKS=KV_NB,
        TOTAL_PARTIAL=total_partial,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )

    partial_mask_offsets_3d = partial_mask_offsets.contiguous()
    return packed_partial_mask, partial_block_table, partial_mask_offsets_3d


def create_block_mask(mask, slen, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128):
    """
    从dense_mask构建BlockMask对象(旧路径), 内部使用_convert_mask_to_block_mask做block分类.
    注意: 此路径内部会调用vmap, 有较大显存峰值.
    """
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
        mask,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=True,
    )

    block_mask = _create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask),
        2,
        (slen, slen),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )
    return block_mask


# ============================================================================
# Mask函数定义 — 每种mask类型返回一个闭包, 接受(b,h,q_idx,kv_idx)返回bool
# 这些闭包仅用于兼容torch的vmap-based create_mask, Triton路径不使用.
# ============================================================================
def _sparse_mask_mod(problem):
    """
    跨样本遵循swa, video内部遵循双向mask
    """
    segment_ids = problem["segment_ids"]
    modality = problem["modality"]
    doc_start = problem["doc_start"]
    W = problem["sliding_window"]
    G = problem["global_window"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = segment_ids[q_idx] == segment_ids[kv_idx]
        causal = q_idx >= kv_idx
        window = causal & ((q_idx - kv_idx) <= W)
        glob = causal & (kv_idx >= doc_start[q_idx]) & (kv_idx < doc_start[q_idx] + G)
        sparse = same_doc & (window | glob)
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return sparse | same_img
    return mask_mod


def _stair_mask_mod(problem):
    """
    阶梯式block mask
    """
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = video_ids[q_idx] == video_ids[kv_idx]
        frame_causal = frame_ids[q_idx] >= frame_ids[kv_idx]
        return same_doc & frame_causal
    return mask_mod


def _video_stair_mask_mod(problem):
    """
    video内部阶梯式, 后面的帧可以看前面的帧, 不同的video之间不可见, 同一帧内部full mask
    """
    video_ids = problem["video_ids"]
    frame_ids = problem["frame_ids"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_video = video_ids[q_idx] == video_ids[kv_idx]
        same_frame = frame_ids[q_idx] == frame_ids[kv_idx]
        prev_frame = frame_ids[q_idx] > frame_ids[kv_idx]
        return same_video & (same_frame | prev_frame)
    return mask_mod


def _cross_sample_causal_video_bidir_mask_mod(problem):
    """
    跨样本遵循causal, video内部遵循双向mask
    """
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        is_video = modality[q_idx] > 0
        same_video = is_video & (modality[q_idx] == modality[kv_idx])
        return causal | same_video
    return mask_mod


def _full_mask_mod(problem):
    document_ids = problem["segment_ids"]
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        samedoc_causal = same_doc & causal
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return samedoc_causal | same_img

    return mask_mod

def _causal_middle_full_mask_mod(problem):
    """
    Overall causal + middle segment fully bidirectional.
    middle_flag: 1=middle segment, 0=other.
    mask = causal | (q in middle & k in middle)
    """
    middle_flag = problem["middle_flag"]

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        mid_q = middle_flag[q_idx] == 1
        mid_k = middle_flag[kv_idx] == 1
        middle_full = mid_q & mid_k
        return causal | middle_full
    return mask_mod


# mask_func → mask_type string 的映射, 用于triton_create_mask
_MASK_FUNC_TO_TYPE = {
    id(_sparse_mask_mod): "sparse",
    id(_stair_mask_mod): "stair",
    id(_video_stair_mask_mod): "video_stair",
    id(_cross_sample_causal_video_bidir_mask_mod): "cross_sample_causal_video_bidir",
    id(_full_mask_mod): "full",
    id(_causal_middle_full_mask_mod): "causal_middle_full",
}


# ============================================================================
# Attention函数封装
# ============================================================================
def _flex_attention_triton(q, k, v, mask, block_mask, dropout_rate=0.0, input_format=None):
    """
    Triton版flex_attention. 当mask!=None时, 将dense_mask绑定到block_mask上(旧路径);
    当mask=None时, 使用block_mask上的packed属性(packed_partial_mask等)进行计算.
    """
    if input_format == "head-first":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    if mask is not None:
        block_mask.dense_mask = mask
    output = flex_attention(q, k, v, block_mask=block_mask, return_lse=False)
    return output.transpose(1, 2)


def _flex_attention_native(q, k, v, mask_mod, seq, input_format=None, device="cpu",
                           block_size=128, enable_gqa=False):
    if input_format == "head-first":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    # device="cpu" 时把 q/k/v 搬到 CPU 计算 (避免 NPU OOM), 结果再搬回原设备
    orig_device = q.device
    if device == "cpu" and orig_device.type != "cpu":
        q = q.cpu()
        k = k.cpu()
        v = v.cpu()

    block_mask = create_block_mask_native(mask_mod, 1, 1, seq, seq,
                                          BLOCK_SIZE=block_size, device=device)
    output = flex_attention_native(q, k, v, block_mask=block_mask,
                                   return_lse=False, enable_gqa=enable_gqa)
    output = output.transpose(1, 2).to(orig_device)
    return output


# ============================================================================
# 数据构建: 生成1D lookup表(segment_ids, modality, video_ids, frame_ids等)
# ============================================================================
def _build_video_indicators(device):
    """构建video类mask所需的1D索引表: video_ids, frame_ids, segment_ids, doc_start"""
    segment_ids = []
    doc_start = []
    video_ids = []
    frame_ids = []
    modality = []
    sample_lens = []
    sample_start = 0
    next_video_id = 0

    for sample_id, sample_videos in enumerate(VIDEO_FRAME_LENGTH):
        cur_sample_len = sum(sum(frame_lens) for frame_lens in sample_videos)
        sample_lens.append(cur_sample_len)

        for frame_lens in sample_videos:
            cur_video_id = next_video_id
            next_video_id += 1

            for frame_id, frame_len in enumerate(frame_lens):
                segment_ids.append(torch.full((frame_len,), sample_id, dtype=torch.long))
                doc_start.append(torch.full((frame_len,), sample_start, dtype=torch.long))
                video_ids.append(torch.full((frame_len,), cur_video_id, dtype=torch.long))
                frame_ids.append(torch.full((frame_len,), frame_id, dtype=torch.long))

                modality.append(torch.full((frame_len,), cur_video_id + 1, dtype=torch.long))
        sample_start += cur_sample_len

    return {
        "sample_lens": sample_lens,
        "segment_ids": torch.cat(segment_ids).to(device),
        "doc_start": torch.cat(doc_start).to(device),
        "video_ids": torch.cat(video_ids).to(device),
        "frame_ids": torch.cat(frame_ids).to(device),
        "modality": torch.cat(modality).to(device),
    }


def _build_modality_indicators(device):
    """构建sparse/cross_causal类mask所需的1D modality表: image_gen类>0, 其他=-1"""
    indicator = []
    iidx = 1

    for sample_types, sample_lens in zip(DATA_INPUT_TYPE, DATA_LENGTH):
        for i, (sample_type, sample_len) in enumerate(zip(sample_types, sample_lens)):
            if sample_type in FULL_MASK_MODALITIES:
                indicator.append(torch.full((sample_len,), iidx, dtype=torch.long))
                iidx += 1
            else:
                indicator.append(torch.full((sample_len,), -1, dtype=torch.long))
    return torch.cat(indicator).to(device)


def _build_middle_flag(data_length, device):
    """
    Build 1D middle_flag: mark the middle segment of each sample as 1, others as 0.
    e.g. data_length=[[72, 4056, 987]] ->
      seg0[0:72)=0, seg1[72:4128)=1, seg2[4128:5115)=0
    """
    flag_parts = []
    for sample_lens in data_length:
        n_seg = len(sample_lens)
        mid_idx = n_seg // 2
        for i, seg_len in enumerate(sample_lens):
            flag_parts.append(torch.full((seg_len,), 1 if i == mid_idx else 0, dtype=torch.long))
    return torch.cat(flag_parts).to(device)


def build_problem(mask_mod):
    device = torch.device("npu")
    torch.manual_seed(SEED)
    local_data_len = DATA_LENGTH
    if mask_mod in [_video_stair_mask_mod, _stair_mask_mod]:
        local_data_len = DATA_LENGTH_VIDEO

    if mask_mod in [_causal_middle_full_mask_mod]:
        local_data_len = DATA_LENGTH_1
        num_q_heads = 4
        num_kv_heads = 1
    else:
        num_q_heads = NUM_Q_HEADS
        num_kv_heads = NUM_KV_HEADS

    sample_lens = [sum(s) for s in local_data_len]
    cu_seqlens = torch.tensor([0, *torch.tensor(sample_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_s = int(cu_seqlens[-1].item())
    segment_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), device=device, dtype=torch.int32),
        torch.tensor(sample_lens, device=device),
    )
    doc_start = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff()).to(torch.long)

    q = torch.rand(1, num_q_heads, total_s, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.rand(1, num_kv_heads, total_s, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.rand(1, num_kv_heads, total_s, HEAD_DIM, device=device, dtype=DTYPE)

    if mask_mod in [_video_stair_mask_mod, _stair_mask_mod]:
        meta = _build_video_indicators(device=device)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": meta["segment_ids"], "doc_start": meta["doc_start"],
            "video_ids": meta["video_ids"], "frame_ids": meta["frame_ids"], "modality": meta["modality"],
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": SLIDING_WINDOW, "global_window": GLOBAL_WINDOW,
            "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads, "head_dim": HEAD_DIM,
        }
    else:
        modality = _build_modality_indicators(device=device)
        middle_flag = _build_middle_flag(local_data_len, device=device)
        return {
            "q": q, "k": k, "v": v,
            "segment_ids": segment_ids.long(), "modality": modality, "middle_flag": middle_flag, "doc_start": doc_start,
            "cu_seqlens": cu_seqlens, "total_s": total_s,
            "sliding_window": SLIDING_WINDOW, "global_window": GLOBAL_WINDOW,
            "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads, "head_dim": HEAD_DIM,
        }


def _sdpa_with_dense_mask(
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout_rate,
    input_format,
):
    """
    AscendC参考实现: 使用torch的scaled_dot_product_attention + dense_mask.
    按Q分块处理, 每块只取KV中mask非空的部分, 减少无效计算.
    """
    if input_format == "head-first":
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    mask_2d = attention_mask
    while mask_2d.dim() > 2:
        mask_2d = mask_2d[0]

    q_len_total = query_states.size(2)
    block_q = APPLY_Q_CHUNK if APPLY_Q_CHUNK is not None else q_len_total
    chunks = []
    for qs in range(0, q_len_total, block_q):
        qe = min(qs + block_q, q_len_total)
        row = mask_2d[qs:qe]
        col_any = row.any(dim=0)
        nz = col_any.nonzero(as_tuple=False)
        if nz.numel() == 0:
            chunks.append(
                query_states.new_zeros(
                    (query_states.size(0), query_states.size(1), qe-qs, query_states.size(3))
                )
            )
            continue
        kmin = int(nz[0].item())
        kmax = int(nz[-1].item()) + 1
        chunks.append(
            torch.nn.functional.scaled_dot_product_attention(
                query_states[:, :, qs:qe],
                key_states[:, :, kmin:kmax],
                value_states[:, :, kmin:kmax],
                attn_mask=row[None, None, :, kmin:kmax],
                dropout_p=dropout_rate,
                enable_gqa=False,
            )
        )
    attn_output = torch.cat(chunks, dim=2)
    return attn_output.transpose(1, 2).contiguous()


# ============================================================================
# 测试主流程: accuracy test + perf benchmark
# ============================================================================
@pytest.mark.parametrize(
    "mask_func",
    [
        _sparse_mask_mod,
        _stair_mask_mod,
        _video_stair_mask_mod,
        _cross_sample_causal_video_bidir_mask_mod,
    ],
    ids=["sparse", "stair", "video_stair", "cross_sample_causal_video_bidir"],
)
def test_flex_attention(mask_func):
    """
    测试流程:
    1. 构建problem数据 (q/k/v + 1D索引表)
    2. 用Triton kernel构建dense_mask [1,1,SEQ,SEQ] bool
    3. Packed mask构建流水线:
       Step2: block_classify — dense_mask → block_flags [1,1,Q_NB,KV_NB] (0=empty/1=partial/2=full)
       Step3: compute_offsets — block_flags → partial_mask_offsets + local_idx + total_partial
       Step4: pack_partial — dense_mask + flags + offsets → packed_partial_mask + partial_block_table
    4. 构建block_mask对象, 挂载packed属性
    5. Accuracy验证: triton_packed vs ascendc (fwd + bwd)
    6. Perf benchmark: 三条路径(triton_dense / triton_packed / ascendc)的显存+耗时对比
    """
    problem = build_problem(mask_func)

    q_base = problem["q"]
    k_base = problem["k"]
    v_base = problem["v"]

    q_triton = q_base.detach().clone().requires_grad_(True)
    k_triton = k_base.detach().clone().requires_grad_(True)
    v_triton = v_base.detach().clone().requires_grad_(True)

    q_ascendc = q_base.detach().clone().requires_grad_(True)
    k_ascendc = k_base.detach().clone().requires_grad_(True)
    v_ascendc = v_base.detach().clone().requires_grad_(True)

    if ENABLE_NATIVE:
        q_native = q_base.detach().clone().requires_grad_(True)
        k_native = k_base.detach().clone().requires_grad_(True)
        v_native = v_base.detach().clone().requires_grad_(True)

    SEQ_LEN = problem["total_s"]

    # Step 1: 用Triton kernel构建dense_mask (替代vmap版create_mask, 避免28GB峰值)
    mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]
    dense_mask = triton_create_mask(problem, mask_type_str)
    torch.npu.synchronize()

    print(">>>>>>>>>>>dense_mask.sum().item(): ", dense_mask.to("cpu").sum().item())

    # Step 2: block级分类 — dense_mask → block_flags
    block_flags = classify_mask_blocks(dense_mask, SEQ_LEN, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Step 3: 偏移计算 — block_flags → offsets + local_idx
    partial_mask_offsets, local_idx, total_partial = _compute_partial_offsets(block_flags)
    
    # Step 4: 打包partial block — dense_mask + flags → packed_partial_mask + table
    packed_partial_mask, partial_block_table, partial_mask_offsets_3d = pack_partial_blocks(
        dense_mask, block_flags, partial_mask_offsets, local_idx,
        total_partial, SEQ_LEN, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )

    # 构建block_mask对象, 挂载packed属性供attention kernel使用
    partial_block_mask = (block_flags == 1).to(dtype=torch.int8)
    full_block_mask = (block_flags == 2).to(dtype=torch.int8)
    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask),
        2,
        (SEQ_LEN, SEQ_LEN),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets_3d
    packed_block_mask.partial_block_table = partial_block_table

    # 释放中间tensor, dense_mask保留供ascendc参考路径使用
    del block_flags, partial_mask_offsets, local_idx, partial_block_mask, full_block_mask

    # ===== Accuracy验证: triton_packed vs ascendc =====
    # mask=None → 使用packed属性路径; 不绑定dense_mask到block_mask
    triton_output = _flex_attention_triton(
        q_triton,
        k_triton,
        v_triton,
        None,
        packed_block_mask,
        0.0,
        None,
    )
    torch.npu.synchronize()

    # ascendc参考路径: 使用dense_mask + scaled_dot_product_attention
    ascendc_output = _sdpa_with_dense_mask(
        q_ascendc,
        k_ascendc,
        v_ascendc,
        dense_mask,
        0.0,
        None,
    )
    torch.npu.synchronize()

    # torch native基线: 使用封装好的 _flex_attention_native (内部创建block_mask + enable_gqa)
    # input_format=None → 与 _flex_attention_triton / _sdpa_with_dense_mask 一致, 返回 [B, S, H, D]
    # 注意: NPU 上 flex_attention_native 回退到 math attention, 长序列 Q@K^T 物化 [B,H,S,S] → 易 OOM
    native_output = None
    if ENABLE_NATIVE:
        native_problem = problem
        if NATIVE_DEVICE == "cpu":
            native_problem = {
                key: (val.cpu() if isinstance(val, torch.Tensor) else val)
                for key, val in problem.items()
            }
        native_output = _flex_attention_native(
            q_native, k_native, v_native,
            mask_mod=mask_func(native_problem),
            seq=SEQ_LEN,
            input_format=None,
            device=NATIVE_DEVICE,
            block_size=Q_BLOCK_SIZE,
            enable_gqa=True,
        )
        torch.npu.synchronize()

    # fwd精度验证: triton vs ascendc (vs native)
    assert triton_output.shape == ascendc_output.shape
    torch.testing.assert_close(triton_output.cpu(), ascendc_output.cpu(), atol=5e-3, rtol=5e-3)
    if ENABLE_NATIVE:
        print(">>>>>>>>> NATIVE ACC verify >>>>>>>>")
        assert triton_output.shape == native_output.shape
        torch.testing.assert_close(triton_output.cpu(), native_output.cpu(), atol=8e-3, rtol=8e-3)
    torch.npu.synchronize()

    # bwd精度验证
    triton_output.float().mean().backward(RETURN_GRID)
    torch.npu.synchronize()

    ascendc_output.float().mean().backward(RETURN_GRID)
    torch.npu.synchronize()

    if ENABLE_NATIVE:
        native_output.float().mean().backward(RETURN_GRID)

    torch.testing.assert_close(q_triton.grad.cpu(), q_ascendc.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(k_triton.grad.cpu(), k_ascendc.grad.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(v_triton.grad.cpu(), v_ascendc.grad.cpu(), atol=5e-3, rtol=5e-3)

    if ENABLE_NATIVE:
        torch.testing.assert_close(q_triton.grad.cpu(), q_native.grad.cpu(), atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(k_triton.grad.cpu(), k_native.grad.cpu(), atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(v_triton.grad.cpu(), v_native.grad.cpu(), atol=5e-3, rtol=5e-3)


    if PROFILING:
        print(f"\n======================== prof begin ====================")
        import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )

        with torch_npu.profiler.profile(
                activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
                record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
                profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
                schedule=torch_npu.profiler.schedule(wait=1,
                                                    warmup=1,
                                                    active=30,
                                                    repeat=1,
                                                    skip_first=1),
                # warmup默认为0，老版本torch_npu包该参数为必填项
                experimental_config=experimental_config,  # 该参数选填，默认为Level0
                # 产生的profling文件的位置
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_dir")
        ) as prof:
            # prof.start()

            for i in range(10):
                # 重新构造数据，避免L2 Cache影响
                problem = build_problem(mask_func)
                q_triton = problem["q"].detach().clone().requires_grad_(True)
                k_triton = problem["k"].detach().clone().requires_grad_(True)
                v_triton = problem["v"].detach().clone().requires_grad_(True)

                triton_output = _flex_attention_triton(
                                                        q_triton,
                                                        k_triton,
                                                        v_triton,
                                                        None,
                                                        packed_block_mask,
                                                        0.0,
                                                        None,
                                                    )
                torch.npu.synchronize()  # 确保 kernel 真正执行完

                # 插入其他算子，重置L2 Cache, 112M
                # 三行生成张量，总访存224MB覆盖112MB L2
                for j in range(5):
                    a = torch.randn(19573419, dtype=torch.float32).npu()
                    b = torch.randn(19573419, dtype=torch.float32).npu()
                    c = torch.empty_like(a)
                    c = a + b                                               # 一行加法冲刷全部L2  


                triton_output.float().mean().backward(RETURN_GRID)
                torch.npu.synchronize()
                prof.step()
                del a,b,c

        # def fn():
        #     # 重新构造数据，避免L2 Cache影响
        #     problem = build_problem(mask_func)
        #     q_triton = problem["q"].detach().clone().requires_grad_(True)
        #     k_triton = problem["k"].detach().clone().requires_grad_(True)
        #     v_triton = problem["v"].detach().clone().requires_grad_(True)

        #     triton_output = _flex_attention_triton(
        #                                             q_triton,
        #                                             k_triton,
        #                                             v_triton,
        #                                             None,
        #                                             packed_block_mask,
        #                                             0.0,
        #                                             None,
        #                                         )
        #     torch.npu.synchronize()  # 确保 kernel 真正执行完

        #     # 插入其他算子，重置L2 Cache, 112M
        #     # 三行生成张量，总访存224MB覆盖112MB L2
        #     for j in range(5):
        #         a = torch.randn(19573419, dtype=torch.float32).npu()
        #         b = torch.randn(19573419, dtype=torch.float32).npu()
        #         c = torch.empty_like(a)
        #         c = a + b                                               # 一行加法冲刷全部L2  


        #     triton_output.float().mean().backward(RETURN_GRID)
        #     torch.npu.synchronize()


        # from triton.backends.ascend.testing import do_bench_npu
        # do_bench_npu(
        #     fn,
        #     active=10,
        #     clear_l2_cache=True,
        #     prof_dir="./prof_dir",
        #     keep_res=True,
        # )
        print(f"======================== prof end ====================")

    # ===== Perf benchmark: 释放accuracy test的所有tensor后运行 =====
    del triton_output, ascendc_output
    del q_triton, k_triton, v_triton, q_ascendc, k_ascendc, v_ascendc
    if ENABLE_NATIVE:
        del q_native, k_native, v_native, native_output
    del q_base, k_base, v_base
    del packed_block_mask, packed_partial_mask, partial_block_table, partial_mask_offsets_3d
    del dense_mask, problem
    import gc; gc.collect()
    torch.npu.empty_cache()
    perf_flex_attention(mask_func)


def _perf_benchmark(label, build_mask_fn, fwd_fn, q, k, v, iters=_ITERS):
    """
    单条路径的性能测试: 分别测量mask构建峰值/稳定占用、fwd耗时/峰值、bwd耗时/峰值.
    
    测量流程:
    1. warmup: fwd+bwd一次, 释放所有tensor
    2. mask构建: 执行build_mask_fn, 记录峰值(max_memory_allocated)和稳定占用(memory_allocated)
    3. fwd: 记录耗时和峰值显存
    4. bwd: 记录耗时和峰值显存
    """
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    # # warmup: fwd+bwd一次, 确保kernel编译完成, 然后释放
    # mask = build_mask_fn()
    # out = fwd_fn(q, k, v, mask)
    # torch.npu.synchronize()
    # out.float().mean().backward(RETURN_GRID)
    # torch.npu.synchronize()
    # q.grad = k.grad = v.grad = None
    # del out, mask
    # torch.npu.empty_cache()
    # import gc; gc.collect()
    # torch.npu.synchronize()

    # mask构建测量: 峰值(构建过程中的瞬时最大值) + 稳定占用(构建完清理后的驻留显存)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    mask = build_mask_fn()
    mask_peak = torch.npu.max_memory_allocated() / _MB
    torch.npu.empty_cache()
    import gc; gc.collect()
    torch.npu.synchronize()
    mask_mem = torch.npu.memory_allocated() / _MB

    # fwd测量 (含grad图构建)
    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    out = fwd_fn(q, k, v, mask)
    e.record()
    torch.npu.synchronize()
    fwd_ms = s.elapsed_time(e)
    fwd_mem = torch.npu.max_memory_allocated() / _MB

    # bwd测量
    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    out.float().mean().backward(RETURN_GRID)
    e.record()
    torch.npu.synchronize()
    bwd_ms = s.elapsed_time(e)
    bwd_mem = torch.npu.max_memory_allocated() / _MB

    q.grad = k.grad = v.grad = None
    peak_mem = max(fwd_mem, bwd_mem)
    print(f"[{label}] mask: {mask_mem:.1f}MB(peak:{mask_peak:.1f}MB), fwd: {fwd_ms:.2f}ms({fwd_mem:.1f}MB), bwd: {bwd_ms:.2f}ms({bwd_mem:.1f}MB), peak: {peak_mem:.1f}MB")
    del out, mask
    torch.npu.empty_cache()
    return {"mask_mem_mb": mask_mem, "mask_peak_mb": mask_peak,
            "fwd_ms": fwd_ms, "fwd_mem_mb": fwd_mem,
            "bwd_ms": bwd_ms, "bwd_mem_mb": bwd_mem,
            "peak_mem_mb": peak_mem}


def _build_packed_block_mask(dense_mask, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    从dense_mask构建packed block_mask的完整流水线:
    classify → compute_offsets → pack → 构建BlockMask对象 + 挂载packed属性
    """
    block_flags = classify_mask_blocks(dense_mask, SEQ_LEN, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    partial_mask_offsets, local_idx, total_partial = _compute_partial_offsets(block_flags)
    packed_partial_mask, partial_block_table, partial_mask_offsets_3d = pack_partial_blocks(
        dense_mask, block_flags, partial_mask_offsets, local_idx,
        total_partial, SEQ_LEN, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    partial_bm = (block_flags == 1).to(dtype=torch.int8)
    full_bm = (block_flags == 2).to(dtype=torch.int8)
    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_bm, full_bm), 2, (SEQ_LEN, SEQ_LEN), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets_3d
    packed_block_mask.partial_block_table = partial_block_table
    del block_flags, partial_mask_offsets, local_idx, partial_bm, full_bm
    return packed_block_mask


def perf_flex_attention(mask_func, problem=None):
    """
    三条路径的性能对比:
    - triton_dense: Triton flex_attention + dense_mask (旧路径, dense_mask绑定到block_mask)
    - triton_packed: Triton flex_attention + packed_partial_mask (新路径, 显存优化)
    - ascendc: torch SDPA + dense_mask (参考基线)

    所有路径均使用triton_create_mask构建dense_mask, 避免vmap的28GB峰值.
    """
    if problem is None:
        problem = build_problem(mask_func)
    SEQ_LEN = problem["total_s"]
    mask_type_str = _MASK_FUNC_TO_TYPE[id(mask_func)]

    results = {}

    # Dense路径: dense_mask + block_mask (旧路径, backward中save dense_mask)
    # def _build_dense_mask():
    #     dm = triton_create_mask(problem, mask_type_str)
    #     bm = create_block_mask(dm, SEQ_LEN, Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE)
    #     return bm, dm

    # results["triton_dense"] = _perf_benchmark(
    #     "triton_dense",
    #     _build_dense_mask,
    #     lambda q, k, v, m: _flex_attention_triton(q, k, v, m[1], m[0], 0.0, None),
    #     problem["q"], problem["k"], problem["v"],
    # )

    # # Packed路径: dense_mask → classify + pack → del dense_mask (新路径, backward中不save dense_mask)
    import gc; gc.collect()
    torch.npu.empty_cache()

    def _build_packed_mask():
        dm = triton_create_mask(problem, mask_type_str)
        pbm = _build_packed_block_mask(dm, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        del dm
        torch.npu.empty_cache()
        return pbm

    results["triton_packed"] = _perf_benchmark(
        "triton_packed",
        _build_packed_mask,
        lambda q, k, v, bm: _flex_attention_triton(q, k, v, None, bm, 0.0, None),
        problem["q"], problem["k"], problem["v"],
    )

    # AscendC路径: torch SDPA + dense_mask
    gc.collect()
    torch.npu.empty_cache()

    results["ascendc"] = _perf_benchmark(
        "ascendc",
        lambda: triton_create_mask(problem, mask_type_str),
        lambda q, k, v, m: _sdpa_with_dense_mask(q, k, v, m, 0.0, None),
        problem["q"], problem["k"], problem["v"],
    )
    return results


if __name__ == '__main__':
    import sys
    mask_map = {
        "sparse": _sparse_mask_mod,
        "stair": _stair_mask_mod,
        "video_stair": _video_stair_mask_mod,
        "cross_sample_causal_video_bidir": _cross_sample_causal_video_bidir_mask_mod,
        "full": _full_mask_mod,
        "causal_middle_full": _causal_middle_full_mask_mod,     # 智创-小shape
    }
    # test_flex_attention(mask_map["full"])
    name = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    if name == "all":
        for n, fn in mask_map.items():
            print(f"\n{'='*60}")
            print(f"Testing: {n}")
            print(f"{'='*60}")
            test_flex_attention(fn)
    else:
        test_flex_attention(mask_map[name])
