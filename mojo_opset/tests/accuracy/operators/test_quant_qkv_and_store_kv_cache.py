"""External accuracy test for MojoQuantQKVAndStoreKVCache (xops vs torch reference).

Covers M13-attn 输入路径 row 6/9 的算子（GQA / SWA 共用）：
对 Q 做 per-token-per-head int8 动态量化、对 K 同样动态量化并写入 paged key cache、
对 V 用静态 per-channel scale 量化并写入 paged value cache。
"""
import pytest
import torch

from mojo_opset.experimental import MojoQuantQKVAndStoreKVCache
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _build_inputs(tokens, q_heads, kv_heads, head_dim, num_blocks, block_size, device, dtype):
    """构造 (q, k, v, value_scale, key_cache, value_cache, slot_mapping)。"""
    query = torch.randn(tokens, q_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=dtype)
    value_scale = torch.rand(kv_heads, head_dim, device=device, dtype=torch.float32) + 0.1
    key_cache = torch.zeros(num_blocks, kv_heads, block_size, head_dim, dtype=torch.int8, device=device)
    value_cache = torch.zeros_like(key_cache)
    # 顺序写入前 tokens 个 slot
    slot_mapping = torch.arange(tokens, dtype=torch.int32, device=device)
    return query, key, value, value_scale, key_cache, value_cache, slot_mapping


@pytest.mark.parametrize(
    "tokens, q_heads, kv_heads, head_dim, num_blocks, block_size",
    [
        (64, 28, 8, 128, 8, 128),    # M13 GQA prefill chunk
        (128, 32, 8, 128, 8, 128),
        (32, 16, 8, 128, 4, 128),    # M13 SWA chunk
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@bypass_not_implemented
def test_quant_qkv_and_store_kv_cache(tokens, q_heads, kv_heads, head_dim, num_blocks, block_size, dtype):
    device = get_torch_device()

    op_ref = MojoQuantQKVAndStoreKVCache._registry.get("torch")()
    op = MojoQuantQKVAndStoreKVCache()
    if type(op_ref) is type(op):
        raise NotImplementedError("both operands resolve to the same impl, skipping comparison.")
    op_ref = op_ref.to(device)
    op = op.to(device)

    torch.manual_seed(42)
    q, k, v, vs, kc_ref, vc_ref, slot = _build_inputs(
        tokens, q_heads, kv_heads, head_dim, num_blocks, block_size, device, dtype,
    )
    kc = kc_ref.clone()
    vc = vc_ref.clone()

    qq_ref, qs_ref, kc_ref_out, _, vc_ref_out, _ = op_ref(
        q, k, v, vs, kc_ref, vc_ref, slot_mapping=slot,
    )
    qq, qs, kc_out, _, vc_out, _ = op(
        q, k, v, vs, kc, vc, slot_mapping=slot,
    )

    # int8 量化输出允许 ±1 LSB 偏差（rounding 实现差异），按 max diff 检查
    assert qq.shape == qq_ref.shape and qq.dtype == torch.int8
    assert qs.shape == qs_ref.shape

    diff_q = (qq.to(torch.int32) - qq_ref.to(torch.int32)).abs().max().item()
    assert diff_q <= 1, f"query int8 max diff {diff_q} > 1"
    diff_kc = (kc_out.to(torch.int32) - kc_ref_out.to(torch.int32)).abs().max().item()
    assert diff_kc <= 1, f"key_cache int8 max diff {diff_kc} > 1"
    diff_vc = (vc_out.to(torch.int32) - vc_ref_out.to(torch.int32)).abs().max().item()
    assert diff_vc <= 1, f"value_cache int8 max diff {diff_vc} > 1"

    # query scale: 浮点精度
    assert_close(qs, qs_ref)
