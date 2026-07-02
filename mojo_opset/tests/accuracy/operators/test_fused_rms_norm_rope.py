"""External accuracy test for MojoFusedRMSNormRope (xops vs torch reference).

Covers M13-attn 输入路径 row 4/7 的算子（GQA / SWA 共用）：
对 query/key 各自做 per-head_dim RMSNorm 后，再对最后一维 ``[rotary_offset, ...)``
范围做 RoPE。inplace 改写 query/key。
"""
import pytest
import torch

from mojo_opset.experimental import MojoFusedRMSNormRope
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _build_rope_table(max_position: int, rotary_dim: int, base: float = 10000.0,
                      device="cpu", dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim // 2, 2, dtype=torch.float32, device=device) / rotary_dim))
    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = freqs.repeat_interleave(2, dim=-1)  # 与 _rotate_half 配套
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return sin, cos


@pytest.mark.parametrize(
    "tokens, q_heads, kv_heads, head_dim, rotary_dim",
    [
        (128, 28, 8, 128, 48),     # M13 GQA
        (256, 32, 8, 128, 48),     # M13 dense layer
        (256, 16, 8, 128, 48),     # M13 SWA layer
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@bypass_not_implemented
def test_fused_rms_norm_rope_packed(tokens, q_heads, kv_heads, head_dim, rotary_dim, dtype):
    """Packed 3-D 输入：(tokens, heads, head_dim)，单 batch。"""
    device = get_torch_device()
    rotary_offset = head_dim - rotary_dim
    max_position = 4096

    op_ref = MojoFusedRMSNormRope._registry.get("torch")(
        head_dim=head_dim, rotary_offset=rotary_offset,
    ).to(device)
    op = MojoFusedRMSNormRope(head_dim=head_dim, rotary_offset=rotary_offset).to(device)
    if type(op_ref) is type(op):
        raise NotImplementedError("both operands resolve to the same impl, skipping comparison.")

    # 同步 weights
    torch.manual_seed(42)
    with torch.no_grad():
        op_ref.q_rms_weight.copy_(torch.randn_like(op_ref.q_rms_weight))
        op_ref.k_rms_weight.copy_(torch.randn_like(op_ref.k_rms_weight))
    op.load_state_dict(op_ref.state_dict(), strict=False)

    sin, cos = _build_rope_table(max_position, rotary_dim, device=device, dtype=dtype)

    q_in = torch.randn(tokens, q_heads, head_dim, device=device, dtype=dtype)
    k_in = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=dtype)

    cumsum_query_len = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    q_ref, k_ref = op_ref(q_in.clone(), k_in.clone(), sin, cos,
                          cumsum_query_len=cumsum_query_len, max_query_len=tokens)
    q_out, k_out = op(q_in.clone(), k_in.clone(), sin, cos,
                      cumsum_query_len=cumsum_query_len, max_query_len=tokens)
    assert_close(q_out, q_ref)
    assert_close(k_out, k_ref)


@pytest.mark.parametrize(
    "batch, seq_len, q_heads, kv_heads, head_dim, rotary_dim",
    [
        (2, 128, 28, 8, 128, 48),
        (1, 256, 32, 8, 128, 48),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_fused_rms_norm_rope_padded(batch, seq_len, q_heads, kv_heads, head_dim, rotary_dim, dtype):
    """非 packed 4-D 输入：(B, S, heads, head_dim)。"""
    device = get_torch_device()
    rotary_offset = head_dim - rotary_dim
    max_position = 4096

    op_ref = MojoFusedRMSNormRope._registry.get("torch")(
        head_dim=head_dim, rotary_offset=rotary_offset,
    ).to(device)
    op = MojoFusedRMSNormRope(head_dim=head_dim, rotary_offset=rotary_offset).to(device)
    if type(op_ref) is type(op):
        raise NotImplementedError("both operands resolve to the same impl, skipping comparison.")

    torch.manual_seed(42)
    with torch.no_grad():
        op_ref.q_rms_weight.copy_(torch.randn_like(op_ref.q_rms_weight))
        op_ref.k_rms_weight.copy_(torch.randn_like(op_ref.k_rms_weight))
    op.load_state_dict(op_ref.state_dict(), strict=False)

    sin, cos = _build_rope_table(max_position, rotary_dim, device=device, dtype=dtype)

    q_in = torch.randn(batch, seq_len, q_heads, head_dim, device=device, dtype=dtype)
    k_in = torch.randn(batch, seq_len, kv_heads, head_dim, device=device, dtype=dtype)

    q_ref, k_ref = op_ref(q_in.clone(), k_in.clone(), sin, cos, max_query_len=seq_len)
    q_out, k_out = op(q_in.clone(), k_in.clone(), sin, cos, max_query_len=seq_len)
    assert_close(q_out, q_ref)
    assert_close(k_out, k_ref)
