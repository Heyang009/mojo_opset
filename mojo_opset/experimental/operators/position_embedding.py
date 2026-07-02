import math
from typing import List
from typing import Optional
from typing import Tuple

import torch

from mojo_opset.core.operator import MojoOperator


class MojoRotaryEmbedding(MojoOperator):
    """Apply rotary embedding with an interface aligned to xpu_ops.modules.RotaryEmbedding."""

    def __init__(
        self,
        rotary_offset: int = 0,
        inplace: bool = False,
        interleaved: bool = False,
        dynamic_ntk: bool = False,
        graph: bool = False,
    ):
        super().__init__()
        self.rotary_offset = rotary_offset
        self.inplace = inplace
        self.interleaved = interleaved
        self.dynamic_ntk = dynamic_ntk
        self.graph = graph

    def extra_repr(self) -> str:
        return (
            f"{self.rotary_offset=}, {self.inplace=}, {self.interleaved=}, "
            f"{self.dynamic_ntk=}, {self.graph=}"
        ).replace("self.", "")

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.interleaved:
            return self._rotate_interleaved(x)
        return self._rotate_half(x)

    def _select_non_packed_positions(
        self,
        table: torch.Tensor,
        batch_size: int,
        seq_len: int,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.zeros(batch_size, dtype=torch.int32, device=table.device)
        offsets = torch.arange(seq_len, dtype=position_ids.dtype, device=position_ids.device)
        positions = position_ids.to(torch.long).unsqueeze(-1) + offsets.to(torch.long)
        if self.dynamic_ntk:
            batch_ids = torch.arange(batch_size, device=table.device).unsqueeze(-1)
            return table[batch_ids, positions]
        return table[positions]

    def _select_packed_positions(
        self,
        table: torch.Tensor,
        cumsum_query_len: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = cumsum_query_len.numel() - 1
        if position_ids is None:
            position_ids = torch.zeros(batch_size, dtype=torch.int32, device=cumsum_query_len.device)
        chunks = []
        for batch_id in range(batch_size):
            start = int(cumsum_query_len[batch_id].item())
            end = int(cumsum_query_len[batch_id + 1].item())
            seq_len = end - start
            if seq_len <= 0:
                continue
            positions = position_ids[batch_id].to(torch.long) + torch.arange(
                seq_len,
                dtype=torch.long,
                device=cumsum_query_len.device,
            )
            if self.dynamic_ntk:
                chunks.append(table[batch_id, positions])
            else:
                chunks.append(table[positions])
        if not chunks:
            return table.new_empty((0, table.shape[-1]))
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        input: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cumsum_query_len: Optional[torch.Tensor] = None,
        max_query_len: int = -1,
    ) -> torch.Tensor:
        assert input.dim() in (3, 4), "input must be 3D packed [T, N, D] or 4D [B, S, N, D]."
        assert sin.shape == cos.shape, "sin and cos must have the same shape."
        assert self.rotary_offset >= 0, "rotary_offset must be non-negative."

        rotary_dim = sin.shape[-1]
        rot_start = self.rotary_offset
        rot_end = rot_start + rotary_dim
        assert rot_end <= input.shape[-1], "rotary_offset + rotary_dim must be less than or equal to input head_dim."

        if input.dim() == 3:
            assert cumsum_query_len is not None, "cumsum_query_len is required for packed input."
            selected_sin = self._select_packed_positions(sin, cumsum_query_len, position_ids).unsqueeze(1)
            selected_cos = self._select_packed_positions(cos, cumsum_query_len, position_ids).unsqueeze(1)
        else:
            assert cumsum_query_len is None, "cumsum_query_len must be None for non-packed input."
            selected_sin = self._select_non_packed_positions(
                sin,
                input.shape[0],
                input.shape[1],
                position_ids,
            ).unsqueeze(2)
            selected_cos = self._select_non_packed_positions(
                cos,
                input.shape[0],
                input.shape[1],
                position_ids,
            ).unsqueeze(2)

        output = input if self.inplace else input.clone()
        rot_input = output[..., rot_start:rot_end]
        rot_output = (self._rotate(rot_input) * selected_sin + rot_input * selected_cos).to(input.dtype)
        output[..., rot_start:rot_end] = rot_output
        return output


class MojoRelativeEmbedding(MojoOperator):
    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
        """
        Initialize T5-style relative position embedding.

        Args:
            num_buckets (int): Number of relative position buckets.
            num_heads (int): Attention heads; also the embedding output channels.
            bidirectional (bool): If True, allocate half buckets for positive direction.
            max_dist (int, default=128): Maximum distance used in logarithmic bucketing.
        """
        super().__init__()
        if not isinstance(num_buckets, int) or num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(bidirectional, bool):
            raise TypeError("bidirectional must be a bool")
        if not isinstance(max_dist, int) or max_dist <= 0:
            raise ValueError("max_dist must be a positive integer")
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = torch.nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        """
        Compute relative position bias tensor for attention.

        Args:
            lq (int): Length of query sequence (Lq).
            lk (int): Length of key/value sequence (Lk).

        Returns:
            torch.Tensor: Bias tensor of shape [1, num_heads, Lq, Lk], dtype follows embedding weights.
        """
        if not isinstance(lq, int) or not isinstance(lk, int) or lq <= 0 or lk <= 0:
            raise ValueError("lq and lk must be positive integers")
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets

    def extra_repr(self) -> str:
        return f"{self.num_buckets=}, {self.num_heads=}, {self.bidirectional=}, {self.max_dist=}".replace("self.", "")


class MojoGridRoPE(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply 3D grid rotary position embeddings (RoPE) over (F, H, W) axes using
        precomputed per-sample frequency tensors.

        Args:
            x (torch.Tensor): [B, L, N, D]; D must be even (paired into complex components).
            grid_sizes (torch.Tensor): [B, 3] per-sample (F, H, W); seq_len = F*H*W.
            freqs_list (List[torch.Tensor]): length-B list; each item is a complex unit-phase tensor
                of shape [seq_len, 1, D/2], broadcastable to [seq_len, N, D/2].

        Returns:
            torch.Tensor: Same shape as `x`. Per sample, the first F*H*W tokens are rotated;
                remaining padding tokens are preserved. Output dtype matches input.
        """
        assert x.dim() == 4, "x must be 4D: [B, L, N, D]"
        assert x.size(-1) % 2 == 0, "D must be even for complex pairing"
        assert grid_sizes.dim() == 2 and grid_sizes.size(1) == 3, "grid_sizes must be [B, 3]"

        n = x.size(2)
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
            freqs_i = freqs_list[i]
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        y = torch.stack(output)
        return y.type_as(x)


class MojoMRoPEInplace(MojoOperator):
    """Multimodal Rotary Position Embedding (MRoPE) for Qwen2-VL."""

    supported_platforms_list = ["npu", "mlu", "meta_device", "ilu"]

    def __init__(self, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.inplace = inplace

    def extra_repr(self) -> str:
        return ""

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        hidden_states_half = hidden_size // 2
        left = hidden_states[..., :hidden_states_half]
        right = hidden_states[..., hidden_states_half:]
        return torch.cat((-right, left), dim=-1)

    @staticmethod
    def _apply_interleaved_mrope(
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_interleaved = cos_table[0].clone()
        cos_interleaved[..., 1 : mrope_section[1] * 3 : 3] = cos_table[1, ..., 1 : mrope_section[1] * 3 : 3]
        cos_interleaved[..., 2 : mrope_section[2] * 3 : 3] = cos_table[2, ..., 2 : mrope_section[2] * 3 : 3]

        sin_interleaved = sin_table[0].clone()
        sin_interleaved[..., 1 : mrope_section[1] * 3 : 3] = sin_table[1, ..., 1 : mrope_section[1] * 3 : 3]
        sin_interleaved[..., 2 : mrope_section[2] * 3 : 3] = sin_table[2, ..., 2 : mrope_section[2] * 3 : 3]

        return cos_interleaved, sin_interleaved

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
        head_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_query = query
        orig_key = key

        num_tokens, n_qh_head_dim = query.shape
        num_tokens_k, n_kh_head_dim = key.shape

        rope_dim = sum(mrope_section) * 2
        half_rope_dim = rope_dim // 2

        if head_dim is None:
            head_dim = rope_dim

        n_qh = n_qh_head_dim // head_dim
        n_kh = n_kh_head_dim // head_dim

        query = query.view(num_tokens, n_qh, head_dim)
        key = key.view(num_tokens_k, n_kh, head_dim)

        query_rot, query_pass = query.split([rope_dim, head_dim - rope_dim], dim=-1)
        key_rot, key_pass = key.split([rope_dim, head_dim - rope_dim], dim=-1)

        if cos_table.dim() == 3:
            if is_interleaved:
                cos_table, sin_table = self._apply_interleaved_mrope(cos_table, sin_table, mrope_section)
            else:
                cos_table = torch.cat([m[i] for i, m in enumerate(cos_table.split(mrope_section, dim=-1))], dim=-1)
                sin_table = torch.cat([m[i] for i, m in enumerate(sin_table.split(mrope_section, dim=-1))], dim=-1)

        cos_table = cos_table.view(num_tokens, half_rope_dim)
        sin_table = sin_table.view(num_tokens, half_rope_dim)

        query_rot_half1 = query_rot[..., :half_rope_dim]
        query_rot_half2 = query_rot[..., half_rope_dim:]
        key_rot_half1 = key_rot[..., :half_rope_dim]
        key_rot_half2 = key_rot[..., half_rope_dim:]

        cos_expanded = cos_table.unsqueeze(1)
        sin_expanded = sin_table.unsqueeze(1)

        query_rot_new_half1 = query_rot_half1 * cos_expanded - query_rot_half2 * sin_expanded
        query_rot_new_half2 = query_rot_half2 * cos_expanded + query_rot_half1 * sin_expanded
        key_rot_new_half1 = key_rot_half1 * cos_expanded - key_rot_half2 * sin_expanded
        key_rot_new_half2 = key_rot_half2 * cos_expanded + key_rot_half1 * sin_expanded

        query_rot = torch.cat([query_rot_new_half1, query_rot_new_half2], dim=-1)
        key_rot = torch.cat([key_rot_new_half1, key_rot_new_half2], dim=-1)

        query = torch.cat([query_rot, query_pass], dim=-1).view(num_tokens, -1)
        key = torch.cat([key_rot, key_pass], dim=-1).view(num_tokens_k, -1)

        if self.inplace:
            orig_query.copy_(query)
            orig_key.copy_(key)
            return orig_query, orig_key
        return query, key


class MojoFusedRMSNormRope(MojoOperator):
    """Fused Q/K RMSNorm + Rotary Embedding.

    对 query/key 各自的最后一维做 RMSNorm，再对 ``[rotary_offset, rotary_offset + rotary_dim)``
    范围做 RoPE。query/key 是 inplace 改写，返回值与输入是同一 tensor。

    Args:
        head_dim (int): query/key 的最后一维大小（HeadDim），即 RMSNorm 的 norm 维度。
        rotary_offset (int): RoPE 在最后一维的起始 offset。实际旋转范围
            ``[rotary_offset, rotary_offset + rotary_dim)``，rotary_dim 由 sin/cos 决定。
        eps (float): RMSNorm 数值稳定项，默认 1e-5。
        interleaved (bool): 是否使用 interleaved RoPE。默认 False（rotate-half）。
        dynamic_ntk (bool): 是否使用 dynamic NTK 模式。关闭时 sin/cos 形状
            ``(MaxPos, RotaryDim)``；开启时 ``(B, MaxPos, RotaryDim)``。

    Forward:
        query (torch.Tensor): ``(B, S, q_heads, head_dim)`` 或 packed ``(T, q_heads, head_dim)``。
        key (torch.Tensor): 同 query rank，head 维度为 k_heads。
        sin (torch.Tensor): ``(MaxPos, RotaryDim)`` 或 ``(B, MaxPos, RotaryDim)`` (dynamic_ntk)。
        cos (torch.Tensor): 同 sin。
        position_ids (torch.Tensor, optional): per-sequence 起始位置，shape ``(B,)``，dtype int32。
            None 表示从 0 开始。
        cumsum_query_len (torch.Tensor, optional): packed 输入必传，shape ``(B+1,)``，dtype int32。
        max_query_len (int): packed 模式最大 q 长度；非 packed 模式 = SeqLen，默认 -1。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 改写后的 (query, key)，与输入同一 storage。
    """

    def __init__(
        self,
        head_dim: int,
        rotary_offset: int = 0,
        eps: float = 1e-5,
        interleaved: bool = False,
        dynamic_ntk: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert head_dim > 0, "head_dim must be positive."
        assert rotary_offset >= 0, "rotary_offset must be non-negative."
        self.head_dim = head_dim
        self.rotary_offset = rotary_offset
        self.eps = eps
        self.interleaved = interleaved
        self.dynamic_ntk = dynamic_ntk
        # 权重存 fp32，对齐 xpu_ops 接口要求；放进 Parameter 让 state_dict 能存。
        factory_kwargs = dict(self.tensor_factory_kwargs)
        factory_kwargs["dtype"] = torch.float32
        self.q_rms_weight = torch.nn.Parameter(torch.empty(head_dim, **factory_kwargs))
        self.k_rms_weight = torch.nn.Parameter(torch.empty(head_dim, **factory_kwargs))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, rotary_offset={self.rotary_offset}, "
            f"eps={self.eps}, interleaved={self.interleaved}, dynamic_ntk={self.dynamic_ntk}"
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.interleaved:
            return self._rotate_interleaved(x)
        return self._rotate_half(x)

    def _select_packed_positions(
        self,
        table: torch.Tensor,
        cumsum_query_len: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = cumsum_query_len.numel() - 1
        if position_ids is None:
            position_ids = torch.zeros(batch_size, dtype=torch.int32, device=cumsum_query_len.device)
        chunks = []
        for batch_id in range(batch_size):
            start = int(cumsum_query_len[batch_id].item())
            end = int(cumsum_query_len[batch_id + 1].item())
            seq_len = end - start
            if seq_len <= 0:
                continue
            positions = position_ids[batch_id].to(torch.long) + torch.arange(
                seq_len, dtype=torch.long, device=cumsum_query_len.device,
            )
            if self.dynamic_ntk:
                chunks.append(table[batch_id, positions])
            else:
                chunks.append(table[positions])
        if not chunks:
            return table.new_empty((0, table.shape[-1]))
        return torch.cat(chunks, dim=0)

    def _select_non_packed_positions(
        self,
        table: torch.Tensor,
        batch_size: int,
        seq_len: int,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.zeros(batch_size, dtype=torch.int32, device=table.device)
        offsets = torch.arange(seq_len, dtype=position_ids.dtype, device=position_ids.device)
        positions = position_ids.to(torch.long).unsqueeze(-1) + offsets.to(torch.long)
        if self.dynamic_ntk:
            batch_ids = torch.arange(batch_size, device=table.device).unsqueeze(-1)
            return table[batch_ids, positions]
        return table[positions]

    def _apply_rmsnorm_inplace(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # RMSNorm in fp32 then cast back; weight 在 fp32 broadcast 到最后一维。
        orig_dtype = x.dtype
        x_fp = x.to(torch.float32)
        var = x_fp.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp * torch.rsqrt(var + self.eps)
        x_norm = x_norm * weight.to(torch.float32)
        x.copy_(x_norm.to(orig_dtype))
        return x

    def _apply_rope_inplace(
        self,
        x: torch.Tensor,
        sin_selected: torch.Tensor,  # 已按 head_first=False 维度展开过
        cos_selected: torch.Tensor,
    ) -> torch.Tensor:
        rotary_dim = sin_selected.shape[-1]
        rot_start = self.rotary_offset
        rot_end = rot_start + rotary_dim
        rot_input = x[..., rot_start:rot_end]
        rot_output = (self._rotate(rot_input) * sin_selected + rot_input * cos_selected).to(x.dtype)
        x[..., rot_start:rot_end] = rot_output
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cumsum_query_len: Optional[torch.Tensor] = None,
        max_query_len: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del max_query_len  # 仅为接口对齐 xpu_ops，参考实现不需要。
        assert query.dim() in (3, 4), "query must be 3D packed [T, N, D] or 4D [B, S, N, D]."
        assert query.dim() == key.dim(), "query and key must have the same rank."
        assert query.shape[-1] == self.head_dim and key.shape[-1] == self.head_dim, (
            "last dim of query/key must equal head_dim."
        )
        assert sin.shape == cos.shape, "sin and cos must have the same shape."
        rotary_dim = sin.shape[-1]
        assert self.rotary_offset + rotary_dim <= self.head_dim, (
            "rotary_offset + rotary_dim must be <= head_dim."
        )

        # 1. RMSNorm（inplace）
        self._apply_rmsnorm_inplace(query, self.q_rms_weight)
        self._apply_rmsnorm_inplace(key, self.k_rms_weight)

        # 2. RoPE（inplace）
        if query.dim() == 3:
            assert cumsum_query_len is not None, "cumsum_query_len is required for packed input."
            sin_sel = self._select_packed_positions(sin, cumsum_query_len, position_ids).unsqueeze(1)
            cos_sel = self._select_packed_positions(cos, cumsum_query_len, position_ids).unsqueeze(1)
        else:
            assert cumsum_query_len is None, "cumsum_query_len must be None for non-packed input."
            sin_sel = self._select_non_packed_positions(
                sin, query.shape[0], query.shape[1], position_ids,
            ).unsqueeze(2)
            cos_sel = self._select_non_packed_positions(
                cos, query.shape[0], query.shape[1], position_ids,
            ).unsqueeze(2)

        self._apply_rope_inplace(query, sin_sel, cos_sel)
        self._apply_rope_inplace(key, sin_sel, cos_sel)
        return query, key


__all__ = [
    "MojoRotaryEmbedding",
    "MojoRelativeEmbedding",
    "MojoGridRoPE",
    "MojoMRoPEInplace",
    "MojoFusedRMSNormRope",
]
