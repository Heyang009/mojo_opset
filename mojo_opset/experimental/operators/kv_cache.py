from typing import Optional
from typing import Tuple

import torch

from mojo_opset.core.operators.kv_cache import assert_paged_kv_layout_contract
from mojo_opset.core.operators.kv_cache import assert_paged_kv_store_contract
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata
from mojo_opset.core.operator import MojoOperator


class MojoStorePagedMLAKVCache(MojoOperator):
    """Append new MLA compressed-KV and positional-key tokens into paged caches.

    MLA (Multi-head Latent Attention) stores a low-rank compressed latent
    ``compressed_kv`` and a positional key ``k_pe`` instead of full K/V per
    head.  This operator writes incoming tokens into the block-based paged
    caches following the same block-table scheme as
    :class:`MojoStorePagedKVCache`.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        compressed_kv_states: torch.Tensor,
        k_pe_states: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_q_lens: torch.Tensor,
        context_kv_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            compressed_kv_states: ``(token_num, kv_lora_rank)`` new compressed
                KV latent tokens.
            k_pe_states: ``(token_num, qk_rope_head_dim)`` new positional key
                tokens.
            compressed_kv_cache: ``(N_blocks, 1, block_size, kv_lora_rank)``
                paged compressed-KV cache (modified in-place).
            k_pe_cache: ``(N_blocks, 1, block_size, qk_rope_head_dim)``
                paged positional-key cache (modified in-place).
            block_table: ``(B, max_blocks_per_seq)`` logical-to-physical block
                mapping.
            cu_q_lens: ``(B+1,)`` cumulative query lengths for prefill.
                ``None`` indicates decode mode (1 token per batch).
            context_kv_lens: ``(B,)`` history sequence lengths before
                storing the current tokens. Padding entries use -1.

        Returns:
            ``(compressed_kv_cache, k_pe_cache)`` after in-place writes.
        """
        assert_paged_kv_layout_contract(block_table, cu_q_lens, context_kv_lens)
        block_size = compressed_kv_cache.shape[2]
        num_batches = len(context_kv_lens) if context_kv_lens is not None else 0
        is_decode = cu_q_lens is None

        for batch_id in range(num_batches):
            if not is_decode:
                t_start = cu_q_lens[batch_id].item()
                t_end = cu_q_lens[batch_id + 1].item()
                seq_len = t_end - t_start
            else:
                t_start = batch_id
                t_end = batch_id + 1
                seq_len = 1

            if seq_len <= 0:
                continue

            ckv_slice = compressed_kv_states[t_start:t_end]  # (seq_len, kv_lora_rank)
            kpe_slice = k_pe_states[t_start:t_end]  # (seq_len, qk_rope_head_dim)

            write_start = context_kv_lens[batch_id].item()
            if write_start < 0:
                continue
            bt = block_table[batch_id]
            if bt.numel() == 0 or bt[0].item() < 0:
                continue

            blk_idx = write_start // block_size
            blk_off = write_start % block_size
            src = 0
            remain = seq_len

            while remain > 0:
                if blk_idx >= bt.shape[0]:
                    break
                phys_id = bt[blk_idx].item()
                if phys_id < 0:
                    break

                cap = block_size - blk_off
                n = min(remain, cap)

                compressed_kv_cache[phys_id, 0, blk_off : blk_off + n, :] = ckv_slice[src : src + n]
                k_pe_cache[phys_id, 0, blk_off : blk_off + n, :] = kpe_slice[src : src + n]

                src += n
                remain -= n
                blk_idx += 1
                blk_off = 0

        return compressed_kv_cache, k_pe_cache


class MojoStorePagedKVCacheC8(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        context_kv_lens: Optional[torch.Tensor] = None,
        *,
        chunk_metadata: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Copy new K/V tokens into a paged int8 per-channel quantized KV cache."""
        assert len(key_states.shape) == 3 and len(value_states.shape) == 3 and key_states.shape == value_states.shape, (
            "key/value states must be (token_num, kv_head_num, head_dim), please check."
        )
        if slot_mapping is not None:
            assert chunk_metadata is None, "slot_mapping path should not be mixed with chunk_metadata."
            assert block_table is None and cu_q_lens is None and context_kv_lens is None, (
                "slot_mapping path should not be mixed with block_table/cu_q_lens/context_kv_lens."
            )
            assert slot_mapping.dim() == 1, "slot_mapping must be a 1D tensor."
            assert slot_mapping.shape[0] == key_states.shape[0], "slot_mapping must have one entry per token."
        elif chunk_metadata is None:
            assert block_table is not None, "block_table is required when chunk_metadata is not provided."
            assert context_kv_lens is not None, "context_kv_lens is required when chunk_metadata is not provided."
            chunk_metadata = build_paged_kv_chunk_metadata(
                block_table,
                cu_q_lens,
                context_kv_lens,
                key_cache.shape[2],
            )
        else:
            assert block_table is None and cu_q_lens is None and context_kv_lens is None, (
                "chunk_metadata path should not be mixed with block_table/cu_q_lens/context_kv_lens."
            )

        assert key_scale is not None and value_scale is not None
        if chunk_metadata is not None:
            assert_paged_kv_store_contract(chunk_metadata)

        if (slot_mapping is not None and slot_mapping.numel() == 0) or (
            chunk_metadata is not None and chunk_metadata.shape[0] == 0
        ):
            return key_cache, value_cache

        key_q = torch.round(key_states / key_scale).clamp(-128, 127).to(torch.int8)
        value_q = torch.round(value_states / value_scale).clamp(-128, 127).to(torch.int8)

        if slot_mapping is not None:
            block_size = key_cache.shape[2]
            for src_token_id, slot in enumerate(slot_mapping.tolist()):
                slot = int(slot)
                if slot < 0:
                    continue
                dst_block_id = slot // block_size
                dst_block_offset = slot % block_size
                key_cache[dst_block_id, :, dst_block_offset, :] = key_q[src_token_id]
                value_cache[dst_block_id, :, dst_block_offset, :] = value_q[src_token_id]
        else:
            for src_token_start, dst_block_id, dst_block_offset, chunk_len in chunk_metadata.tolist():
                src_end = src_token_start + chunk_len
                dst_end = dst_block_offset + chunk_len
                key_cache[dst_block_id, :, dst_block_offset:dst_end, :] = key_q[src_token_start:src_end].permute(
                    1, 0, 2
                )
                value_cache[dst_block_id, :, dst_block_offset:dst_end, :] = value_q[src_token_start:src_end].permute(
                    1, 0, 2
                )

        return key_cache, value_cache


__all__ = [
    "MojoStorePagedMLAKVCache",
    "MojoStorePagedKVCacheC8",
    "MojoQuantQKVAndStoreKVCache",
]


class MojoQuantQKVAndStoreKVCache(MojoOperator):
    """Quantize Q/K/V and store quantized K/V into paged KV cache (C8).

    与 ``MojoStorePagedKVCacheC8`` 不同的是：
      - Q 在算子内做 per-token-per-head 动态量化（输出 int8 query + per-token-per-head scale）；
      - K 同样做 per-token-per-head 动态量化，并写入 paged key cache；
      - V 用外部传入的 per-channel 静态 scale 量化，并写入 paged value cache。

    Forward signature 对齐 ``xpu_ops.modules.QuantQKVAndStoreKVCache``，便于 xops 后端薄封装；
    参考实现走纯 torch，主要服务于精度对比。

    Args:
        block_size (int): paged cache 的 block_size，与 cache 第 3 维一致。
        slot_mapping_path (bool): 是否走 slot_mapping 直接写 paged cache（True 表示 forward
            必传 slot_mapping）。当前 ILU 默认 True。
    """

    def __init__(self, *, block_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        # block_size 留作可选 hint；实际 forward 也可以从 cache.shape[2] 读出。
        self.block_size = block_size

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}"

    @staticmethod
    def _per_token_per_head_quant_int8(
        x: torch.Tensor,  # [T, H, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_fp = x.to(torch.float32)
        amax = x_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [T, H, 1]
        scale = (amax / 127.0).squeeze(-1)  # [T, H]
        out = torch.round(x_fp / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
        # xpu_ops 返回 query scale 形状是 [H, T]（per-head 行向量），与 [T, H] 转置后等价。
        return out, scale.transpose(0, 1).contiguous()

    @staticmethod
    def _per_channel_quant_int8(
        x: torch.Tensor,         # [T, H, D]
        scale: torch.Tensor,     # [H, D]
    ) -> torch.Tensor:
        return torch.round(x.to(torch.float32) / scale).clamp(-128, 127).to(torch.int8)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        value_scale: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        cumsum_query_len: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        kv_ids: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        # 与 xpu_ops 接口保持兼容的 placeholder，参考实现仅用 slot_mapping
        key_scale_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run quant + paged store reference impl.

        Args:
            query / key / value: ``(T, *_heads, head_dim)`` 浮点输入。
            value_scale: ``(kv_heads, head_dim)`` per-channel value 量化 scale (fp32)。
            key_cache: ``(num_blocks, kv_heads, block_size, head_dim)``，dtype int8。
            value_cache: 形状 dtype 同 key_cache。
            slot_mapping: ``(T,)`` int32，每个 token 的全局 cache offset。
            cumsum_query_len / kv_len / kv_ids / block_table: 与 xpu_ops 接口对齐，参考实现不消费。
            key_scale_cache: 与 xpu_ops 接口兼容的 K scale cache（per-token-per-head），
                参考实现不读不写，只透传 None 用于对齐返回值数量。

        Returns:
            (output_quant_query, output_query_scale, key_cache, key_scale_cache,
             value_cache, value_scale_cache)。
            - output_quant_query: int8 ``(T, q_heads, head_dim)``
            - output_query_scale: fp32 ``(T, q_heads)``
            - key_cache, value_cache: 原 cache（已 inplace 改写）
            - key_scale_cache: 透传输入（可能为 None）
            - value_scale_cache: 始终 None（per-channel scale 是输入，不是输出）
        """
        del cumsum_query_len, kv_len, kv_ids, block_table  # 接口对齐占位

        assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3, (
            "query/key/value must be 3D packed [T, N, D]."
        )
        assert key.shape == value.shape, "key/value shape must match."
        assert key_cache.shape == value_cache.shape, "key/value cache shape must match."
        assert key_cache.dim() == 4, "cache must be (num_blocks, kv_heads, block_size, head_dim)."
        assert value_scale is not None, "value_scale (per-channel) is required."
        assert slot_mapping is not None, "slot_mapping is required (paged write path)."
        assert slot_mapping.dim() == 1 and slot_mapping.shape[0] == key.shape[0]

        # 1. Q 量化（per-token-per-head）
        q_quant, q_scale = self._per_token_per_head_quant_int8(query)

        # 2. K 量化（per-token-per-head），用于写 paged key cache
        k_quant, k_scale = self._per_token_per_head_quant_int8(key)

        # 3. V 量化（per-channel，scale 是输入）
        v_quant = self._per_channel_quant_int8(value, value_scale)

        # 4. paged write
        block_size = key_cache.shape[2]
        if key_scale_cache is None:
            num_blocks = key_cache.shape[0]
            kv_heads = key_cache.shape[1]
            key_scale_cache = torch.zeros(
                (num_blocks, kv_heads, block_size),
                dtype=torch.float32,
                device=key_cache.device,
            )
        # k_scale 形状 [H, T]，按 slot 写入 [num_blocks, H, block_size]
        for src_idx, slot in enumerate(slot_mapping.tolist()):
            slot = int(slot)
            if slot < 0:
                continue
            blk = slot // block_size
            off = slot % block_size
            key_cache[blk, :, off, :] = k_quant[src_idx]
            value_cache[blk, :, off, :] = v_quant[src_idx]
            key_scale_cache[blk, :, off] = k_scale[:, src_idx]

        return q_quant, q_scale, key_cache, key_scale_cache, value_cache, None
