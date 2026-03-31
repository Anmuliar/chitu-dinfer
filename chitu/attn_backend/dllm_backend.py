# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
DLLM Attention Backend - Specialized backend for dLLM (Diffusion LLM) inference.

Key features:
1. Bidirectional attention (causal=False)
2. KV Cache replacement mode (not append)
3. Flash Attention acceleration where possible
4. CUDA Graph compatible KV cache update
"""

from typing import Optional
from typing_extensions import override
import torch
import torch.nn.functional as F

from chitu.attn_backend.flash_attn_backend import FlashAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import DenseKVCacheAccessor, KVCacheManagerBase, PagedKVCacheAccessor
from chitu.ops import append_to_paged_kv_cache
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")


class DLLMAttnBackend(FlashAttnBackend):
    """
    Specialized attention backend for dLLM (Diffusion LLM).

    Core features:
    1. Bidirectional attention (causal=False)
    2. KV Cache replacement mode (instead of append)
    3. Flash Attention acceleration support
    4. CUDA Graph compatible operations
    """

    def __init__(self):
        super().__init__()
        self._cache_managers = None
        self._block_length = None
        self._cache_length = None

    def prepare_for_dllm_decode(
        self,
        cache_managers: dict[str, "KVCacheManagerBase"],
        block_length: int,
        cache_length: int,
    ):
        """Prepare for dLLM decode step.

        This should be called before CUDA graph capture or replay.

        Args:
            cache_managers: KV cache managers dict
            block_length: Current block length being decoded
            cache_length: Total cache length (including current block)
        """
        self._cache_managers = cache_managers
        self._block_length = block_length
        self._cache_length = cache_length

    def bidirectional_prefill(
        self,
        q: torch.Tensor,  # [total_q, nheads, head_dim]
        k: torch.Tensor,  # [total_k, nheads, head_dim]
        v: torch.Tensor,  # [total_k, nheads, head_dim]
        *,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale: Optional[float] = None,
        window_size=(-1, -1),
        softcap=0.0,
    ) -> torch.Tensor:
        """
        Prefill with bidirectional attention using Flash Attention varlen.

        Args:
            q, k, v: Query, Key, Value tensors in ragged format
            seq_len_delta: BatchedSeqLenDelta containing sequence length info
            softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
            window_size: Sliding window size (-1 means infinite)
            softcap: Softcap value (0 means disabled)

        Returns:
            Attention output tensor
        """
        return self.prefill_ragged_qkvo(
            q, k, v,
            seq_len_delta=seq_len_delta,
            causal=False,  # Bidirectional attention
            softmax_scale=softmax_scale,
            window_size=window_size,
            softcap=softcap,
        )

    def bidirectional_prefill_with_mask(
        self,
        q: torch.Tensor,  # [batch, heads, seq, head_dim]
        k: torch.Tensor,  # [batch, heads, seq, head_dim]
        v: torch.Tensor,  # [batch, heads, seq, head_dim]
        *,
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq, seq] or [batch, 1, seq, seq]
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Prefill with custom attention mask (e.g., block-diagonal attention).

        For complex attention masks, use PyTorch SDPA which handles arbitrary masks.

        Args:
            q, k, v: Query, Key, Value tensors [batch, heads, seq, head_dim]
            attention_mask: Attention mask tensor [batch, seq, seq] or [batch, 1, seq, seq]
            softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))

        Returns:
            Attention output tensor [batch, heads, seq, head_dim]
        """
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

        # Expand mask from [batch, seq, seq] to [batch, 1, seq, seq] if needed
        if attention_mask is not None and len(attention_mask.shape) == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,  # Bidirectional attention for dLLM
            scale=softmax_scale,
        )

    @override
    def decode_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """
        Decode with bidirectional attention.

        For single-token decode (s_q=1), causal setting doesn't affect results.
        For multi-token decode, we set causal=False for bidirectional attention.
        """
        if topk_indices is not None:
            raise NotImplementedError("topk_indices not supported in DLLMAttnBackend")

        if q.numel() == 0:
            return torch.empty(
                0, q.shape[1], kv_cache.v.shape[-1], device=q.device, dtype=q.dtype
            )

        # Extra kwargs for flash_attn
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        bsz = seq_len_delta.batch_size
        s_q = 1 if seq_len_delta.is_classic_decoding else getattr(self, "mtp_size", 1)

        # For dLLM, we want bidirectional attention (causal=False)
        # For single token decode (s_q=1), causal doesn't matter
        # For multi-token, we explicitly set causal=False
        output = self._fa.flash_attn_with_kvcache(
            q.view(bsz, s_q, q.shape[-2], q.shape[-1]),
            kv_cache.k,
            kv_cache.v,
            k=k.view(bsz, s_q, k.shape[-2], k.shape[-1]) if k is not None else None,
            v=v.view(bsz, s_q, v.shape[-2], v.shape[-1]) if v is not None else None,
            cache_seqlens=seq_len_delta.old.lens_tensor_device,
            causal=False,  # Bidirectional attention for dLLM
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
        )
        output = output.view(bsz * s_q, output.shape[-2], output.shape[-1])
        return output

    def decode_with_kv_replace(
        self,
        q: torch.Tensor,  # [batch, heads, seq_q, head_dim]
        kv_cache: DenseKVCacheAccessor,
        k: torch.Tensor,  # [batch, heads, seq_k, head_dim]
        v: torch.Tensor,  # [batch, heads, seq_k, head_dim]
        *,
        replace_start: int,
        replace_end: int,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        dLLM decode: Replace KV cache and execute bidirectional attention.

        This is the core operation for dLLM iterative decoding:
        1. Replace KV cache at specific positions using slice_scatter
        2. Execute bidirectional attention over the full cache

        Args:
            q: Query tensor [batch, heads, seq_q, head_dim]
            kv_cache: DenseKVCacheAccessor (will be modified in-place)
            k, v: Key/Value tensors to write [batch, heads, seq_k, head_dim]
            replace_start: Start position for KV replacement
            replace_end: End position for KV replacement
            softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))

        Returns:
            Attention output tensor [batch, heads, seq_q, head_dim]
        """
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

        # 1. Replace KV cache at specified positions
        # slice_scatter: scatter source tensor into destination at specified slice
        kv_cache.kv["k"] = kv_cache.kv["k"].slice_scatter(
            k, dim=2, start=replace_start, end=replace_end
        )
        kv_cache.kv["v"] = kv_cache.kv["v"].slice_scatter(
            v, dim=2, start=replace_start, end=replace_end
        )

        # 2. Execute bidirectional attention using Flash Attention
        # flash_attn_func expects [batch, seqlen, nheads, headdim]
        # We receive [batch, heads, seq, head_dim], need to transpose
        if has_flash_attn:
            q_t = q.transpose(1, 2)  # [batch, seq_q, heads, head_dim]
            k_t = kv_cache.k.transpose(1, 2)  # [batch, seq_k, heads, head_dim]
            v_t = kv_cache.v.transpose(1, 2)  # [batch, seq_k, heads, head_dim]

            output = flash_attn.flash_attn_func(
                q_t, k_t, v_t,
                causal=False,  # Bidirectional attention
                softmax_scale=softmax_scale,
            )
            # Transpose back to [batch, heads, seq_q, head_dim]
            return output.transpose(1, 2)
        else:
            # Fallback to SDPA if Flash Attention not available
            return F.scaled_dot_product_attention(
                q, kv_cache.k, kv_cache.v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )

    def decode_bidirectional_simple(
        self,
        q: torch.Tensor,  # [batch, heads, seq_q, head_dim]
        k: torch.Tensor,  # [batch, heads, seq_k, head_dim]
        v: torch.Tensor,  # [batch, heads, seq_k, head_dim]
        *,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Simple bidirectional attention for decode without KV cache.

        Args:
            q, k, v: Query, Key, Value tensors [batch, heads, seq, head_dim]
            softmax_scale: Scale factor for softmax

        Returns:
            Attention output tensor [batch, heads, seq_q, head_dim]
        """
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

        if has_flash_attn:
            # flash_attn_func expects [batch, seqlen, nheads, headdim]
            # We receive [batch, heads, seq, head_dim], need to transpose
            q_t = q.transpose(1, 2)  # [batch, seq_q, heads, head_dim]
            k_t = k.transpose(1, 2)  # [batch, seq_k, heads, head_dim]
            v_t = v.transpose(1, 2)  # [batch, seq_k, heads, head_dim]

            output = flash_attn.flash_attn_func(
                q_t, k_t, v_t,
                causal=False,
                softmax_scale=softmax_scale,
            )
            # Transpose back to [batch, heads, seq_q, head_dim]
            return output.transpose(1, 2)
        else:
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )

    def dllm_decode_attention(
        self,
        q: torch.Tensor,  # [batch, heads, seq_q, head_dim]
        k: torch.Tensor,  # [batch, kv_heads, seq_k, head_dim]
        v: torch.Tensor,  # [batch, kv_heads, seq_k, head_dim]
        layer_id: int,
        kv_accessor: DenseKVCacheAccessor,
        *,
        softmax_scale: Optional[float] = None,
        cache_length: Optional[int] = None,
        block_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        dLLM decode attention with KV cache update.

        This method is designed to be CUDA graph compatible:
        1. Uses in-place index assignment for KV cache update (graph-safe)
        2. Executes bidirectional attention over full cache

        IMPORTANT: This method updates KV cache in-place using index assignment.
        The KV cache tensor addresses must remain fixed across graph replays.

        Args:
            q: Query tensor [batch, heads, seq_q, head_dim]
            k: Key tensor to write to cache [batch, kv_heads, seq_k, head_dim]
            v: Value tensor to write to cache [batch, kv_heads, seq_k, head_dim]
            layer_id: Layer ID (for future use with cache_managers)
            kv_accessor: DenseKVCacheAccessor for KV cache access
            softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
            cache_length: Total cache length (uses self._cache_length if None)
            block_length: Block length to replace (uses self._block_length if None)

        Returns:
            Attention output tensor [batch, heads, seq_q, head_dim]
        """
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

        # Use provided values or fall back to instance values
        if cache_length is None:
            cache_length = self._cache_length
        if block_length is None:
            block_length = self._block_length

        if cache_length is None or block_length is None:
            raise ValueError(
                "cache_length and block_length must be provided either as arguments "
                "or through prepare_for_dllm_decode()"
            )

        # 1. Update KV cache: replace last block_length positions
        # Note: In-place index assignment is CUDA graph safe if tensor addresses are fixed
        cache_k = kv_accessor.k  # [batch, kv_heads, seq, head_dim]
        cache_v = kv_accessor.v

        replace_start = cache_length - block_length
        replace_end = cache_length

        # In-place update (graph-safe if tensor addresses are fixed)
        cache_k[:, :, replace_start:replace_end, :] = k
        cache_v[:, :, replace_start:replace_end, :] = v

        # 2. Execute bidirectional attention over full cache
        if has_flash_attn:
            # flash_attn_func expects [batch, seqlen, nheads, headdim]
            q_t = q.transpose(1, 2)
            k_t = cache_k.transpose(1, 2)
            v_t = cache_v.transpose(1, 2)

            output = flash_attn.flash_attn_func(
                q_t, k_t, v_t,
                causal=False,
                softmax_scale=softmax_scale,
            )
            return output.transpose(1, 2)
        else:
            return F.scaled_dot_product_attention(
                q, cache_k, cache_v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )

    def dllm_decode_attention_paged(
        self,
        q: torch.Tensor,  # [batch, heads, seq_q, head_dim]
        k: torch.Tensor,  # [batch, kv_heads, seq_k, head_dim]
        v: torch.Tensor,  # [batch, kv_heads, seq_k, head_dim]
        layer_id: int,
        kv_accessor: PagedKVCacheAccessor,
        *,
        decoding_start_list: list[int],
        softmax_scale: Optional[float] = None,
        cache_length: Optional[int] = None,
        block_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        dLLM decode attention with PagedKVCacheAccessor.

        This method is designed to be CUDA graph compatible:
        1. Updates paged KV cache using append_to_paged_kv_cache (triton kernel, graph-safe)
        2. Executes bidirectional attention using flash_attn_with_kvcache

        For dLLM, we REPLACE KV at specific positions (decoding_start to decoding_start + block_length),
        not append. This is achieved by passing the correct position_ids.

        Args:
            q: Query tensor [batch, heads, seq_q, head_dim]
            k: Key tensor to write to cache [batch, kv_heads, seq_k, head_dim]
            v: Value tensor to write to cache [batch, kv_heads, seq_k, head_dim]
            layer_id: Layer ID
            kv_accessor: PagedKVCacheAccessor for KV cache access
            decoding_start_list: List of starting positions for each sequence
            softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
            cache_length: Total cache length (uses self._cache_length if None)
            block_length: Block length to replace (uses self._block_length if None)

        Returns:
            Attention output tensor [batch, heads, seq_q, head_dim]
        """
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

        # Use provided values or fall back to instance values
        if cache_length is None:
            cache_length = self._cache_length
        if block_length is None:
            block_length = self._block_length

        if cache_length is None or block_length is None:
            raise ValueError(
                "cache_length and block_length must be provided either as arguments "
                "or through prepare_for_dllm_decode()"
            )

        batch_size = q.shape[0]
        device = q.device

        # 1. Update paged KV cache: replace positions [decoding_start, decoding_start + block_length)
        # Build delta_position_ids and delta_seq_ids for append_to_paged_kv_cache
        delta_pos_list = []
        delta_seq_list = []
        for i, ds in enumerate(decoding_start_list):
            for p in range(block_length):
                delta_pos_list.append(ds + p)
                delta_seq_list.append(i)

        delta_position_ids = torch.tensor(delta_pos_list, device=device, dtype=torch.long)
        delta_seq_ids = torch.tensor(delta_seq_list, device=device, dtype=torch.long)

        # k, v shape: [batch, kv_heads, block_length, head_dim]
        # Need to reshape to [batch * block_length, kv_heads, head_dim] for append_to_paged_kv_cache
        k_ragged = k.transpose(1, 2).reshape(-1, k.shape[1], k.shape[-1]).contiguous()
        v_ragged = v.transpose(1, 2).reshape(-1, v.shape[1], v.shape[-1]).contiguous()

        # Write K and V to paged cache
        for kv_name, kv_tensor in [("k", k_ragged), ("v", v_ragged)]:
            append_to_paged_kv_cache(
                kv_accessor.kv[kv_name],
                kv_accessor.block_table,
                kv_tensor,
                delta_position_ids,
                delta_seq_ids,
                get_page_ids=kv_accessor.get_page_ids,
                get_offs_in_page=kv_accessor.get_offs_in_page,
                use_i64_offsets=kv_accessor.use_i64_offsets,
            )

        # 2. Execute bidirectional attention using flash_attn_with_kvcache
        # This reads from the updated paged cache
        if has_flash_attn:
            # flash_attn_with_kvcache expects:
            # q: [batch, seqlen, nheads, headdim]
            # k_cache, v_cache: paged cache [num_blocks, block_size, nheads, headdim]
            q_t = q.transpose(1, 2)  # [batch, seq_q, heads, head_dim]

            # Build cache_seqlens: current length of each sequence in the cache
            cache_seqlens = torch.tensor(
                [ds + block_length for ds in decoding_start_list],
                device=device,
                dtype=torch.int32,
            )

            # Build kwargs for flash_attn_with_kvcache
            kwargs = dict(
                q=q_t,
                k_cache=kv_accessor.k,  # paged K cache [num_blocks, block_size, kv_heads, head_dim]
                v_cache=kv_accessor.v,  # paged V cache
                k=None,  # Already written to cache
                v=None,
                cache_seqlens=cache_seqlens,
                causal=False,  # Bidirectional attention for dLLM
                softmax_scale=softmax_scale,
            )

            # Add block_table/page_table based on Flash Attention version
            if hasattr(self, "_use_fa3") and self._use_fa3:
                # FA3 uses page_table
                kwargs["page_table"] = kv_accessor.block_table
            else:
                # FA2 uses block_table
                kwargs["block_table"] = kv_accessor.block_table

            output = self._fa.flash_attn_with_kvcache(**kwargs)
            # Transpose back to [batch, heads, seq_q, head_dim]
            return output.transpose(1, 2)
        else:
            # Fallback: For paged cache without Flash Attention, we need to gather to dense
            # This is slower but functional
            raise NotImplementedError(
                "dLLM decode with paged KV cache requires Flash Attention. "
                "Please install flash-attn package."
            )
