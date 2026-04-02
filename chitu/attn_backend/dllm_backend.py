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

from typing import Optional, Tuple, List
from typing_extensions import override
import torch
import torch.nn.functional as F

from chitu.attn_backend.flash_attn_backend import FlashAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import DenseKVCacheAccessor, KVCacheManagerBase, PagedKVCacheAccessor
from chitu.ops import append_to_paged_kv_cache, read_from_paged_kv_cache
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
        # State management for prefill/decode phases
        self._cache_managers = None
        self._block_length = None
        self._cache_length = None
        self._is_prefill = True  # Current phase
        self._num_layers = None
        self._prefilling_lengths = None
        self._decoding_start_list = None
        self._batch_size = None
        self._attention_mask = None  # For prefill phase
        # 存储 decode 阶段的 K、V tensor
        # shape: [num_layers, 2, batch * block_len, n_kv_heads, head_dim]
        # 其中 dim=1: 0 for K, 1 for V
        self._decode_kv_cache = None
        self._kv_heads = None
        self._head_dim = None

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

    def prepare_prefill(
        self,
        cache_managers: dict[str, "KVCacheManagerBase"],
        num_layers: int,
        prefilling_lengths: list[int],
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Prepare for dLLM prefill step.

        Args:
            cache_managers: KV cache managers dict
            num_layers: Number of layers in the model
            prefilling_lengths: List of prefilling lengths for each sequence
            batch_size: Batch size
            attention_mask: Attention mask for block-diagonal attention
        """
        self._is_prefill = True
        self._cache_managers = cache_managers
        self._num_layers = num_layers
        self._prefilling_lengths = prefilling_lengths
        self._batch_size = batch_size
        self._attention_mask = attention_mask

    def prepare_decode(
        self,
        cache_managers: dict[str, "KVCacheManagerBase"],
        num_layers: int,
        decoding_start_list: list[int],
        block_length: int,
        batch_size: int,
        kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Prepare for dLLM decode step.

        Args:
            cache_managers: KV cache managers dict
            num_layers: Number of layers in the model
            decoding_start_list: List of starting positions for each sequence
            block_length: Block length being decoded
            batch_size: Batch size
            kv_heads: Number of KV heads (for pre-allocating KV cache tensor)
            head_dim: Head dimension (for pre-allocating KV cache tensor)
            device: Device for KV cache tensor
            dtype: Data type for KV cache tensor
        """
        self._is_prefill = False
        self._cache_managers = cache_managers
        self._num_layers = num_layers
        self._decoding_start_list = decoding_start_list
        self._block_length = block_length
        self._batch_size = batch_size

        # 预分配 KV cache tensor
        # shape: [num_layers, 2, batch * block_len, n_kv_heads, head_dim]
        if kv_heads is not None and head_dim is not None and device is not None and dtype is not None:
            self._kv_heads = kv_heads
            self._head_dim = head_dim
            self._decode_kv_cache = torch.zeros(
                num_layers, 2, batch_size * block_length, kv_heads, head_dim,
                device=device, dtype=dtype
            )
        else:
            self._decode_kv_cache = None

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

    @override
    def __call__(
        self,
        q: torch.Tensor,
        kv_cache,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        layer_id: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Unified entry point for dLLM attention.

        Dispatches to prefill or decode based on _is_prefill state.
        Handles KV cache read/write internally.

        Args:
            q: Query tensor [total_q, nheads, headdim]
            kv_cache: KVCacheAccessor (PagedKVCacheAccessor for DLLM)
            k: Key tensor [total_k, nheads_k, headdim]
            v: Value tensor [total_k, nheads_k, headdim]
            seq_len_delta: BatchedSeqLenDelta containing sequence length info
            causal: Always False for dLLM (bidirectional attention)
            layer_id: Layer ID for cache indexing
            attention_mask: Attention mask for prefill phase
            **kwargs: Additional arguments

        Returns:
            Attention output tensor
        """
        if self._is_prefill:
            return self._prefill_attention(
                q, kv_cache, k, v,
                seq_len_delta=seq_len_delta,
                attention_mask=attention_mask,
                layer_id=layer_id,
            )
        else:
            return self._decode_attention(
                q, kv_cache, k, v,
                seq_len_delta=seq_len_delta,
                layer_id=layer_id,
            )

    def _prefill_attention(
        self,
        q: torch.Tensor,
        kv_cache,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        attention_mask: Optional[torch.Tensor],
        layer_id: int,
    ) -> torch.Tensor:
        """
        Prefill phase: compute bidirectional attention and write KV to paged cache.

        Args:
            q: Query tensor [total_q, nheads, headdim] (ragged format)
            kv_cache: PagedKVCacheAccessor
            k: Key tensor [total_k, nheads_k, headdim] (ragged format)
            v: Value tensor [total_k, nheads_k, headdim] (ragged format)
            seq_len_delta: Sequence length delta info
            attention_mask: Block-diagonal attention mask [batch, seq, seq]
            layer_id: Layer ID

        Returns:
            Attention output tensor [total_q, nheads, headdim]
        """
        # For prefill with block-diagonal mask, we need to use SDPA
        # since flash_attn_varlen doesn't support custom masks
        if attention_mask is not None:
            # Convert from ragged to dense format for SDPA
            # q, k, v are in ragged format [total_tokens, heads, head_dim]
            # attention_mask is [batch, seq, seq]

            batch_size = seq_len_delta.batch_size

            # Reshape to [batch, seq, heads, head_dim]
            # For now, use the attention mask shape to determine dimensions
            if len(attention_mask.shape) == 3:
                attn_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq, seq]
            else:
                attn_mask = attention_mask

            # Get dimensions from attention mask
            _, seq_q, seq_k = attention_mask.shape[-3:]

            # Get head info
            n_heads = q.shape[1]
            n_kv_heads = k.shape[1]
            head_dim = q.shape[2]

            # Create dense tensors [batch, heads, seq, head_dim]
            q_dense = torch.zeros(batch_size, n_heads, seq_q, head_dim,
                                  device=q.device, dtype=q.dtype)
            k_dense = torch.zeros(batch_size, n_kv_heads, seq_k, head_dim,
                                  device=k.device, dtype=k.dtype)
            v_dense = torch.zeros(batch_size, n_kv_heads, seq_k, head_dim,
                                  device=v.device, dtype=v.dtype)

            # Scatter ragged to dense - data is in [batch * seq, heads, head_dim] format
            # where each batch's tokens are contiguous
            for i in range(batch_size):
                # Each batch has seq_q tokens, placed contiguously
                start_idx = i * seq_q
                end_idx = (i + 1) * seq_q
                q_dense[i, :, :, :] = q[start_idx:end_idx].transpose(0, 1)
                k_dense[i, :, :, :] = k[start_idx:end_idx].transpose(0, 1)
                v_dense[i, :, :, :] = v[start_idx:end_idx].transpose(0, 1)

            # Handle GQA: repeat K and V if needed to match Q's number of heads
            if n_heads != n_kv_heads:
                n_rep = n_heads // n_kv_heads
                k_dense = k_dense.repeat_interleave(n_rep, dim=1)
                v_dense = v_dense.repeat_interleave(n_rep, dim=1)

            # Use bidirectional_prefill_with_mask
            output = self.bidirectional_prefill_with_mask(
                q_dense, k_dense, v_dense,
                attention_mask=attn_mask,
            )

            # Convert back to ragged format [total_tokens, heads, head_dim]
            output = output.transpose(1, 2)  # [batch, seq, heads, head_dim]
            output = output.reshape(-1, n_heads, head_dim)

            # Write KV to paged cache after attention computation
            # Use the same format as the original executor.py
            if isinstance(kv_cache, PagedKVCacheAccessor):
                # Write K and V to paged cache using seq_len_delta for position info
                # Note: we use the original k, v (not the repeated ones for GQA)
                append_to_paged_kv_cache(
                    kv_cache.k,
                    kv_cache.block_table,
                    k.contiguous(),
                    seq_len_delta.delta_position_ids_tensor_device,
                    seq_len_delta.delta_seq_ids_tensor_device,
                    get_page_ids=kv_cache.get_page_ids,
                    get_offs_in_page=kv_cache.get_offs_in_page,
                    use_i64_offsets=kv_cache.use_i64_offsets,
                )
                append_to_paged_kv_cache(
                    kv_cache.v,
                    kv_cache.block_table,
                    v.contiguous(),
                    seq_len_delta.delta_position_ids_tensor_device,
                    seq_len_delta.delta_seq_ids_tensor_device,
                    get_page_ids=kv_cache.get_page_ids,
                    get_offs_in_page=kv_cache.get_offs_in_page,
                    use_i64_offsets=kv_cache.use_i64_offsets,
                )

            return output
        else:
            # No attention mask, use standard ragged prefill
            # First write KV to paged cache
            if isinstance(kv_cache, PagedKVCacheAccessor):
                if k is not None:
                    append_to_paged_kv_cache(
                        kv_cache.k,
                        kv_cache.block_table,
                        k.contiguous(),
                        seq_len_delta.delta_position_ids_tensor_device,
                        seq_len_delta.delta_seq_ids_tensor_device,
                        get_page_ids=kv_cache.get_page_ids,
                        get_offs_in_page=kv_cache.get_offs_in_page,
                        use_i64_offsets=kv_cache.use_i64_offsets,
                    )
                if v is not None:
                    append_to_paged_kv_cache(
                        kv_cache.v,
                        kv_cache.block_table,
                        v.contiguous(),
                        seq_len_delta.delta_position_ids_tensor_device,
                        seq_len_delta.delta_seq_ids_tensor_device,
                        get_page_ids=kv_cache.get_page_ids,
                        get_offs_in_page=kv_cache.get_offs_in_page,
                        use_i64_offsets=kv_cache.use_i64_offsets,
                    )

            return self.prefill_ragged_qkvo(
                q, k, v,
                seq_len_delta=seq_len_delta,
                causal=False,
            )

    def _decode_attention(
        self,
        q: torch.Tensor,
        kv_cache,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Decode phase: read KV from paged cache, compute attention, write back.

        For dLLM, decode operates on block_length tokens at once, replacing
        KV cache positions [decoding_start, decoding_start + block_length).

        Args:
            q: Query tensor [batch * block_len, nheads, headdim]
            kv_cache: PagedKVCacheAccessor
            k: Key tensor [batch * block_len, nheads_k, headdim]
            v: Value tensor [batch * block_len, nheads_k, headdim]
            seq_len_delta: Sequence length delta info
            layer_id: Layer ID

        Returns:
            Attention output tensor [batch * block_len, nheads, headdim]
        """
        # 存储当前层的 K、V 到预分配的 tensor 中
        if self._decode_kv_cache is not None:
            # 直接写入预分配的 tensor，避免 detach().clone()
            # shape: [num_layers, 2, batch * block_len, n_kv_heads, head_dim]
            self._decode_kv_cache[layer_id, 0].copy_(k)
            self._decode_kv_cache[layer_id, 1].copy_(v)

        # For dLLM decode, we use bidirectional attention over the full cache
        # The KV cache is already populated from prefill or previous decode steps

        # Get batch dimensions
        block_length = self._block_length or seq_len_delta.delta_max_len
        batch_size = self._batch_size or seq_len_delta.batch_size
        decoding_start_list = self._decoding_start_list

        if decoding_start_list is None:
            # Fallback: no past KV, just use current K, V
            block_length = seq_len_delta.delta_max_len
            batch_size = seq_len_delta.batch_size

            # Reshape from ragged to [batch, block_len, heads, head_dim]
            q_4d = q.view(batch_size, block_length, q.shape[1], q.shape[2])
            k_4d = k.view(batch_size, block_length, k.shape[1], k.shape[2])
            v_4d = v.view(batch_size, block_length, v.shape[1], v.shape[2])

            # Transpose to [batch, heads, seq, head_dim] for attention
            q_t = q_4d.transpose(1, 2)
            k_t = k_4d.transpose(1, 2)
            v_t = v_4d.transpose(1, 2)

            output = self.decode_bidirectional_simple(
                q_t, k_t, v_t,
                softmax_scale=1.0 / (q.shape[-1] ** 0.5),
            )

            output = output.transpose(1, 2)
            return output.reshape(batch_size * block_length, output.shape[2], output.shape[3])

        # Get dimensions
        n_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        device = q.device
        dtype = q.dtype

        # Calculate cache length (aligned to power of 2)
        current_cache_length = max(decoding_start_list) + block_length
        def align_exp2(x):
            return 1 << (x - 1).bit_length() if x > 0 else 1
        current_cache_length = max(128, align_exp2(current_cache_length))

        # Read past KV from paged cache if available
        # Check if ANY batch has historical KV to read (not just batch 0)
        if isinstance(kv_cache, PagedKVCacheAccessor) and any(ds > 0 for ds in decoding_start_list):
            # Build position and sequence IDs for reading past KV
            pos_list = []
            seq_list = []
            for i, ds in enumerate(decoding_start_list):
                for p in range(ds):
                    pos_list.append(p)
                    seq_list.append(i)

            if pos_list:
                position_ids = torch.tensor(pos_list, device=device, dtype=torch.long)
                seq_ids = torch.tensor(seq_list, device=device, dtype=torch.long)

                # Read past K and V
                past_k = read_from_paged_kv_cache(
                    kv_cache.k,
                    kv_cache.block_table,
                    position_ids,
                    seq_ids,
                )
                past_v = read_from_paged_kv_cache(
                    kv_cache.v,
                    kv_cache.block_table,
                    position_ids,
                    seq_ids,
                )

                # Create dense tensors [batch, heads, cache_len, head_dim]
                past_k_dense = torch.zeros(
                    batch_size, n_kv_heads, current_cache_length, head_dim,
                    device=device, dtype=dtype
                )
                past_v_dense = torch.zeros(
                    batch_size, n_kv_heads, current_cache_length, head_dim,
                    device=device, dtype=dtype
                )

                # Scatter ragged to dense
                # past_k shape: [total_past_tokens, n_kv_heads, head_dim]
                past_k_dense[:, :, :max(decoding_start_list), :].permute(0, 2, 1, 3)[seq_ids, position_ids, :, :] = past_k
                past_v_dense[:, :, :max(decoding_start_list), :].permute(0, 2, 1, 3)[seq_ids, position_ids, :, :] = past_v

                # Reshape current K, V to [batch, kv_heads, block_len, head_dim]
                k_4d = k.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)
                v_4d = v.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)

                # Merge past KV with current KV
                full_k = torch.cat([past_k_dense, k_4d], dim=2)
                full_v = torch.cat([past_v_dense, v_4d], dim=2)

                # Handle GQA: repeat KV if needed
                if n_heads != n_kv_heads:
                    n_rep = n_heads // n_kv_heads
                    full_k = full_k.repeat_interleave(n_rep, dim=1)
                    full_v = full_v.repeat_interleave(n_rep, dim=1)

                # Reshape Q to [batch, heads, block_len, head_dim]
                q_4d = q.view(batch_size, block_length, n_heads, head_dim).transpose(1, 2)

                # Compute bidirectional attention
                output = self.decode_bidirectional_simple(
                    q_4d, full_k, full_v,
                    softmax_scale=1.0 / (head_dim ** 0.5),
                )

                # Transpose back to ragged format
                output = output.transpose(1, 2)  # [batch, block_len, heads, head_dim]
                return output.reshape(batch_size * block_length, n_heads, head_dim)

        # No past KV, just use current K, V
        q_4d = q.view(batch_size, block_length, n_heads, head_dim).transpose(1, 2)
        k_4d = k.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)
        v_4d = v.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)

        # Handle GQA
        if n_heads != n_kv_heads:
            n_rep = n_heads // n_kv_heads
            k_4d = k_4d.repeat_interleave(n_rep, dim=1)
            v_4d = v_4d.repeat_interleave(n_rep, dim=1)

        output = self.decode_bidirectional_simple(
            q_4d, k_4d, v_4d,
            softmax_scale=1.0 / (head_dim ** 0.5),
        )

        output = output.transpose(1, 2)
        return output.reshape(batch_size * block_length, n_heads, head_dim)

    def write_finished_kv_cache(
        self,
        block_finished_list: list[bool],
        batch_size: int,
    ):
        """
        Write KV cache for finished blocks to paged cache.

        This should be called after decode attention when some blocks are finished.
        Only writes KV for sequences where block_finished is True.

        Args:
            block_finished_list: List of booleans indicating which sequences finished their block
            batch_size: Batch size
        """
        if not any(block_finished_list) or self._decode_kv_cache is None:
            return

        # 只写入 block_finished 的 batch
        finished_indices = [i for i, f in enumerate(block_finished_list) if f]
        if not finished_indices:
            return

        device = self._decode_kv_cache.device
        n_kv_heads = self._kv_heads
        head_dim = self._head_dim

        # 预构建 position_ids 和 seq_ids（所有层共用）
        delta_pos_list = []
        delta_seq_list = []
        for orig_idx in finished_indices:
            ds = self._decoding_start_list[orig_idx]
            delta_pos_list.extend(range(ds, ds + self._block_length))
            delta_seq_list.extend([orig_idx] * self._block_length)  # Use orig_idx to index into full batch block_table

        delta_position_ids = torch.tensor(delta_pos_list, device=device, dtype=torch.long)
        delta_seq_ids = torch.tensor(delta_seq_list, device=device, dtype=torch.long)

        # 遍历每一层
        for layer_id in range(self._num_layers):
            # 从预分配的 tensor 中获取 K、V
            # shape: [batch * block_len, n_kv_heads, head_dim]
            layer_k = self._decode_kv_cache[layer_id, 0]
            layer_v = self._decode_kv_cache[layer_id, 1]

            for mgr in self._cache_managers.values():
                try:
                    accessor = mgr.get_accessor(layer_id)
                except KeyError:
                    continue

                # 提取已完成 batch 的 K、V
                # layer_k, layer_v: [batch * block_len, n_kv_heads, head_dim]
                finished_k = layer_k.view(batch_size, self._block_length, n_kv_heads, head_dim)[finished_indices]
                finished_v = layer_v.view(batch_size, self._block_length, n_kv_heads, head_dim)[finished_indices]

                # 写入 K
                append_to_paged_kv_cache(
                    accessor.k,
                    accessor.block_table,
                    finished_k.reshape(-1, n_kv_heads, head_dim).contiguous(),
                    delta_position_ids,
                    delta_seq_ids,
                    get_page_ids=accessor.get_page_ids,
                    get_offs_in_page=accessor.get_offs_in_page,
                    use_i64_offsets=accessor.use_i64_offsets,
                )
                # 写入 V
                append_to_paged_kv_cache(
                    accessor.v,
                    accessor.block_table,
                    finished_v.reshape(-1, n_kv_heads, head_dim).contiguous(),
                    delta_position_ids,
                    delta_seq_ids,
                    get_page_ids=accessor.get_page_ids,
                    get_offs_in_page=accessor.get_offs_in_page,
                    use_i64_offsets=accessor.use_i64_offsets,
                )

