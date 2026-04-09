# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""DLLM Attention Backend - Bidirectional attention for dLLM inference."""

from typing import Optional
from typing_extensions import override
import torch
import torch.nn.functional as F

from chitu.attn_backend.flash_attn_backend import FlashAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import DenseKVCacheAccessor, PagedKVCacheAccessor, PagedKVCache
from chitu.ops import append_to_paged_kv_cache, read_from_paged_kv_cache
from chitu.static_tensor import StaticTensor
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")


class DLLMAttnBackend(FlashAttnBackend):
    """Specialized attention backend for dLLM with bidirectional attention."""

    def __init__(self):
        super().__init__()
        self._cache_dict = None
        self._block_length = None
        self._is_prefill = True
        self._num_layers = None
        self._prefilling_lengths = None
        self._decoding_start = None  # Tensor [batch_size]
        self._batch_size = None
        self._attention_mask = None
        self._decode_kv_cache = None  # [num_layers, 2, batch*block_len, n_kv_heads, head_dim]
        self._kv_heads = None
        self._head_dim = None
        self._static_tensors = {}
        self._max_cache_length = 0
        self._max_batch_size = 0
        self._use_cuda_graph = False

    def prepare_prefill(
        self,
        cache_dict: dict[str, "PagedKVCache"],
        num_layers: int,
        prefilling_lengths: list[int],
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        self._is_prefill = True
        self._cache_dict = cache_dict
        self._num_layers = num_layers
        self._prefilling_lengths = prefilling_lengths
        self._batch_size = batch_size
        self._attention_mask = attention_mask

    def prepare_decode(
        self,
        cache_dict: dict[str, "PagedKVCache"],
        num_layers: int,
        decoding_start: torch.Tensor,
        block_length: int,
        batch_size: int,
        kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._is_prefill = False
        self._cache_dict = cache_dict
        self._num_layers = num_layers
        self._decoding_start = decoding_start
        self._block_length = block_length
        self._batch_size = batch_size
        self._kv_heads = kv_heads
        self._head_dim = head_dim

        if not self._use_cuda_graph:
            self._decode_kv_cache = torch.zeros(
                num_layers, 2, batch_size * block_length, kv_heads, head_dim,
                device=device, dtype=dtype
            )

    def init_static_tensors_for_decode(
        self,
        max_batch_size: int,
        max_cache_length: int,
        kv_heads: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._max_batch_size = max_batch_size
        self._max_cache_length = max_cache_length
        self._use_cuda_graph = True

        aligned_cache_length = max(128, 1 << max(0, (max_cache_length - 1)).bit_length())
        max_total_past_tokens = max_batch_size * max_cache_length

        # Metadata tensors
        self._static_tensors["decoding_start"] = StaticTensor(
            torch.zeros(max_batch_size, dtype=torch.long, device=device), max_nelem=max_batch_size)
        self._static_tensors["batch_size"] = StaticTensor(
            torch.zeros(1, dtype=torch.long, device=device), max_nelem=1)
        self._static_tensors["max_ds"] = StaticTensor(
            torch.zeros(1, dtype=torch.long, device=device), max_nelem=1)
        self._static_tensors["total_past"] = StaticTensor(
            torch.zeros(1, dtype=torch.long, device=device), max_nelem=1)
        self._static_tensors["position_ids"] = StaticTensor(
            torch.empty(max_total_past_tokens, dtype=torch.long, device=device),
            max_nelem=max_total_past_tokens)
        self._static_tensors["seq_ids"] = StaticTensor(
            torch.empty(max_total_past_tokens, dtype=torch.long, device=device),
            max_nelem=max_total_past_tokens)

        # KV cache tensors per layer
        kv_shape = (num_layers, max_batch_size, kv_heads, aligned_cache_length, head_dim)
        kv_nelem = num_layers * max_batch_size * kv_heads * aligned_cache_length * head_dim
        for name in ["past_k_dense_per_layer", "past_v_dense_per_layer", "full_k_per_layer", "full_v_per_layer"]:
            self._static_tensors[name] = StaticTensor(
                torch.empty(*kv_shape, dtype=dtype, device=device), max_nelem=kv_nelem)

        # Attention mask and write positions
        self._static_tensors["kv_valid_mask"] = StaticTensor(
            torch.zeros(max_batch_size, aligned_cache_length, dtype=torch.bool, device=device),
            max_nelem=max_batch_size * aligned_cache_length)
        self._static_tensors["kv_write_positions"] = StaticTensor(
            torch.zeros(max_batch_size, aligned_cache_length, dtype=torch.long, device=device),
            max_nelem=max_batch_size * aligned_cache_length)

        # Decode KV cache for current block
        max_block_length = 128
        self._static_tensors["decode_kv_cache"] = StaticTensor(
            torch.zeros(num_layers, 2, max_batch_size * max_block_length, kv_heads, head_dim,
                        dtype=dtype, device=device),
            max_nelem=num_layers * 2 * max_batch_size * max_block_length * kv_heads * head_dim)
        self._decode_kv_cache = self._static_tensors["decode_kv_cache"].get()
        self._max_block_length = max_block_length
        self._max_ds_for_graph = None

    def update_static_tensors_for_decode(self, decoding_start: torch.Tensor, batch_size: int):
        """Update static tensors before CUDA Graph replay."""
        self._decoding_start = decoding_start
        self._batch_size = batch_size

        if not self._static_tensors:
            return

        # Store decoding_start to static tensor
        decoding_start_padded = torch.zeros(self._max_batch_size, dtype=torch.long, device=decoding_start.device)
        decoding_start_padded[:batch_size] = decoding_start
        self._static_tensors["decoding_start"].set(decoding_start_padded)
        self._static_tensors["batch_size"].set(torch.tensor([batch_size], dtype=torch.long, device=decoding_start.device))

        max_ds = decoding_start.max().item() if batch_size > 0 else 0
        has_historical_kv = (decoding_start > 0).any().item()
        self._max_ds_for_graph = max_ds
        self._graph_write_start = max_ds
        self._static_tensors["max_ds"].set(torch.tensor([max_ds], dtype=torch.long, device=decoding_start.device))

        full_k = self._static_tensors["full_k_per_layer"].get()
        full_v = self._static_tensors["full_v_per_layer"].get()
        cache_len = full_k.shape[3]
        block_length = self._block_length
        device = full_k.device

        # Build kv_valid_mask and kv_write_positions using tensor operations
        positions = torch.arange(cache_len, device=device).unsqueeze(0)
        kv_valid_mask = positions < (decoding_start.unsqueeze(1) + block_length)
        offsets = torch.arange(block_length, device=device).unsqueeze(0)
        kv_write_positions = decoding_start.unsqueeze(1) + offsets

        # Pad to max_batch_size
        pad_size = self._max_batch_size - batch_size
        self._static_tensors["kv_valid_mask"].set(
            torch.cat([kv_valid_mask, torch.zeros(pad_size, cache_len, dtype=torch.bool, device=device)], dim=0))
        self._static_tensors["kv_write_positions"].set(
            torch.cat([kv_write_positions, torch.zeros(pad_size, block_length, dtype=torch.long, device=device)], dim=0))

        # Check if refresh needed
        prev_max_ds = getattr(self, '_prev_max_ds', -1)
        need_refresh = (max_ds != prev_max_ds)
        self._prev_max_ds = max_ds

        if not need_refresh or not has_historical_kv:
            if need_refresh:
                full_k.zero_()
                full_v.zero_()
            return

        full_k.zero_()
        full_v.zero_()

        # Build position_ids and seq_ids for historical KV
        total_past = decoding_start.sum().item()
        if total_past == 0:
            return

        position_ids = torch.zeros(total_past, dtype=torch.long, device=device)
        seq_ids = torch.zeros(total_past, dtype=torch.long, device=device)
        offset = 0
        for i, ds in enumerate(decoding_start.tolist()):
            if ds > 0:
                position_ids[offset:offset + ds] = torch.arange(ds, device=device)
                seq_ids[offset:offset + ds] = i
                offset += ds

        # Read and scatter historical KV from paged cache
        main_cache = self._cache_dict.get("main")
        if main_cache is not None:
            for layer_id in range(self._num_layers):
                accessor = main_cache.get_accessor(layer_id)

                past_k = read_from_paged_kv_cache(accessor.k, accessor.block_table, position_ids, seq_ids)
                past_v = read_from_paged_kv_cache(accessor.v, accessor.block_table, position_ids, seq_ids)
                full_k[layer_id, seq_ids, :, position_ids, :] = past_k
                full_v[layer_id, seq_ids, :, position_ids, :] = past_v

    def _decode_attention_graph_safe(
        self,
        q: torch.Tensor,
        kv_cache,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        layer_id: int,
    ) -> torch.Tensor:
        block_length = self._block_length or seq_len_delta.delta_max_len
        batch_size = self._batch_size or seq_len_delta.batch_size
        n_heads, n_kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[2]

        # Store K/V for write_finished_kv_cache
        if self._decode_kv_cache is not None:
            kv_size = batch_size * block_length
            self._decode_kv_cache[layer_id, 0, :kv_size].copy_(k)
            self._decode_kv_cache[layer_id, 1, :kv_size].copy_(v)

        # Get full KV tensors and scatter current K/V
        full_k = self._static_tensors["full_k_per_layer"].get()[layer_id, :batch_size]
        full_v = self._static_tensors["full_v_per_layer"].get()[layer_id, :batch_size]
        k_t = k.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)
        v_t = v.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)
        kv_write_positions = self._static_tensors["kv_write_positions"].get()[:batch_size, :block_length]

        # Scatter current K/V to full tensors
        write_idx = kv_write_positions.unsqueeze(1).unsqueeze(2).expand(-1, n_kv_heads, head_dim, -1)
        full_k.permute(0, 1, 3, 2).scatter_(dim=3, index=write_idx, src=k_t.permute(0, 1, 3, 2))
        full_v.permute(0, 1, 3, 2).scatter_(dim=3, index=write_idx, src=v_t.permute(0, 1, 3, 2))

        # Handle GQA
        if n_heads != n_kv_heads:
            n_rep = n_heads // n_kv_heads
            full_k = full_k.repeat_interleave(n_rep, dim=1)
            full_v = full_v.repeat_interleave(n_rep, dim=1)

        q_4d = q.view(batch_size, block_length, n_heads, head_dim).transpose(1, 2)
        kv_valid_mask = self._static_tensors["kv_valid_mask"].get()[:batch_size]
        attn_mask = kv_valid_mask.unsqueeze(1).unsqueeze(2).to(q_4d.dtype).masked_fill(~kv_valid_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        output = F.scaled_dot_product_attention(q_4d, full_k, full_v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=1.0 / (head_dim ** 0.5))
        return output.transpose(1, 2).reshape(batch_size * block_length, n_heads, head_dim)

    def bidirectional_prefill(self, q, k, v, *, seq_len_delta, softmax_scale=None, window_size=(-1, -1), softcap=0.0):
        return self.prefill_ragged_qkvo(q, k, v, seq_len_delta=seq_len_delta, causal=False,
                                        softmax_scale=softmax_scale, window_size=window_size, softcap=softcap)

    def bidirectional_prefill_with_mask(self, q, k, v, *, attention_mask=None, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
        if attention_mask is not None and len(attention_mask.shape) == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=softmax_scale)

    @override
    def decode_dense_kv(self, q, kv_cache: DenseKVCacheAccessor, k=None, v=None, *, seq_len_delta, window_size=(-1, -1), softcap=0.0, softmax_scale=None, sinks=None, topk_indices=None):
        if topk_indices is not None:
            raise NotImplementedError("topk_indices not supported in DLLMAttnBackend")
        if q.numel() == 0:
            return torch.empty(0, q.shape[1], kv_cache.v.shape[-1], device=q.device, dtype=q.dtype)

        extra_kvargs = {"softcap": softcap} if softcap != 0.0 else {}
        bsz = seq_len_delta.batch_size
        s_q = 1 if seq_len_delta.is_classic_decoding else getattr(self, "mtp_size", 1)

        output = self._fa.flash_attn_with_kvcache(
            q.view(bsz, s_q, q.shape[-2], q.shape[-1]), kv_cache.k, kv_cache.v,
            k=k.view(bsz, s_q, k.shape[-2], k.shape[-1]) if k is not None else None,
            v=v.view(bsz, s_q, v.shape[-2], v.shape[-1]) if v is not None else None,
            cache_seqlens=seq_len_delta.old.lens_tensor_device, causal=False,
            window_size=window_size, softmax_scale=softmax_scale, **extra_kvargs)
        return output.view(bsz * s_q, output.shape[-2], output.shape[-1])

    def decode_bidirectional_simple(self, q, k, v, *, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
        if has_flash_attn:
            return flash_attn.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               causal=False, softmax_scale=softmax_scale).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=softmax_scale)

    @override
    def __call__(self, q, kv_cache, k, v, *, seq_len_delta, causal=False, layer_id=0, attention_mask=None, **kwargs):
        if self._is_prefill:
            return self._prefill_attention(q, kv_cache, k, v, seq_len_delta=seq_len_delta, attention_mask=attention_mask, layer_id=layer_id)
        if self._use_cuda_graph and self._static_tensors:
            return self._decode_attention_graph_safe(q, kv_cache, k, v, seq_len_delta=seq_len_delta, layer_id=layer_id)
        return self._decode_attention(q, kv_cache, k, v, seq_len_delta=seq_len_delta, layer_id=layer_id)

    def _prefill_attention(self, q, kv_cache, k, v, *, seq_len_delta, attention_mask, layer_id):
        if attention_mask is not None:
            batch_size = seq_len_delta.batch_size
            attn_mask = attention_mask.unsqueeze(1) if len(attention_mask.shape) == 3 else attention_mask
            _, seq_q, seq_k = attention_mask.shape[-3:]
            n_heads, n_kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[2]

            # Scatter ragged to dense
            q_dense = torch.zeros(batch_size, n_heads, seq_q, head_dim, device=q.device, dtype=q.dtype)
            k_dense = torch.zeros(batch_size, n_kv_heads, seq_k, head_dim, device=k.device, dtype=k.dtype)
            v_dense = torch.zeros(batch_size, n_kv_heads, seq_k, head_dim, device=v.device, dtype=v.dtype)
            for i in range(batch_size):
                start, end = i * seq_q, (i + 1) * seq_q
                q_dense[i], k_dense[i], v_dense[i] = q[start:end].transpose(0, 1), k[start:end].transpose(0, 1), v[start:end].transpose(0, 1)

            if n_heads != n_kv_heads:
                n_rep = n_heads // n_kv_heads
                k_dense = k_dense.repeat_interleave(n_rep, dim=1)
                v_dense = v_dense.repeat_interleave(n_rep, dim=1)

            output = self.bidirectional_prefill_with_mask(q_dense, k_dense, v_dense, attention_mask=attn_mask)
            output = output.transpose(1, 2).reshape(-1, n_heads, head_dim)

            if isinstance(kv_cache, PagedKVCacheAccessor):
                for tensor in [k, v]:
                    append_to_paged_kv_cache(
                        kv_cache.k if tensor is k else kv_cache.v, kv_cache.block_table, tensor.contiguous(),
                        seq_len_delta.delta_position_ids_tensor_device, seq_len_delta.delta_seq_ids_tensor_device,
                        get_page_ids=kv_cache.get_page_ids, get_offs_in_page=kv_cache.get_offs_in_page, use_i64_offsets=kv_cache.use_i64_offsets)
            return output

        # No attention mask - standard ragged prefill
        if isinstance(kv_cache, PagedKVCacheAccessor):
            for tensor in [k, v]:
                if tensor is not None:
                    append_to_paged_kv_cache(
                        kv_cache.k if tensor is k else kv_cache.v, kv_cache.block_table, tensor.contiguous(),
                        seq_len_delta.delta_position_ids_tensor_device, seq_len_delta.delta_seq_ids_tensor_device,
                        get_page_ids=kv_cache.get_page_ids, get_offs_in_page=kv_cache.get_offs_in_page, use_i64_offsets=kv_cache.use_i64_offsets)
        return self.prefill_ragged_qkvo(q, k, v, seq_len_delta=seq_len_delta, causal=False)

    def _decode_attention(self, q, kv_cache, k, v, *, seq_len_delta, layer_id):
        if self._decode_kv_cache is not None:
            self._decode_kv_cache[layer_id, 0].copy_(k)
            self._decode_kv_cache[layer_id, 1].copy_(v)

        block_length = self._block_length or seq_len_delta.delta_max_len
        batch_size = self._batch_size or seq_len_delta.batch_size
        decoding_start = self._decoding_start
        n_heads, n_kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[2]
        device, dtype = q.device, q.dtype
        scale = 1.0 / (head_dim ** 0.5)

        # Helper for GQA and output
        def compute_output(q_4d, k_4d, v_4d):
            if n_heads != n_kv_heads:
                n_rep = n_heads // n_kv_heads
                k_4d = k_4d.repeat_interleave(n_rep, dim=1)
                v_4d = v_4d.repeat_interleave(n_rep, dim=1)
            out = self.decode_bidirectional_simple(q_4d, k_4d, v_4d, softmax_scale=scale)
            return out.transpose(1, 2).reshape(batch_size * block_length, n_heads, head_dim)

        q_4d = q.view(batch_size, block_length, n_heads, head_dim).transpose(1, 2)
        k_4d = k.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)
        v_4d = v.view(batch_size, block_length, n_kv_heads, head_dim).transpose(1, 2)

        if decoding_start is None:
            return compute_output(q_4d, k_4d, v_4d)

        max_ds = decoding_start.max().item()
        if not isinstance(kv_cache, PagedKVCacheAccessor) or max_ds == 0:
            return compute_output(q_4d, k_4d, v_4d)

        # Read historical KV
        total_past = decoding_start.sum().item()
        if total_past == 0:
            return compute_output(q_4d, k_4d, v_4d)

        position_ids = torch.zeros(total_past, dtype=torch.long, device=device)
        seq_ids = torch.zeros(total_past, dtype=torch.long, device=device)
        offset = 0
        for i, ds in enumerate(decoding_start.tolist()):
            if ds > 0:
                position_ids[offset:offset + ds] = torch.arange(ds, device=device)
                seq_ids[offset:offset + ds] = i
                offset += ds

        past_k = read_from_paged_kv_cache(kv_cache.k, kv_cache.block_table, position_ids, seq_ids)
        past_v = read_from_paged_kv_cache(kv_cache.v, kv_cache.block_table, position_ids, seq_ids)

        cache_len = max(128, 1 << max(0, (max_ds + block_length - 1)).bit_length())
        past_k_dense = torch.zeros(batch_size, n_kv_heads, cache_len, head_dim, device=device, dtype=dtype)
        past_v_dense = torch.zeros(batch_size, n_kv_heads, cache_len, head_dim, device=device, dtype=dtype)
        past_k_dense[:, :, :max_ds].permute(0, 2, 1, 3)[seq_ids, position_ids] = past_k
        past_v_dense[:, :, :max_ds].permute(0, 2, 1, 3)[seq_ids, position_ids] = past_v

        full_k = torch.cat([past_k_dense, k_4d], dim=2)
        full_v = torch.cat([past_v_dense, v_4d], dim=2)
        return compute_output(q_4d, full_k, full_v)

    def write_finished_kv_cache(self, block_finished: torch.Tensor, batch_size: int):
        if not block_finished.any() or self._decode_kv_cache is None:
            return

        finished_indices = block_finished.nonzero(as_tuple=True)[0].tolist()
        if not finished_indices:
            return

        device = self._decode_kv_cache.device
        n_kv_heads, head_dim = self._kv_heads, self._head_dim
        block_length = self._block_length
        num_finished = len(finished_indices)

        delta_position_ids = torch.zeros(num_finished * block_length, device=device, dtype=torch.long)
        delta_seq_ids = torch.zeros(num_finished * block_length, device=device, dtype=torch.long)
        for i, idx in enumerate(finished_indices):
            ds = self._decoding_start[idx].item()
            start, end = i * block_length, (i + 1) * block_length
            delta_position_ids[start:end] = torch.arange(ds, ds + block_length, device=device)
            delta_seq_ids[start:end] = idx

        kv_size = batch_size * block_length
        main_cache = self._cache_dict.get("main")
        if main_cache is None:
            return

        for layer_id in range(self._num_layers):
            layer_k = self._decode_kv_cache[layer_id, 0, :kv_size]
            layer_v = self._decode_kv_cache[layer_id, 1, :kv_size]

            accessor = main_cache.get_accessor(layer_id)

            finished_k = layer_k.view(batch_size, block_length, n_kv_heads, head_dim)[finished_indices]
            finished_v = layer_v.view(batch_size, block_length, n_kv_heads, head_dim)[finished_indices]
            for tensor, name in [(finished_k, 'k'), (finished_v, 'v')]:
                append_to_paged_kv_cache(
                    getattr(accessor, name), accessor.block_table, tensor.reshape(-1, n_kv_heads, head_dim).contiguous(),
                    delta_position_ids, delta_seq_ids,
                    get_page_ids=accessor.get_page_ids, get_offs_in_page=accessor.get_offs_in_page, use_i64_offsets=accessor.use_i64_offsets)
