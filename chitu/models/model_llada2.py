# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
LLaDA V2 Model Implementation - Inherits from TransformerHFLlama

This module provides LLaDA (Diffusion Large Language Model) support with:
- Bidirectional attention (non-causal)
- MoE (Mixture of Experts) support
- KV-Cache replacement mode for iterative decoding
- Tensor Parallelism support

Checkpoint structure:
- word_embeddings.weight: Embedding
- lm_head.weight: Output projection
- norm.weight: Final layer norm
- layers.{i}.input_layernorm.weight
- layers.{i}.post_attention_layernorm.weight
- layers.{i}.attention.query_key_value.weight: Merged QKV projection
- layers.{i}.attention.query_layernorm.weight: LayerNorm for Q
- layers.{i}.attention.key_layernorm.weight: LayerNorm for K
- layers.{i}.attention.dense.weight: Output projection
- layers.{i}.mlp.gate_proj.weight / up_proj.weight / down_proj.weight (Dense)
- layers.{i}.mlp.experts.{e}.gate_proj.weight / up_proj.weight / down_proj.weight (MoE)
"""

import functools
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cache_manager import KVCacheManagerBase
from chitu.dllm.decoder import DLLMDecoder
from chitu.models.model import MoeGate, ParallelMoeBlock, RMSNorm, TransformerBlock, get_linear_layout_contig_y
from chitu.models.model_hf_llama import (
    AttentionHFLlama,
    FeedForwardHFLlama,
    TransformerBlockHFLlama,
    TransformerHFLlama,
    get_rms_norm_impl,
)
from chitu.models.registry import ModelType, register_model
from chitu.moe import get_moe_impl, MoEImplBase, MoEImplEP
from chitu.ops import apply_rotary_pos_emb, silu_and_mul
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from chitu.distributed.parallel_state import (
    get_ep_size,
    get_ep_group,
    get_tp_size,
    get_tp_group,
    get_etp_size,
    get_etp_group,
)

from chitu.distributed.partition import compute_expert_dist_in_ep
from chitu.utils import parse_dtype

logger = getLogger(__name__)


@dataclass
class DLLMModelOutput:
    """Output from dLLM forward pass."""
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


class DLLMModelRunner:
    """Wrapper class for dLLM model execution."""

    def __init__(self, model: "TransformerLLaDA2"):
        self.model = model
        self.config = type('Config', (), {
            'num_hidden_layers': len(model.layers),
        })()

    def __call__(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = True,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,  # [num_layers, 2, batch, heads, seq, head_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DLLMModelOutput:
        """Forward pass for dLLM decode step.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            use_cache: Whether to return past_key_values
            position_ids: Position IDs for RoPE
            past_key_values: KV cache tensor [num_layers, 2, batch, heads, seq, head_dim] from previous steps
            attention_mask: Attention mask for block-diagonal attention [batch, seq, seq]
        """
        batch_size, block_length = input_ids.shape
        device = input_ids.device

        # Get freqs_cis for RoPE
        # position_ids: [batch_size, block_length] - each batch element has its own positions
        # We need to flatten to [batch_size * block_length] for indexing the cos/sin cache
        if position_ids is not None:
            # Flatten position_ids to get position for each token
            pos_ids_flat = position_ids.reshape(-1)  # [batch_size * block_length]
        else:
            # Default: sequential positions starting from 0
            pos_ids_flat = torch.arange(batch_size * block_length, device=device)

        # Index into the RoPE cache - each token gets its correct position embedding
        cos_single = self.model.rotary_emb.cos_cached[pos_ids_flat]  # [batch_size * block_length, head_dim]
        sin_single = self.model.rotary_emb.sin_cached[pos_ids_flat]  # [batch_size * block_length, head_dim]
        freqs_cis = BatchedFreqsCis(cos_single, sin_single)

        # Embedding
        h = self.model.embed_tokens(input_ids)
        h = h.view(batch_size * block_length, -1)

        new_past_key_values = []
        for layer_idx, layer in enumerate(self.model.layers):
            # Attention
            normed = layer.input_layernorm(h, impl=get_rms_norm_impl())
            
            xq, xk, xv = layer.attention._run_linear(normed)

            # Get dimensions
            n_local_heads = layer.attention.n_local_heads
            n_local_kv_heads = layer.attention.n_local_kv_heads
            head_dim = layer.attention.head_dim

            # QK LayerNorm - apply before reshape, using 2D tensor
            # dInfer approach: q_by_head = q.reshape(-1, self.head_dim)
            # This ensures the tensor is contiguous for Triton RMSNorm
            if hasattr(layer.attention, "query_layernorm"):
                # xq is [batch * seq, heads * head_dim], reshape to [batch * seq * heads, head_dim]
                # Use reshape() instead of view() since xq from split() may not be contiguous
                xq_flat = xq.reshape(-1, head_dim)
                xq_flat = layer.attention.query_layernorm(xq_flat)
                 # layernorm output is contiguous, so view() is safe
                xq = xq_flat.view(-1, n_local_heads * head_dim)

            if hasattr(layer.attention, "key_layernorm"):
                # xk is [batch * seq, kv_heads * head_dim], reshape to [batch * seq * kv_heads, head_dim]
                # Use reshape() instead of view() since xk from split() may not be contiguous
                xk_flat = xk.reshape(-1, head_dim)
                xk_flat = layer.attention.key_layernorm(xk_flat)

                # layernorm output is contiguous, so view() is safe
                xk = xk_flat.view(-1, n_local_kv_heads * head_dim)

            # Now reshape to [batch, seq, heads, head_dim] for RoPE
            xq = xq.view(batch_size, -1, n_local_heads, head_dim)
            xk = xk.view(batch_size, -1, n_local_kv_heads, head_dim)
            xv = xv.view(batch_size, -1, n_local_kv_heads, head_dim)

            # Apply RoPE (expects [batch, seq, heads, head_dim] or [seq*batch, heads, head_dim])
            # Need to transpose to [batch*seq, heads, head_dim] for apply_rotary_pos_emb
            xq_flat = xq.view(-1, n_local_heads, head_dim)
            xk_flat = xk.view(-1, n_local_kv_heads, head_dim)
            
            xq_flat, xk_flat = apply_rotary_pos_emb(xq_flat, xk_flat, freqs_cis, rotary_type="separated-half")
            
            xq = xq_flat.view(batch_size, -1, n_local_heads, head_dim)
            xk = xk_flat.view(batch_size, -1, n_local_kv_heads, head_dim)

            # Reshape for cache: [batch, heads, seq, head_dim]
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # KV Cache replacement mode for dLLM iterative decoding
            # dInfer uses slice_scatter to replace KV at specific positions
            # CRITICAL: Replace the last block_length positions in the cache                     
            # This is how dLLM works - iterative refinement replaces the last block              
            # past_key_values shape: [num_layers, 2, batch_size, num_kv_heads, seq_len, head_dim]
            if past_key_values is not None:                                                      
                cache_k = past_key_values[layer_idx, 0]  # [batch, heads, seq, head_dim]         
                cache_v = past_key_values[layer_idx, 1]  # [batch, heads, seq, head_dim]         
                cache_length = cache_k.shape[2]  # Total cache length                            
                block_length = xk.shape[2]       # Current block length                                    

                # Use slice_scatter to replace the last block_length positions
                # This matches dInfer's behavior                                                     
                k = cache_k.slice_scatter(xk, dim=2, start=cache_length - block_length, end=cache_length)
                v = cache_v.slice_scatter(xv, dim=2, start=cache_length - block_length, end=cache_length)
            else:
                k = xk
                v = xv

            new_past_key_values.extend((k, v))

            # 1. Reshape Q for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
            xq = xq.transpose(1, 2)

            # 2. Handle GQA: repeat K and V if needed
            if n_local_heads != n_local_kv_heads:
                n_rep = n_local_heads // n_local_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)

            # 3. Attention with bidirectional (non-causal) mask
            # LLaDA is a bidirectional model, so is_causal=False
            # This is critical: LLaDA uses bidirectional attention, not causal!
            # Process attention_mask if provided (for block-diagonal attention)
            current_attn_mask = None
            if attention_mask is not None:
                # attention_mask shape: [batch, seq, seq] -> need [batch, 1, seq, seq] for SDPA
                if len(attention_mask.shape) == 3:
                    current_attn_mask = attention_mask.unsqueeze(1)
                else:
                    current_attn_mask = attention_mask
            scale = 1.0 / (head_dim ** 0.5)

            attn_output = F.scaled_dot_product_attention(
                xq, k, v,
                attn_mask=current_attn_mask,
                dropout_p=0.0,
                is_causal=False,  # Bidirectional attention for dLLM
                scale=scale,
            )

            # 4. Reshape output and apply output projection
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * block_length, -1)
            
            attn_output = layer.attention._run_output_linear(attn_output)
            
            # Residual
            h = h + attn_output
            # FFN/MoE
            mlp_out = layer.mlp(layer.post_attention_layernorm(h, impl=get_rms_norm_impl()))

            
            h = h + mlp_out

        # Output projection
        h = h.view(batch_size * block_length, -1)
        logits = self.model.lm_head(self.model.norm(h, impl=get_rms_norm_impl()))
        # Reshape to [batch_size, block_length, vocab_size] for 3D indexing in executor

        logits = logits.view(batch_size, block_length, -1)

        return DLLMModelOutput(logits=logits, past_key_values=new_past_key_values)


class AttentionLLaDA2(AttentionHFLlama):
    """LLaDA Attention with merged QKV and QK LayerNorm.

    Key differences from standard HFLlama attention:
    - Uses `query_key_value` instead of `qkv_proj` for merged QKV
    - Uses `dense` instead of `o_proj` for output projection
    - Has `query_layernorm` and `key_layernorm` for QK normalization
    """

    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="separated-half",
        op_impl: str = "torch",
        checkpoint_prefix="",
    ):
        # Don't call super().__init__ since we have different structure
        nn.Module.__init__(self)

        self.layer_id = layer_id
        self.cache = cache
        self.attn_backend = attn_backend
        self.rotary_type = rotary_type
        self.op_impl = op_impl

        # Handle GQA
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        tensor_parallel_size = get_tp_size()
        self.n_local_heads = args.n_heads // tensor_parallel_size

        if self.n_kv_heads >= tensor_parallel_size:
            self.n_local_kv_heads = self.n_kv_heads // tensor_parallel_size
            self.n_kv_head_multiplier = 1
        else:
            self.n_local_kv_heads = 1
            self.n_kv_head_multiplier = tensor_parallel_size // self.n_kv_heads

        self.head_dim = (
            args.head_dim if hasattr(args, "head_dim") else args.dim // args.n_heads
        )

        # Merged QKV projection (named query_key_value in checkpoint)
        self.query_key_value = ColumnParallelLinear(
            args.dim,
            (args.n_heads + 2 * self.n_kv_heads * self.n_kv_head_multiplier) * self.head_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.query_key_value",
        )

        # QK LayerNorm
        self.query_layernorm = RMSNorm(
            self.head_dim,
            eps=args.norm_eps,
            dtype=torch.float32,
        )
        self.key_layernorm = RMSNorm(
            self.head_dim,
            eps=args.norm_eps,
            dtype=torch.float32,
        )

        # Output projection (named dense in checkpoint)
        self.dense = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            checkpoint_prefix=f"{checkpoint_prefix}.dense",
        )

    def _run_linear(self, x):
        """Split merged QKV into Q, K, V."""
        qkv = self.query_key_value(x)
        q, k, v = qkv.split(
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        return q, k, v

    def _run_output_linear(self, x):
        """Output projection."""
        return self.dense(x)


class LLaDA2MoeGate(MoeGate):
    """MoE Gate for LLaDA2"""

    def __init__(self, params, op_impl: str):
        super().__init__(
            op_impl,
            params.dim,
            topk=getattr(params, "num_experts_per_tok", 1),
            n_groups=params.n_group,
            topk_groups=params.topk_group,
            topk_as_topk_group_criteria=2,
            score_func="sigmoid",
            route_scale=params.routed_scaling_factor,
            n_experts=params.num_experts,
            bias=None,
            e_score_correction_bias=None,
            norm_prob=getattr(params, "norm_topk_prob", False),
            n_fused_shared_experts=0,
        )
        self.n_group = params.n_group
        self.routed_scaling_factor = params.routed_scaling_factor
        self.top_k = params.num_experts_per_tok
        self.topk_group = params.topk_group
        self.num_experts = params.num_experts
        self.expert_bias = nn.Parameter(torch.zeros(params.num_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, hidden_states):
        self.e_score_correction_bias = self.expert_bias
        return super().forward(hidden_states)
        ...
        

    # def group_limited_topk(
    #     self,
    #     scores: torch.Tensor,
    # ):
    #     num_tokens, _ = scores.size()
    #     # Organize the experts into groups
    #     group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    #     group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
    #     group_mask = torch.zeros_like(group_scores)
    #     group_mask.scatter_(1, group_idx, 1)

    #     # Mask the experts based on selection groups
    #     score_mask = (
    #         group_mask.unsqueeze(-1)
    #         .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
    #         .reshape(num_tokens, -1)
    #     )

    #     masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    #     probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

    #     return probs, top_indices
    
    # def forward(self, hidden_states):
    #     # compute gating score
    #     hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    #     logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

    #     scores = torch.sigmoid(logits.float()).type_as(logits)

    #     scores_for_routing = scores + self.expert_bias
    #     _, topk_idx = self.group_limited_topk(scores_for_routing)

    #     scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

    #     topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
    #     # topk_weight = topk_weight * self.routed_scaling_factor

    #     return topk_weight, topk_idx

class MLPLLaDA2(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(
        self,
        args,
        role: str,  # "standalone" or "shared_experts"
        op_impl: str,
        checkpoint_prefix: str,
        merge_gate_up=None,  # only work when role is "shared_experts"
        layer_id: int = 0,
    ):
        super().__init__()
        if role == "shared_experts":
            assert merge_gate_up is not None
            self.merge_gate_up = merge_gate_up
        else:
            self.merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
                checkpoint_prefix
            )

        self.op_impl = op_impl

        if role == "standalone":
            inter_dim = args.intermediate_dim
        elif role == "shared_experts":
            inter_dim = args.moe_intermediate_dim
        else:
            raise ValueError(
                f"Invalid role: {role}. Expected 'standalone' or 'shared_experts'."
            )

        if self.merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim * 2,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    quant_kwargs={
                        "blockfp4": {
                            "block_shape_2": (args.dim, inter_dim // get_tp_size())
                        }
                    },
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                # FIXME: f"{checkpoint_prefix}.gate_up_proj" is not a real checkpoint prefix,
                # implement a joint checkpoint prefix for gate_proj and up_proj.
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )
            self.up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )
        self.down_proj = RowParallelLinear(
            inter_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            reduce_output=(role == "standalone"),
            base_linear_class=get_linear_layout_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        if self.merge_gate_up:
            gate_up_proj_out = self.gate_up_proj(x)
            return self.down_proj(silu_and_mul(gate_up_proj_out))
        else:
            gate_proj_out = self.gate_proj(x)
            up_proj_out = self.up_proj(x)
            return self.down_proj(F.silu(gate_proj_out) * up_proj_out)

def LLaDA2MoeExperts(
    args,
    global_n_experts: int,
    experts_start_idx: int,
    experts_end_idx: int,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    *,
    checkpoint_prefix: str,
):
    """Create MoE experts for LLaDA2."""
    quant = get_quant_from_checkpoint_prefix(checkpoint_prefix, args.quant_config.rules)
    merge_gate_up = quant in QuantizationRegistry._allowed_quant_for_merge_gate_up

    if base_moe_experts_class is None:
        base_moe_experts_class = (
            QuantizationRegistry.get_quantized_moe_experts_class_from_global_args(
                merge_gate_up=merge_gate_up,
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=checkpoint_prefix,
            )
        )

    assert args.moe_intermediate_dim % get_etp_size() == 0
    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_intermediate_dim // get_etp_size(),
        global_n_experts=global_n_experts,
        experts_start_idx=experts_start_idx,
        experts_end_idx=experts_end_idx,
        n_shared_experts=0,
        n_activated_experts=0,
        fuse_shared_experts=False,
        checkpoint_prefix=checkpoint_prefix,
    )


class ParallelMoeBlockLLaDA2(ParallelMoeBlock):
    """MoE Block for LLaDA2."""

    def __init__(
        self,
        args,
        op_impl: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
        moe_impl: Optional[MoEImplBase] = None,
        *,
        checkpoint_prefix: str,
    ):
        if moe_impl is None:
            moe_impl = get_moe_impl()

        if isinstance(moe_impl, MoEImplEP):
            num_local_slots = moe_impl.load_balancer[layer_id].get_num_local_slots()
            experts_start_idx = moe_impl.ep_group.rank_in_group * num_local_slots
            experts_end_idx = experts_start_idx + num_local_slots
        else:
            experts_start_idx = 0
            experts_end_idx = args.num_experts

        merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
            checkpoint_prefix
        )

        non_fused_shared_experts = MLPLLaDA2(
            args,
            role="shared_experts",
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.shared_experts",
            merge_gate_up=merge_gate_up
        )

        super().__init__(
            gate=LLaDA2MoeGate(args, op_impl),
            experts=LLaDA2MoeExperts(
                args,
                args.num_experts,
                experts_start_idx,
                experts_end_idx,
                base_moe_experts_class,
                quant_kwargs,
                checkpoint_prefix=f"{checkpoint_prefix}.experts",
            ),
            non_fused_shared_experts=non_fused_shared_experts,
            layer_id=layer_id,
            moe_impl=moe_impl,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockLLaDA2(TransformerBlock):
    """LLaDA2 Transformer Block with support for Dense/MoE layers."""

    def __init__(
        self,
        layer_id: int,
        args,
        cache_managers: dict[str, KVCacheManagerBase],
        attn_backend,
        op_impl,
        rotary_type,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id, args, cache_managers, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        self.attention = AttentionLLaDA2(
            args,
            layer_id,
            cache_managers["main"],
            attn_backend,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.attention",
        )
        base_moe_experts_class = None
        self.mlp = (
            MLPLLaDA2(
                args,
                role="standalone",
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            )
            if layer_id < args.n_dense_layers
            else (
                ParallelMoeBlockLLaDA2(
                    args,
                    op_impl=op_impl,
                    base_moe_experts_class=base_moe_experts_class,
                    checkpoint_prefix=f"{checkpoint_prefix}.mlp",
                    layer_id=layer_id,
                )
            )
        )
        self.input_layernorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )
        self.post_attention_layernorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )

    @override
    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        x = x + self.attention(
            self.input_layernorm(x, compute_dtype=x.dtype), freqs_cis, is_mtp
        )
        x = x + self.mlp(self.post_attention_layernorm(x, compute_dtype=x.dtype))
        return x


@register_model(ModelType.LLADA2)
class TransformerLLaDA2(TransformerHFLlama):
    """LLaDA2 Model inheriting from TransformerHFLlama.

    Supports:
    - Bidirectional attention for dLLM
    - MoE layers
    - dLLM iterative decoding
    """

    def __init__(
        self,
        params,
        cache_managers: dict[str, KVCacheManagerBase],
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        rotary_type: str = "separated-half",
        layer_type: Optional[type] = None,
        layer_type_callback: Optional[Callable[[int], type]] = None,
        **kvargs,
    ):
        # dLLM-specific parameters
        self.mask_id = getattr(params, "mask_id", 156895)
        self.eos_id = getattr(params, "eos_id", 126081)
        self.block_length = getattr(params, "block_length", 32)
        self.threshold = getattr(params, "threshold", 0.9)
        self.temperature = getattr(params, "temperature", 0.0)

        # Set default layer type
        if layer_type is None and layer_type_callback is None:
            layer_type = TransformerBlockLLaDA2

        # Call parent initialization
        super().__init__(
            params,
            cache_managers,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type=layer_type,
            layer_type_callback=layer_type_callback,
            op_impl=op_impl,
            **kvargs,
        )

        # Initialize the dLLM decoder
        self.decoder = DLLMDecoder(
            temperature=self.temperature,
            threshold=self.threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
        )

        # Model runner for executor compatibility (lazy init)
        self._model_runner = None

        logger.info(
            f"TransformerLLaDAV2 initialized with mask_id={self.mask_id}, "
            f"block_length={self.block_length}, threshold={self.threshold}"
        )

    @property
    def model(self) -> "DLLMModelRunner":
        """Lazy initialization of model runner."""
        if self._model_runner is None:
            self._model_runner = DLLMModelRunner(self)
        return self._model_runner

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        return [
            "query_key_value",  # LLaDA2 uses this instead of qkv_proj
            "q_proj",  # Fallback
            "k_proj",
            "v_proj",
            "gate_up_proj",
            "gate_proj",
            "up_proj",
            "word_embeddings",
            "lm_head",
        ]

    @override
    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return [
            "dense",  # LLaDA2 uses this instead of o_proj
            "down_proj",
            "o_proj",  # Fallback
        ]

    @override
    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["word_embeddings."]

    @override
    def _get_post_layer_prefixes(self) -> list[str]:
        return ["lm_head.", "norm."]

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        """Mapping from checkpoint keys to model keys."""
        mappings = []
        if self.pp_stage == 0:
            mappings.append(("model.word_embeddings.", "embed_tokens."))
        if self.pp_stage == self.pp_end_stage:
            mappings.append(("model.norm.", "norm."))
            mappings.append(("lm_head.", "lm_head."))
        return mappings

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"model.layers.{i}.", f"layers.{i}.")

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        """Process checkpoint to merge expert weights for MoE layers."""
        if not hasattr(self.params, 'num_experts'):
            return checkpoint

        if not hasattr(self, 'moe_impl') or self.moe_impl is None:
            return checkpoint

        local_experts = compute_expert_dist_in_ep(
            self.args.models.n_layers,
            self.ep_size,
            self.params.num_experts,
            self.moe_impl,
        )[self.ep_group.rank_in_group]

        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            key_split = k.split(".")
            if key_split[0] != "layers":
                continue
            layer_id = int(key_split[1])

            # Check if this is an expert weight
            if any(
                k.endswith(f"{layer_id}.mlp.experts.{local_experts[layer_id][0]}.{w}.{part}")
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = f"layers.{layer_id}.mlp."
                parts = []
                for i in local_experts[layer_id]:
                    parts.append(prefix + f"experts.{i}.{w}.{part}")
                checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    [checkpoint.pop(key) for key in parts], dim=0
                )

        return checkpoint

    def _init_pre_layers(self):
        """Initialize embedding layer."""
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.params.vocab_size,
            embedding_dim=self.params.dim,
        )

    def _init_post_layers(self):
        """Initialize output layers."""
        self.norm = RMSNorm(
            self.params.dim,
            eps=self.params.norm_eps,
            dtype=parse_dtype(self.params.rms_norm_dtype) if hasattr(self.params, "rms_norm_dtype") else None,
        )
        self.lm_head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            checkpoint_prefix="lm_head",
        )

    def _pre_layers(self, h, **args):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        h = self.norm(h, impl=get_rms_norm_impl())
        h = self.lm_head(h)
        return h

    def get_decoder(self) -> DLLMDecoder:
        """Get the dLLM decoder instance."""
        return self.decoder
