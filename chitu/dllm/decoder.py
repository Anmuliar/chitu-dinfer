# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
dLLM Decoder Module - Independent implementation within chitu framework.

This module implements parallel decoding strategy based on confidence threshold
for Diffusion Large Language Models (dLLM) like LLaDA series.

Key features:
- Iterative parallel decoding until all mask tokens are decoded
- Gumbel noise for sampling diversity
- Confidence-based token selection
"""

import math
from logging import getLogger
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = getLogger(__name__)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise to logits for sampling.

    Gumbel noise helps introduce controlled randomness in the sampling process,
    allowing for more diverse outputs while maintaining quality.

    Args:
        logits: Input logits, shape [..., vocab_size]
        temperature: Noise temperature parameter.
                    - 0.0 means no noise (greedy)
                    - Higher values introduce more randomness

    Returns:
        Logits with Gumbel noise added, same shape as input
    """
    if math.isclose(temperature, 0.0):
        return logits

    # Use float64 for numerical stability
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    # Gumbel distribution: -log(-log(U)) where U ~ Uniform(0,1)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class DLLMDecoder:
    """dLLM Parallel Decoder.

    Implements parallel decoding strategy for dLLM models based on confidence threshold:
    - Tokens with prediction confidence >= threshold are decoded
    - Supports iterative decoding until all mask tokens are resolved
    - Handles batch decoding with different block positions

    The decoding process for dLLM is fundamentally different from autoregressive LLM:
    1. Multiple mask tokens are decoded in parallel within each iteration
    2. High-confidence tokens are decoded first, low-confidence ones wait
    3. Process repeats until all masks are replaced with actual tokens

    Args:
        temperature: Gumbel noise temperature (0.0 for greedy, >0 for sampling)
        threshold: Confidence threshold for decoding (0.0-1.0)
        mask_id: Token ID for mask token
        eos_id: Token ID for end-of-sequence token
    """

    def __init__(
        self,
        temperature: float = 0.0,
        threshold: float = 0.9,
        mask_id: int = 126336,
        eos_id: int = 126081,
    ):
        self.temperature = temperature
        self.threshold = threshold
        self.mask_id = mask_id
        self.eos_id = eos_id

    def decode(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Decode mask tokens within a single block.

        This method processes one block of tokens and updates mask positions
        with predicted tokens if their confidence meets the threshold.

        Args:
            logits: Prediction logits, shape [batch_size, block_length, vocab_size]
            tokens: Current token sequence, shape [batch_size, total_length]
            block_start: Start position of the block in token sequence
            block_end: End position of the block in token sequence

        Returns:
            Updated token sequence with some mask tokens potentially decoded
        """
        # Get current block tokens and identify mask positions
        block_tokens = tokens[:, block_start:block_end]
        mask_index = (block_tokens == self.mask_id)

        # Early return if no masks to decode
        if not mask_index.any():
            return tokens

        # Add Gumbel noise for sampling diversity
        noisy_logits = add_gumbel_noise(logits, self.temperature)

        # Get predicted tokens via argmax
        predicted_tokens = torch.argmax(noisy_logits, dim=-1)

        # Calculate confidence scores using softmax probabilities
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        confidence = probs.gather(
            dim=-1,
            index=predicted_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Set non-mask positions to negative infinity (won't be decoded)
        confidence = torch.where(
            mask_index,
            confidence,
            torch.full_like(confidence, float('-inf'))
        )

        # Determine which positions to decode:
        # Either confidence >= max_confidence - epsilon, or >= threshold
        max_confidence = confidence.max(dim=-1, keepdim=True)[0]
        actual_threshold = torch.clamp(max_confidence - 1e-5, max=self.threshold)
        transfer_index = (confidence >= actual_threshold) & mask_index

        # Update tokens at decoded positions
        new_block_tokens = torch.where(transfer_index, predicted_tokens, block_tokens)
        tokens[:, block_start:block_end] = new_block_tokens

        return tokens

    def batch_decode(
        self,
        logits: torch.Tensor,
        block_starts: torch.Tensor,
        token_array,
        block_length: int,
    ) -> None:
        """Batch decode multiple blocks with different starting positions.

        This method signature matches the executor's calling convention:
        decoder.batch_decode(logits, decoding_start_t, x, block_length)

        The token_array is expected to have a .data attribute containing
        the token tensor, which is updated in-place.

        Args:
            logits: Prediction logits, shape [batch_size, block_length, vocab_size]
            block_starts: Starting position for each block, shape [batch_size]
            token_array: Object with .data attribute containing tokens,
                        shape [batch_size, total_length]
            block_length: Length of each block

        Note:
            This method modifies token_array.data in-place.
        """
        # Handle both direct tensor and object with .data attribute
        if hasattr(token_array, 'data'):
            tokens = token_array.data
        else:
            tokens = token_array

        batch_size, total_length = tokens.shape
        device = tokens.device

        # Compute absolute position indices for each block
        offsets = torch.arange(block_length, device=device).unsqueeze(0)
        indices = block_starts.unsqueeze(1) + offsets  # [batch_size, block_length]

        # Gather block tokens from each sequence
        block_tokens = torch.gather(
            tokens, dim=1, index=indices.clamp(max=total_length - 1)
        )
        mask_index = (block_tokens == self.mask_id)

        # Early return if no masks
        if not mask_index.any():
            return

        # Decode logic same as single block
        noisy_logits = add_gumbel_noise(logits, self.temperature)
        predicted_tokens = torch.argmax(noisy_logits, dim=-1)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        confidence = probs.gather(
            dim=-1, index=predicted_tokens.unsqueeze(-1)
        ).squeeze(-1)
        confidence = torch.where(
            mask_index,
            confidence,
            torch.full_like(confidence, float('-inf'))
        )

        max_confidence = confidence.max(dim=-1, keepdim=True)[0]
        actual_threshold = torch.clamp(max_confidence - 1e-5, max=self.threshold)
        transfer_index = (confidence >= actual_threshold) & mask_index

        new_block_tokens = torch.where(transfer_index, predicted_tokens, block_tokens)

        # Scatter updated tokens back to original positions
        tokens.scatter_(dim=1, index=indices, src=new_block_tokens)

    def has_mask(
        self,
        tokens: torch.Tensor,
        block_start: int,
        block_end: int
    ) -> bool:
        """Check if there are remaining mask tokens in a block.

        Args:
            tokens: Token sequence, shape [batch_size, total_length]
            block_start: Start position of block
            block_end: End position of block

        Returns:
            True if any mask tokens remain, False otherwise
        """
        return (tokens[:, block_start:block_end] == self.mask_id).any().item()

    def batch_has_mask(
        self,
        tokens: torch.Tensor,
        block_starts: torch.Tensor,
        block_length: int,
    ) -> torch.Tensor:
        """Check for remaining mask tokens in batch of blocks.

        Args:
            tokens: Token sequences, shape [batch_size, total_length]
            block_starts: Starting positions, shape [batch_size]
            block_length: Block length

        Returns:
            Boolean tensor, shape [batch_size], True if masks remain
        """
        batch_size, total_length = tokens.shape
        device = tokens.device

        offsets = torch.arange(block_length, device=device).unsqueeze(0)
        indices = block_starts.unsqueeze(1) + offsets

        block_tokens = torch.gather(
            tokens, dim=1, index=indices.clamp(max=total_length - 1)
        )

        return (block_tokens == self.mask_id).any(dim=-1)

    def count_masks(
        self,
        tokens: torch.Tensor,
        block_start: int,
        block_end: int
    ) -> int:
        """Count remaining mask tokens in a block.

        Args:
            tokens: Token sequence, shape [batch_size, total_length]
            block_start: Start position of block
            block_end: End position of block

        Returns:
            Number of remaining mask tokens
        """
        return (tokens[:, block_start:block_end] == self.mask_id).sum().item()

    def get_mask_positions(
        self,
        tokens: torch.Tensor,
        block_start: int,
        block_end: int
    ) -> torch.Tensor:
        """Get positions of mask tokens in a block.

        Args:
            tokens: Token sequence, shape [batch_size, total_length]
            block_start: Start position of block
            block_end: End position of block

        Returns:
            Boolean mask of shape [batch_size, block_length], True at mask positions
        """
        return tokens[:, block_start:block_end] == self.mask_id
