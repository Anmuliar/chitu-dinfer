# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""dLLM (Diffusion Large Language Model) inference module.

This module provides independent implementation of dLLM inference within chitu framework.
"""

from chitu.dllm.decoder import DLLMDecoder, add_gumbel_noise
from chitu.dllm.utils import TokenArray

__all__ = ["DLLMDecoder", "add_gumbel_noise"]
