# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers.models.attention import Attention

from healda.models.attention_processors import TEAttnProcessor


def create_attention_module(device, dim=512, num_heads=8, head_dim=64):
    """Create a standard Attention module for testing."""
    attn = Attention(
        query_dim=dim,
        heads=num_heads,
        dim_head=head_dim,
        dropout=0.0,
        bias=False,
    )
    return attn.to(device)


def test_te_attn_processor_forward_pass(device):
    """Test basic forward pass with TEAttnProcessor."""
    attn = create_attention_module(device)
    processor = TEAttnProcessor()
    attn.set_processor(processor)

    batch_size = 2
    seq_len = 32
    dim = 512

    hidden_states = torch.randn(batch_size, seq_len, dim, device=device)
    output = attn(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.device == hidden_states.device
