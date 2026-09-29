# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Custom attention processors for diffusion models.

Adapted from diffusers attention processors to use alternative attention implementations.

Usage Example:
    ```python
    from diffusers.models.attention import Attention
    from healda.models.attention_processors import TEAttnProcessor

    # Create an attention module
    attn = Attention(
        query_dim=512,
        heads=8,
        dim_head=64,
    )

    # Set the processor
    processor = TEAttnProcessor()
    attn.set_processor(processor)

    # Use the attention module as normal
    hidden_states = torch.randn(2, 32, 512)
    output = attn(hidden_states)
    ```
"""

from typing import Optional

import torch
from diffusers.models.attention import Attention
from diffusers.utils import deprecate

try:
    from transformer_engine.pytorch.attention import DotProductAttention

    _HAS_TRANSFORMER_ENGINE = True
except (ImportError, OSError):
    _HAS_TRANSFORMER_ENGINE = False


def _set_transformer_engine_domain_parallel_group(attn, group):
    """This is needed to setup domain parallel group for Transformer Engine.

    The set_context_parallel_group does not work without explicitly setting the
    ranks and cp_stream, even though they are not required arguments in the
    function signature.

    Similar implementation as in Megatron-LM:
    https://github.com/NVIDIA/Megatron-LM/blob/5b1ef0703184299fbf71f6131bf2f9a5331e7238/megatron/core/extensions/transformer_engine.py#L1228C1-L1232C72

    """
    if group is None:
        return

    if attn.cp_stream is None:
        cp_stream = torch.cuda.Stream()
    else:
        cp_stream = attn.cp_stream

    ranks = torch.distributed.get_process_group_ranks(group)
    attn.set_context_parallel_group(group, ranks, cp_stream)


class TEAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention using Transformer Engine.

    This processor uses Transformer Engine's DotProductAttention module instead of PyTorch's
    scaled_dot_product_attention function. It provides a drop-in replacement for AttnProcessor2_0
    that can leverage Transformer Engine optimizations.

    The DotProductAttention module is initialized lazily on first use since it requires
    knowledge of the attention configuration (number of heads, head dimension).

    Example:
        ```python
        from diffusers.models.attention import Attention
        from healda.models.attention_processors import TEAttnProcessor

        # Create attention module
        attn = Attention(query_dim=512, heads=8, dim_head=64)

        # Apply TE processor
        attn.set_processor(TEAttnProcessor())

        # Use normally
        output = attn(hidden_states)
        ```

    Note:
        Requires transformer-engine to be installed. The processor automatically
        handles device placement and shape transformations to match Transformer Engine's
        expected input format (batch, seq_len, heads, head_dim).
    """

    def __init__(
        self, domain_parallel_group: Optional[torch.distributed.ProcessGroup] = None
    ):
        if not _HAS_TRANSFORMER_ENGINE:
            raise ImportError(
                "TEAttnProcessor requires Transformer Engine. "
                "Please install it with: pip install transformer-engine[pytorch]"
            )

        # Initialize Transformer Engine's DotProductAttention module lazily
        # since it requires num_attention_heads and kv_channels parameters
        self.attn_fn = None
        self._initialized_config = None
        self._domain_parallel_group = domain_parallel_group

    def _init_attention_module(self, attn: Attention):
        """Initialize the Transformer Engine DotProductAttention module lazily."""
        num_heads = attn.heads
        head_dim = attn.inner_dim // num_heads

        config_key = (num_heads, head_dim)

        # Only initialize if config has changed or not yet initialized
        if self._initialized_config != config_key:
            self.attn_fn = DotProductAttention(
                num_attention_heads=num_heads,
                kv_channels=head_dim,
                attention_dropout=0.0,
                qkv_format="bshd",  # batch, seqlen, heads, head_dim
                attn_mask_type="no_mask",  # We'll handle masking ourselves
            )
            _set_transformer_engine_domain_parallel_group(
                self.attn_fn, self._domain_parallel_group
            )
            self._initialized_config = config_key

            # Move to the same device as the attention module
            if hasattr(attn, "to_q") and hasattr(attn.to_q, "weight"):
                device = attn.to_q.weight.device
                self.attn_fn = self.attn_fn.to(device)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        # Initialize attention module if needed
        self._init_attention_module(attn)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # Transformer Engine expects attention_mask shape to be
            # (batch, heads, source_length, target_length) - same as PyTorch
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to (batch, seq_len, heads, head_dim) - the format expected by Transformer Engine
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Use Transformer Engine's DotProductAttention
        # Input shape: (batch, seq_len, heads, head_dim)
        # Output shape: (batch, seq_len, heads, head_dim)
        hidden_states = self.attn_fn(
            query,
            key,
            value,
            attention_mask=attention_mask,
            qkv_format="bshd",  # batch, seqlen, heads, head_dim
            attn_mask_type="arbitrary" if attention_mask is not None else "no_mask",
        )

        # Reshape from (batch, seq_len, heads, head_dim) to (batch, seq_len, heads * head_dim)
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
