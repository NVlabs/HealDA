# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/dit_transformer_2d.py
"""
Adapted from the DiT code in the huggingface diffusers library

NVIDIA Modifications
- simplify code and remove unused layer normalizations
- pass emb directly to the AdaLayerNormZero
- incorporate the noise, class label, position, and calendar embeddings from the other cBottle code
- add fsdp support
"""

import dataclasses
from functools import partial
from typing import Any, Dict, Optional, Tuple

import earth2grid.healpix
import einops
import torch
from torch import nn
import torch.distributed.fsdp
import torch.utils.checkpoint
from diffusers.models.attention import Attention, FeedForward

import healda.utils.profiling
from healda.config.models import ModelSensorConfig, ObsConfig, SensorEmbedderConfig
from healda.observations.types import (
    UnifiedObservation,
    AttentionPacking,
    healpix_pixel_order,
)
from healda.varlen import lengths_to_idx
from healda.domain import HealPixDomain
from healda.models.sharding import shard_t, shard_x
from healda.models.embedding import EmbedNoiseLabels
from healda.models.healpix_layers import HPXPatchDecode, HPXPatchEmbed, Subdomain
from healda.models.obs_embedding.decoder import ObsDecoder
from healda.models.obs_embedding.point_embed_v2 import (
    MultiSensorObsEmbedding,
    ObsTokenizerFiLM,
)
from healda.models.obs_embedding.pixel_cross_attention import (
    PixelCrossAttention,
)
from healda.models.attention import (
    TEMPORAL_ATTN_EINSUM,
    TemporalAttention,
    SpatialAttention,
    SpatioTemporalAttention,
    TESpatialAttention,
    DiskSpatialAttention,
)


def normalize_attention_backend(attention_backend: str) -> str:
    """Checkpoints written before the rename spell the disk backend "flex-disk-*".

    Both ModelConfigV1 and TrainingLoop store the backend, so normalise on read rather
    than in either one's __post_init__: the two must not disagree about the token order.
    """
    if attention_backend.startswith("flex-disk"):
        return attention_backend.replace("flex-", "", 1)
    return attention_backend


def _spatial_attention_cls(attention_backend: str, pixel_order: str):
    attention_backend = normalize_attention_backend(attention_backend)
    if attention_backend == "te":
        return TESpatialAttention
    if attention_backend == "diffusers":
        return SpatialAttention
    # "disk-nest:10.46" -> 10.46 degree geodesic disk per query, NEST token order.
    if attention_backend.startswith("disk"):
        _, _, radius = attention_backend.partition(":")
        return partial(
            DiskSpatialAttention,
            radius_deg=float(radius),
            pixel_order=pixel_order,
        )
    raise ValueError(f"Unknown attention_backend '{attention_backend}'")


def pipeline_token_order(attention_backend: str) -> str:
    """The one pixel order the whole run uses: transform, obs packing, backbone, decoder."""
    attention_backend = normalize_attention_backend(attention_backend)
    return "nest" if attention_backend.startswith("disk-nest") else "hpxpadxy"


@dataclasses.dataclass
class Output:
    out: torch.Tensor
    obs: torch.Tensor | None = None


class DropPath(torch.nn.Module):
    """
    Stochastic Depth (DropPath)
    """

    def __init__(self, p=0.0):
        super().__init__()
        # Store as a non-persistent buffer so torch.compile treats it as a
        # dynamic tensor input rather than specializing on the float value
        # (each block has a different p, which would cause N recompiles).
        self.register_buffer("p", torch.tensor(float(p)), persistent=False)

    def forward(self, x):
        if not self.training:
            return x
        keep = 1.0 - self.p
        # broadcast mask over non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.3f})"


def _broadcast_modulation(param: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a per-sample modulation vector for broadcasting over hidden states.

    ``param`` is ``[batch, channels]``; it is reshaped to ``[batch, 1, ..., 1,
    channels]`` so it broadcasts over a hidden-state tensor with ``ndim`` dims
    (e.g. 4D ``[b, t, x, c]`` DiT-block tensors or 3D ``[b*t, npix, c]`` flattened
    pixel-attention tensors).
    """
    assert param.ndim == 2
    shape = (param.shape[0],) + (1,) * (ndim - 2) + (param.shape[1],)
    return param.view(shape)


class AdaLayerNormZero(nn.Module):
    r"""Adaptive layer norm zero (adaLN-Zero) modulation.

    Emits ``shift``/``scale``/``gate`` triples from the conditioning embedding and
    applies the affine-free LayerNorm + shift + scale to the hidden states. One
    triple is consumed here (for the immediately-following attention); any extra
    triples are returned for the caller to apply later (e.g. the feed-forward
    block). The same implementation serves both the 4D DiT-block hidden states and
    the 3D flattened pixel-attention hidden states: modulation vectors are
    broadcast to match ``x.ndim``.

    Parameters:
        embedding_dim: Channel dimension of the hidden states.
        emb_channels: Channel dimension of the conditioning embedding.
        n_blocks: Number of ``(shift, scale, gate)`` triples to produce. ``2`` for
            a DiT block (attention + feed-forward); ``1`` for a single
            cross-attention (the backbone observation attention).
    """

    def __init__(
        self,
        embedding_dim: int,
        emb_channels: int,
        n_blocks: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.linear = nn.Linear(emb_channels, 3 * n_blocks * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        chunks = self.linear(emb).chunk(3 * self.n_blocks, dim=1)
        shift, scale, gate = chunks[0], chunks[1], chunks[2]
        x = self.norm(x) * (
            1 + _broadcast_modulation(scale, x.ndim)
        ) + _broadcast_modulation(shift, x.ndim)
        outputs = [x, _broadcast_modulation(gate, x.ndim)]
        outputs.extend(_broadcast_modulation(extra, x.ndim) for extra in chunks[3:])
        return tuple(outputs)


class AdaLayerNormTemporalAttn(nn.Module):
    r"""Ada Layernorm which is only use for the temporal attn
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Could be fused with AdaLayerNormZero for a slight computational gain

    """

    def __init__(self, embedding_dim: int, emb_channels: int, bias=True):
        super().__init__()
        # TODO silu unused. Is this a bug? --noah 9/5/25
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_channels, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(emb)
        shift, scale, gate = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale[:, None, None]) + shift[:, None, None]
        return x, gate[:, None, None]


class TransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        drop_path: float = 0.0,
        temporal_attention: bool = False,  # TODO change to default of False
        qk_rms_norm: bool = False,
        linear_temporal_attention: bool = True,
        temporal_attn_legacy_scaling_bug: bool = False,
        temporal_causal_window: int | None = None,
        fused_temporal_attn: int = TEMPORAL_ATTN_EINSUM,
        attention_backend: str = "diffusers",
        pixel_order: str = "hpxpadxy",
        obs_cross_attention: bool = False,
        obs_token_dim: int | None = None,
        obs_q_heads: int | None = None,
        obs_kv_heads: int = 1,
        obs_q_head_dim: int = 32,
        pixel_attn_impl: int = 1,
        *,
        emb_channels: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = AdaLayerNormZero(dim, emb_channels=emb_channels)

        self.temporal_attn = None
        if temporal_attention:
            attn_cls = _spatial_attention_cls(attention_backend, pixel_order)
            self.temporal_attn_norm = AdaLayerNormTemporalAttn(dim, emb_channels)
            self.temporal_attn = TemporalAttention(
                embed_dim=dim,
                num_heads=num_attention_heads,
                linear_attention=linear_temporal_attention,
                temporal_attn_legacy_scaling_bug=temporal_attn_legacy_scaling_bug,
                causal_window=temporal_causal_window,
                temporal_attn_impl=fused_temporal_attn,
            )
        else:
            attn_cls = SpatioTemporalAttention

        self.attn1 = attn_cls(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            qk_norm="rms_norm" if qk_rms_norm else None,
            elementwise_affine=not qk_rms_norm,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.drop_path = DropPath(drop_path)

        # Backbone observation cross-attention (pixel latents attend to packed obs tokens).
        self.obs_attn = None
        self.obs_attn_norm = None
        if obs_cross_attention:
            if obs_token_dim is None:
                raise ValueError(
                    "obs_token_dim is required when obs_cross_attention=True"
                )
            if obs_q_heads is None:
                if dim % obs_token_dim != 0:
                    raise ValueError(
                        f"TransformerBlock dim={dim} must be divisible by "
                        f"obs_token_dim={obs_token_dim}"
                    )
                obs_q_heads = dim // obs_token_dim
                obs_q_head_dim = obs_token_dim
            self.obs_attn_norm = AdaLayerNormZero(
                embedding_dim=dim,
                emb_channels=emb_channels,
                n_blocks=1,
            )
            self.obs_attn = PixelCrossAttention(
                token_dim=obs_token_dim,
                input_dim=dim,
                output_dim=dim,
                n_q_heads=obs_q_heads,
                n_kv_heads=obs_kv_heads,
                d_head=obs_q_head_dim,
                use_proj_bias=True,
                pixel_attn_impl=pixel_attn_impl,
            )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            # self.fuser = GatedSelfAttentionDense(
            #     dim, cross_attention_dim, num_attention_heads, attention_head_dim
            # )
            raise NotImplementedError()

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self._parallel_group = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def _apply_obs_attention(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        obs_tokens,
        attention_packing,
    ) -> torch.Tensor:
        if self.obs_attn is None or self.obs_attn_norm is None:
            return hidden_states
        if obs_tokens is None or attention_packing is None:
            raise ValueError(
                "TransformerBlock with obs_cross_attention=True requires packed "
                "observation tokens."
            )

        batch_size, time_steps, npix, _ = hidden_states.shape
        total_pixels = attention_packing.cu_seqlens_k.shape[0] - 1
        expected_total_pixels = batch_size * time_steps * npix
        if total_pixels != expected_total_pixels:
            raise ValueError(
                f"Inconsistent obs attention packing: total_pixels={total_pixels}, "
                f"expected {expected_total_pixels}"
            )

        hidden_states_bt = hidden_states.reshape(
            batch_size * time_steps, npix, self.dim
        )
        emb_bt = (
            emb[:, None, :]
            .expand(batch_size, time_steps, -1)
            .reshape(batch_size * time_steps, emb.shape[-1])
        )
        norm_hidden_states, gate_obs = self.obs_attn_norm(hidden_states_bt, emb_bt)
        attn_output = self.obs_attn(
            norm_hidden_states,
            obs_tokens,
            total_pixels,
            attention_packing.cu_seqlens_k,
            group_map=attention_packing.group_map,
        ).view_as(hidden_states_bt)
        hidden_states_bt = torch.addcmul(
            hidden_states_bt, self.drop_path(gate_obs), attn_output
        )
        return hidden_states_bt.view_as(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        obs_tokens,
        attention_packing,
        input_t_sharded: bool,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        loop=None,
        is_causal: bool = False,  # if temporal attn is causal
        checkpoint_ff: bool = False,
    ) -> torch.Tensor:
        # input_t_sharded is carried through to the return rather than acted on: the caller
        # shards before the block so every block traces at one shape.

        # 0. Self-Attention
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb
        )

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        # Don't mutate dict
        if cross_attention_kwargs is not None:
            gligen_kwargs = cross_attention_kwargs.get("gligen")
            cross_attention_kwargs = {
                k: v for k, v in cross_attention_kwargs.items() if k != "gligen"
            }
        else:
            gligen_kwargs = None
            cross_attention_kwargs = {}

        with healda.utils.profiling.nvtx_range("attn1"):
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=(
                    encoder_hidden_states if self.only_cross_attention else None
                ),
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate_msa), attn_output
            )

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = self.drop_path(attn_output) + hidden_states

        # backbone observation cross-attention
        if self.obs_attn is not None:
            with healda.utils.profiling.nvtx_range("obs_attn"):
                hidden_states = self._apply_obs_attention(
                    hidden_states, emb, obs_tokens, attention_packing
                )

        # temporal attention
        if self.temporal_attn is not None:
            with healda.utils.profiling.nvtx_range("temporal_attn"):
                if self._parallel_group is not None:
                    hidden_states = shard_x(hidden_states, self._parallel_group)
                    input_t_sharded = False

                norm_hidden_states, gate = self.temporal_attn_norm(hidden_states, emb)
                attn_output = self.temporal_attn(norm_hidden_states, is_causal)
                hidden_states = torch.addcmul(
                    hidden_states, self.drop_path(gate), attn_output
                )

        # 4. Feed-forward
        # i2vgen doesn't have this norm ?????
        with healda.utils.profiling.nvtx_range("norm3"):
            norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp

        with healda.utils.profiling.nvtx_range("ff"):
            if checkpoint_ff:
                ff_output = torch.utils.checkpoint.checkpoint(
                    self.ff, norm_hidden_states, use_reentrant=False
                )
            else:
                ff_output = self.ff(norm_hidden_states)
            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate_mlp), ff_output
            )
        return hidden_states, input_t_sharded

    def set_domain_parallel_group(self, group):
        self.attn1.set_domain_parallel_group(group)

    def set_time_parallel_group(self, group):
        self._parallel_group = group


def my_subdomain(
    spatial_group, hpx_level, device: str = "cuda", batch_size: int = 1
) -> Subdomain | None:
    """
    Args:
        spatial_group: The spatial process group to use for context parallelism.
        hpx_level: The level of the HEALPix grid to use.
        device: The device to use for the subdomain.
        batch_size: The batch size to use for the subdomain.
    """
    if spatial_group is None:
        return None

    sp_rank = torch.distributed.get_rank(spatial_group)
    sp_size = torch.distributed.get_world_size(spatial_group)
    faces = torch.arange(12, device=device).view(batch_size, 12)
    faces = faces.chunk(sp_size, dim=-1)[sp_rank]

    return Subdomain(
        x=torch.zeros_like(faces),
        y=torch.zeros_like(faces),
        f=faces,
        n=2**hpx_level,
        level=hpx_level,
    )


class DiT(torch.nn.Module):
    r"""
    A 2D Transformer model as introduced in DiT (https://huggingface.co/papers/2212.09748).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    pixel_order = earth2grid.healpix.HEALPIX_PAD_XY

    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _supports_gradient_checkpointing = True
    _supports_group_offloading = False

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_size: int = 32,
        activation_fn: str = "gelu-approximate",
        qk_rms_norm: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        # hpx grid info
        level_in: int = 6,
        level_model: int = 4,
        time_length: int = 1,
        label_dim: int = 0,
        label_dropout: float = 0.0,
        legacy_label_bias: bool = False,
        obs_config: Optional[ObsConfig] = None,
        drop_path: float = 0.0,
        drop_path_uniform: bool = False,
        group_norm_eps: float = 1e-6,
        use_gains: bool = False,
        temporal_attention: bool = False,
        obs_decoder: bool = False,
        dense_decoder: bool = True,
        # FLAGS
        embed_v2: bool = False,
        embed_v2_meta_dim: int = 28,
        embed_v2_n_embed=1024,
        obs_hpx_level: int = 6,
        sensor_embedder_config: Optional[
            SensorEmbedderConfig
        ] = None,  # Per-sensor observation embedder config
        sensors: Optional[dict[str, "ModelSensorConfig"]] = None,  # Sensor configs
        compile_dit: bool = False,  # Enable torch.compile for _forward_DiT
        allow_nans_condition: bool = False,
        emb_channels: int | None = None,
        noise_channels: int | None = None,
        gradient_checkpointing: int = 0,
        gradient_checkpointing_last_n: int = 0,
        as_vit: bool = False,
        linear_temporal_attention: bool = True,
        temporal_attn_legacy_scaling_bug: bool = False,
        temporal_causal_window: int | None = None,
        # Temporal-attention implementation; see attention.TEMPORAL_ATTN_*.
        fused_temporal_attn: int = TEMPORAL_ATTN_EINSUM,
        attention_backend: str = "diffusers",
        add_calendar: bool = True,
        backbone_pixel_attention: bool = False,
        pixel_attn_impl: int = 1,
        drop_path_zero_first_n_blocks: int = 0,
    ):
        """
        Args:
            gradient_checkpointing:
                0: no gradient checkpointing
                1: gradient checkpointing for ff
                >1: gradient checkpoint for all blocks
            gradient_checkpointing_last_n:
                0 (default) applies the mode above to every block; N > 0 only to the last
                N. Each uncheckpointed block costs ~2.45 GiB at the production shape.
            as_vit: If True, skip noise/label conditioning entirely.
                Sets emb_channels=0 internally, making all AdaLN linear layers bias-only
        """
        super().__init__()
        self._level_in = level_in
        self._obs_hpx_level = obs_hpx_level
        attention_backend = normalize_attention_backend(attention_backend)
        self.spatial_token_order = pipeline_token_order(attention_backend)
        if attention_backend.startswith("disk") and not temporal_attention:
            # SpatioTemporalAttention is dense global, so the disk would be silently unused
            # while the whole pipeline still ran in nest.
            raise NotImplementedError(
                f"attention_backend={attention_backend!r} requires temporal_attention=True."
            )
        self.temporal_attention = temporal_attention
        self.compile_dit = compile_dit
        self.as_vit = as_vit
        assert level_in >= level_model
        patch_size = 2 ** (level_in - level_model)
        self.level_model = level_model

        self.time_length = time_length

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_last_n = gradient_checkpointing_last_n

        # The scatter-infill obs embedding and the backbone obs cross-attention are
        # mutually exclusive observation pathways.
        self.embed_v2_patch = None
        if (
            embed_v2
            and sensor_embedder_config is not None
            and not backbone_pixel_attention
        ):
            if sensors is None:
                raise ValueError(
                    "sensors is required when sensor_embedder_config is provided"
                )
            if obs_hpx_level != level_in:
                raise ValueError(
                    "obs_hpx_level must equal level_in while embed_v2 observations "
                    "are concatenated directly with hidden_states."
                )

            self.embed_v2_patch = MultiSensorObsEmbedding(
                sensor_embedder_config=sensor_embedder_config,
                sensors=sensors,
                hpx_level=self._obs_hpx_level,
                compile=compile_dit,
            )
            pos_embed_in_channels = in_channels + sensor_embedder_config.fusion_dim
        else:
            self.patch_size = patch_size
            pos_embed_in_channels = in_channels

        # Backbone observation cross-attention: tokenize observations once and let
        # each transformer block's pixel latents cross-attend to the packed tokens.
        self.backbone_pixel_attention = backbone_pixel_attention
        self.backbone_obs_tokenizer = None
        self.backbone_obs_token_dim = None
        if backbone_pixel_attention:
            if sensor_embedder_config is None:
                raise ValueError(
                    "backbone_pixel_attention=True requires a sensor_embedder_config"
                )
            # The conv/sat metadata split is applied on the data side (the transform
            # emits the pre-expanded [shared, sat, conv] layout); the tokenizer just
            # consumes the (possibly wider) metadata with a plain first linear.
            self.backbone_obs_tokenizer = ObsTokenizerFiLM(
                meta_dim=sensor_embedder_config.tokenizer_meta_dim,
                out_dim=sensor_embedder_config.embed_dim,
                obs_type_embed_dim=sensor_embedder_config.obs_type_embed_dim,
                channel_embed_dim=sensor_embedder_config.channel_embed_dim,
                platform_embed_dim=sensor_embedder_config.platform_embed_dim,
                use_fused_mlp=sensor_embedder_config.use_fused_mlp,
                use_global_channel_platform_ids=True,
                hidden_dim=sensor_embedder_config.film_hidden_dim,
                max_global_channels=sensor_embedder_config.max_global_channels,
            )
            self.backbone_obs_token_dim = sensor_embedder_config.embed_dim

        self.pos_embed = HPXPatchEmbed(
            in_channels=pos_embed_in_channels,
            out_channels=self.inner_dim,
            level_fine=level_in,
            level_coarse=level_model,
            use_gains=use_gains,
            allow_nans=allow_nans_condition,
            token_order=self.spatial_token_order,
            add_calendar=add_calendar,
        )

        if as_vit:
            emb_channels = 0
            self.noise_embed = None
        else:
            if emb_channels is None:
                emb_channels = 4 * self.inner_dim
            self.noise_embed = EmbedNoiseLabels(
                emb_channels,
                label_dim,
                noise_channels=(
                    noise_channels if noise_channels is not None else self.inner_dim
                ),
                label_dropout=label_dropout,
                legacy_label_bias=legacy_label_bias,
            )

        # 2. Initialize the position embedding and transformer blocks.
        self.height = sample_size
        self.width = sample_size

        if drop_path_uniform:
            drop_path_schedule = [drop_path] * num_layers
        else:
            drop_path_schedule = [
                drop_path * i / max(1, num_layers - 1) for i in range(num_layers)
            ]
        # Optionally zero drop-path for the first N blocks (keeps early-layer signal).
        for i in range(min(drop_path_zero_first_n_blocks, num_layers)):
            drop_path_schedule[i] = 0.0

        obs_q_heads = None
        obs_q_head_dim = attention_head_dim
        obs_kv_heads = 1
        if backbone_pixel_attention:
            obs_q_heads = sensor_embedder_config.backbone_pixel_attn_num_heads
            obs_q_head_dim = sensor_embedder_config.pixel_attn_head_dim
            obs_kv_heads = sensor_embedder_config.pixel_attn_kv_heads

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    emb_channels=emb_channels,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    qk_rms_norm=qk_rms_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    drop_path=drop_path_schedule[i],
                    temporal_attention=temporal_attention,
                    linear_temporal_attention=linear_temporal_attention,
                    temporal_attn_legacy_scaling_bug=temporal_attn_legacy_scaling_bug,
                    temporal_causal_window=temporal_causal_window,
                    fused_temporal_attn=fused_temporal_attn,
                    attention_backend=attention_backend,
                    pixel_order=self.spatial_token_order,
                    obs_cross_attention=backbone_pixel_attention,
                    obs_token_dim=self.backbone_obs_token_dim,
                    obs_q_heads=obs_q_heads,
                    obs_kv_heads=obs_kv_heads,
                    obs_q_head_dim=obs_q_head_dim,
                    pixel_attn_impl=pixel_attn_impl,
                )
                for i in range(num_layers)
            ]
        )

        # 3. Output blocks.
        self.proj_out_1 = nn.Linear(emb_channels, 2 * self.inner_dim)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)

        if dense_decoder:
            self.patch_decode = HPXPatchDecode(
                in_channels=self.inner_dim,
                out_channels=out_channels,
                level_fine=level_in,
                level_coarse=level_model,
                token_order=self.spatial_token_order,
            )
        else:
            self.patch_decode = None

        if obs_decoder:
            self.decode_obs = ObsDecoder(
                self.inner_dim,
                metadata_dim=embed_v2_meta_dim,
                hpx_fine_level=8,
                hpx_in_level=level_model,
                max_embed_id=1024,
                obs_dim=1,
            )
        else:
            self.decode_obs = None

        if self.inner_dim % 4 != 0:
            raise ValueError(self.inner_dim)

        self._parallel_group = None
        self._domain_parallel_group = None
        # Collected so forward() can build each disk attention's block mask eagerly: the
        # traced mask_mod fails to lower inside the compiled region.
        self._disk_attentions = [
            m for m in self.modules() if isinstance(m, DiskSpatialAttention)
        ]
        self._compile()

    def _compile(self):
        if self.compile_dit:
            self._call_forward_blocks = torch.compile(self._forward_blocks)
        else:
            self._call_forward_blocks = self._forward_blocks

    @property
    def grid(self):
        return earth2grid.healpix.Grid(
            level=self._level_in, pixel_order=self.pixel_order
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def domain(self):
        return HealPixDomain(self.grid)

    def load_from_checkpoint(self, checkpoint):
        """Load from checkpoint adjusting settings for fine tuning"""
        state_dict = checkpoint.read_model_state_dict()
        config = checkpoint.read_model_config()

        # old checkpoints before I added fsdp training
        # were saved with label_dim = 0
        if self.noise_embed is not None and self.noise_embed.map_label is None:
            state_dict.pop("noise_embed.map_label.weight", None)
            state_dict.pop("noise_embed.map_label.bias", None)

        if self.temporal_attention and (not config.dit_temporal_attention):
            my_state_dict = self.state_dict()
            import re

            for key in my_state_dict:
                if not re.match(r"transformer_blocks\..*\.temporal_attn", key):
                    continue

                if key in state_dict:
                    continue

                state_dict[key] = my_state_dict[key]

        self.load_state_dict(state_dict)

    def encode(
        self,
        hidden_states: torch.Tensor,
        unified_obs: UnifiedObservation | None = None,
        second_of_day: torch.Tensor | None = None,
        day_of_year: torch.Tensor | None = None,
        subdomain: Subdomain | None = None,
    ) -> torch.Tensor:
        # scatter states from sp_rank 0 to all ranks
        # inputs from other ranks are ignored
        if self.embed_v2_patch is not None:
            if unified_obs is None:
                raise ValueError("Need to provide observation inputs.")

            with healda.utils.profiling.nvtx_range("dit:obs_embed"):
                obs_emb = self.embed_v2_patch(unified_obs)

            hidden_states = torch.cat([hidden_states, obs_emb], dim=1)

        # 1. Input
        hidden_states = self.pos_embed(
            hidden_states,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
            subdomain=subdomain,
        )

        return hidden_states

    def _forward_blocks(
        self,
        hidden_states,
        input_t_sharded,
        obs_tokens,
        attention_packing,
        blocks,
        gradient_checkpointing,
        noise_labels,
        class_labels,
        is_causal,
        gradient_checkpointing_last_n=0,
    ):
        # In vit mode, emb is ignored - Adapative scaling layers just return bias
        if self.as_vit:
            b = hidden_states.shape[0]
            emb = torch.empty(
                b, 0, device=hidden_states.device, dtype=hidden_states.dtype
            )
        else:
            if noise_labels is None:
                raise ValueError("noise_labels is required when as_vit=False")
            emb = self.noise_embed(noise_labels, class_labels)

        n_blocks = len(blocks)
        if gradient_checkpointing_last_n > 0:
            checkpointed = set(
                range(max(0, n_blocks - gradient_checkpointing_last_n), n_blocks)
            )
        else:
            checkpointed = set(range(n_blocks))
        for i, block in enumerate(blocks):
            block_checkpointing = gradient_checkpointing if i in checkpointed else 0
            with healda.utils.profiling.nvtx_range(f"dit:block{i}"):
                # Time-shard here, not at the block's own entry, so every block is traced
                # at one shape: a second shape makes dynamo treat t and x as symbolic, and
                # the DDP graph splitter then passes symints where AOTAutograd wants ints.
                if self._parallel_group is not None and not input_t_sharded:
                    hidden_states = shard_t(hidden_states, self._parallel_group)
                    input_t_sharded = True
                args = (hidden_states, emb, obs_tokens, attention_packing)
                if block_checkpointing > 1:
                    hidden_states, input_t_sharded = torch.utils.checkpoint.checkpoint(
                        partial(
                            block,
                            input_t_sharded=input_t_sharded,
                        ),
                        *args,
                        use_reentrant=False,
                        is_causal=is_causal,
                    )

                else:
                    hidden_states, input_t_sharded = block(
                        *args,
                        input_t_sharded=input_t_sharded,
                        is_causal=is_causal,
                        checkpoint_ff=block_checkpointing == 1,
                    )

        # the output should be sharded in time
        if self._parallel_group is not None and not input_t_sharded:
            hidden_states = shard_t(hidden_states, self._parallel_group)

        # 3. Output
        shift, scale = self.proj_out_1(emb).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None, None])
            + shift[:, None, None]
        )

        return hidden_states

    def _get_backbone_obs_tokenizer(self) -> torch.nn.Module:
        if self.backbone_obs_tokenizer is None:
            raise ValueError(
                "backbone_pixel_attention=True requires a backbone_obs_tokenizer."
            )
        return self.backbone_obs_tokenizer

    def _get_backbone_obs_inputs(
        self,
        unified_obs: UnifiedObservation | None,
    ) -> tuple[torch.Tensor | None, AttentionPacking | None]:
        """Return ``(obs_tokens, attention_packing)`` for backbone pixel attention.

        ``obs_tokens`` are the tokenized, pixel-sorted, packed observations and
        ``attention_packing`` holds the per-pixel ``counts``/``cu_seqlens_k`` over
        the full pixel grid.

        The backbone pixel-attention path *requires* physically packed
        observations (``attention_prepack=True`` in the data transform). The
        alternative scatter/``embed_v2_patch`` path does not use packed obs and
        returns ``(None, None)`` here.
        """
        if not self.backbone_pixel_attention:
            return None, None
        if unified_obs is None:
            raise ValueError(
                "Need to provide unified_obs when backbone_pixel_attention=True."
            )
        attention_packing = unified_obs.attention_packing
        if attention_packing is None:
            raise ValueError(
                "backbone_pixel_attention=True requires prepacked attention observations."
            )
        if attention_packing.hpx_level != self.level_model:
            raise ValueError(
                f"Expected obs attention packing at model hpx_level {self.level_model}, "
                f"got {attention_packing.hpx_level}"
            )
        if attention_packing.pixel_order != self.spatial_token_order:
            raise ValueError(
                f"Observations packed in {attention_packing.pixel_order!r} but the backbone runs "
                f"in {self.spatial_token_order!r}; set the transform's pixel_order from "
                "dit.pipeline_token_order(...)."
            )
        if not attention_packing.is_packed:
            raise ValueError(
                "backbone_pixel_attention=True expects physically packed observations."
            )
        if unified_obs.obs_tokens is not None:
            return unified_obs.obs_tokens, attention_packing
        tokenizer = self._get_backbone_obs_tokenizer()
        return tokenizer(unified_obs), attention_packing

    def tokenize_observations(
        self, unified_obs: UnifiedObservation
    ) -> UnifiedObservation:
        """``unified_obs`` with its tokens computed, which the forward then uses as given.

        Drops the reference to ``float_metadata`` so its memory can be freed: the
        tokenizer is its only reader, and it is (n_obs, 50) float32.
        """
        if not self.backbone_pixel_attention:
            return unified_obs
        tokens, _ = self._get_backbone_obs_inputs(unified_obs)
        return dataclasses.replace(unified_obs, obs_tokens=tokens, float_metadata=None)

    @healda.utils.profiling.nvtx
    def forward(
        self,
        hidden_states: torch.Tensor,
        noise_labels: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        day_of_year: torch.Tensor | None = None,
        second_of_day: torch.Tensor | None = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        unified_obs: UnifiedObservation | None = None,
        timestamp=None,
        is_causal: bool = False,
        subdomain: Subdomain | None = None,
        level_localize: int | None = None,
        input_t_sharded: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states: network input. shaped [b, c, t, x]. Sharded along t input_t_sharded is True, otherwise it is sharded along x.
            unified_obs: UnifiedObservation object. Sharded along nobs dimension
                if run with spatial parallelism. with seperable parallelism it is
                sharded along time.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            timestamp: [b t] shaped tensor. contains integer timestamp for each frame.
            input_t_sharded: whether the input is sharded across the time or spatial dimension. Default is True. Only used when
                set_parallel_group has been called.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        obs_tokens, attention_packing = self._get_backbone_obs_inputs(unified_obs)
        if (
            self._domain_parallel_group is None
            or not torch.distributed.is_initialized()
        ):
            sp_size = 1
        else:
            sp_size = torch.distributed.get_world_size(self._domain_parallel_group)

        hidden_states = self.encode(
            hidden_states, unified_obs, second_of_day, day_of_year, subdomain
        )  # (b, t, TP x, c) # t-sharded

        if level_localize is not None:
            if self._disk_attentions:
                # The disk mask is built from a global token count; the localized blocks
                # below would hand it a per-block one.
                raise NotImplementedError(
                    "level_localize does not support the disk attention backend"
                )
            hidden_states = hidden_states.movedim(-2, -1)
            hidden_states = earth2grid.healpix.reorder(
                hidden_states,
                healpix_pixel_order(self.spatial_token_order),
                earth2grid.healpix.NEST,
            )
            hidden_states = einops.rearrange(
                hidden_states,
                "b t c (n x) -> (b n) t x c",
                n=12 * 4 ** (self.level_model - level_localize) // sp_size,
            )

        # Eager, so disk-mask construction stays out of the compiled graph.
        n_tokens = hidden_states.shape[-2]
        for attention in self._disk_attentions:
            attention.prepare_block_mask(n_tokens, hidden_states.device)

        hidden_states = self._call_forward_blocks(
            hidden_states,
            input_t_sharded,
            obs_tokens,
            attention_packing,
            self.transformer_blocks,
            self.gradient_checkpointing,
            noise_labels,
            class_labels,
            is_causal,
            gradient_checkpointing_last_n=self.gradient_checkpointing_last_n,
        )

        if level_localize:
            hidden_states = einops.rearrange(
                hidden_states,
                "(b n) t x c -> b t c (n x)",
                n=12 * 4 ** (self.level_model - level_localize),
            )
            hidden_states = earth2grid.healpix.reorder(
                hidden_states,
                earth2grid.healpix.NEST,
                healpix_pixel_order(self.spatial_token_order),
            )
            hidden_states = hidden_states.movedim(-1, -2)

        # Coarsen subdomain to model level before passing to decoder
        subdomain_coarse = None
        if subdomain is not None:
            subdomain_coarse = subdomain.coarsen(self._level_in - self.level_model)

        # 4. Decode (not compiled - includes HEALPix operations)
        return Output(
            out=self._decode_dense(hidden_states, subdomain_coarse),
            obs=self._decode_obs(hidden_states, unified_obs),
        )

    @healda.utils.profiling.nvtx
    def _decode_dense(
        self,
        hidden_states: torch.Tensor,
        subdomain: Subdomain | None,
    ) -> torch.Tensor:
        if self.patch_decode is None:
            return hidden_states

        return self.patch_decode(hidden_states, subdomain)

    def _decode_obs(
        self,
        hidden_states: torch.Tensor,
        unified_obs: UnifiedObservation | None,
    ) -> None | torch.Tensor:
        if self.decode_obs is None:
            return None

        if unified_obs is None or unified_obs.lengths is None:
            raise ValueError(
                "Need to provide unified_obs object with lengths for decoder."
            )

        _, B, T = unified_obs.lengths.shape
        batch_idx = lengths_to_idx(
            unified_obs.lengths, output_size=unified_obs.obs.shape[0]
        ) % (B * T)
        return self.decode_obs(
            latent=hidden_states,
            batch_idx=batch_idx,
            metadata=unified_obs.float_metadata,
            pix=unified_obs.pix,
            platform=unified_obs.global_platform,
            obs_type=unified_obs.obs_type,
            channel=unified_obs.global_channel,
            hpx_level=unified_obs.hpx_level,
        )

    def set_domain_parallel_group(self, group):
        """Set the Transformer Engine context parallel group for spatial or spatio temporal attention.

        Uses Transformer Engine's context parallelism implementation.
        Allows parallelism up to the number of spatial tokens.

        Args:
            group: The process group to use for context parallelism.
        """
        sp_size = torch.distributed.get_world_size(group)
        if 12 % sp_size != 0:
            raise ValueError(
                f"SP_size {sp_size} must divide 12 (the number of healpix faces)."
            )
        self._domain_parallel_group = group
        if self.embed_v2_patch is not None:
            self.embed_v2_patch.set_spatial_sharding_group(group)

        for block in self.transformer_blocks:
            block.set_domain_parallel_group(group)

        if self.decode_obs is not None:
            self.decode_obs.set_parallel_group(group)

        # need to recompile since the attention processor is changed
        torch._dynamo.reset()
        self._compile()

    def set_time_parallel_group(self, group):
        """Set the separable context parallel group.

        Uses all_to_all to distribute data before temporal and spatial attention.
        Allows parallelism up to gcd(number of frames, number of spatial tokens).

        Args:
            group: The process group to use for context parallelism.
        """
        self._parallel_group = group
        for block in self.transformer_blocks:
            block.set_time_parallel_group(group)

    def fully_shard(
        self,
        mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
        mp_policy: "torch.distributed.fsdp.MixedPrecisionPolicy | None" = None,
    ):
        """
        Can be composed with set_time_parallel_group or set_domain_parallel_group to achieve
        hybrid 2D parallelism.

        Args:
            mesh: This data parallel mesh defines the sharding and device. If
                1D, then parameters are fully sharded across the 1D mesh (FSDP) with (Shard(0),)
                placement. If 2D, then parameters are sharded across the 1st dim and replicated
                across the 0th dim (HSDP) with (Replicate(), Shard(0)) placement. The mesh’s
                device type gives the device type used for communication; if a CUDA or CUDA-like
                device type, then we use the current device.
            mp_policy: FSDP mixed-precision policy. If None, no mixed precision is used.

        """
        kwargs = {}
        if mp_policy is not None:
            kwargs["mp_policy"] = mp_policy

        for block in self.transformer_blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh, **kwargs)

        torch.distributed.fsdp.fully_shard(self, mesh=mesh, **kwargs)
