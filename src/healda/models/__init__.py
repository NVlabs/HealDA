# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from healda.models import dit
from healda.config.models import ARCHITECTURES, ModelConfigV1
from healda.datasets.da.transform import TransformOptions
from healda.models.obs_embedding.pixel_cross_attention import PIXEL_ATTN_TRITON


def _dit_level_in(config: ModelConfigV1) -> int:
    if config.level_in is not None:
        return config.level_in
    return ARCHITECTURES[config.architecture].level_in


def transform_options_for(config: ModelConfigV1) -> TransformOptions:
    """How this model wants observations packed.

    Training and inference must agree exactly -- the packing decides pixel ids and token
    order -- so it is derived from the model config in one place rather than twice.
    """
    embedder = config.sensor_embedder_config
    return TransformOptions(
        # The backbone packs at the coarser level_model; the obs encoder ingests at
        # level_in, which model_config already stores as obs_hpx_level.
        observation_hpx_level=(
            ARCHITECTURES[config.architecture].level_model
            if config.backbone_pixel_attention
            else config.obs_hpx_level
        ),
        features_v2=embedder is not None and embedder.features_v2,
        attention_prepack=config.backbone_pixel_attention,
        build_attention_group_map=config.pixel_attn_impl == PIXEL_ATTN_TRITON,
        pixel_order=dit.pipeline_token_order(config.attention_backend),
    )


def get_model(config: ModelConfigV1) -> torch.nn.Module:
    sensor_embedder_config = config.sensor_embedder_config
    # TODO remove embed_v2 option flag since it is redundant with sensor config
    embed_v2 = sensor_embedder_config is not None and config.sensors is not None
    meta_dim = sensor_embedder_config.meta_dim if sensor_embedder_config else 0

    if config.architecture == "dit-b_reg_hpx6":
        return dit.DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=12,
            num_layers=12,
            attention_head_dim=64,
            level_in=_dit_level_in(config),
            dense_decoder=config.dense_decoder,
            attention_backend=config.attention_backend,
            add_calendar=config.add_calendar,
            level_model=5,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            drop_path_uniform=config.drop_path_uniform,
            dropout=config.p_dropout,
            time_length=config.time_length,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            temporal_attention=config.dit_temporal_attention,
            embed_v2=embed_v2,
            embed_v2_meta_dim=meta_dim,
            obs_hpx_level=config.obs_hpx_level,
            sensor_embedder_config=sensor_embedder_config,
            sensors=config.sensors,
            compile_dit=config.compile_dit,
            qk_rms_norm=config.qk_rms_norm,
            allow_nans_condition=config.allow_nans_condition,
            as_vit=config.as_vit,
            linear_temporal_attention=config.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=config.temporal_attn_legacy_scaling_bug,
        )
    elif config.architecture == "dit-test":
        """dit with configs for fast testing"""
        return dit.DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=2,
            num_layers=1,
            attention_head_dim=64,
            level_in=_dit_level_in(config),
            dense_decoder=config.dense_decoder,
            attention_backend=config.attention_backend,
            add_calendar=config.add_calendar,
            level_model=5,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            drop_path_uniform=config.drop_path_uniform,
            dropout=config.p_dropout,
            time_length=config.time_length,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            temporal_attention=config.dit_temporal_attention,
            embed_v2=embed_v2,
            obs_hpx_level=config.obs_hpx_level,
            compile_dit=config.compile_dit,
            qk_rms_norm=config.qk_rms_norm,
            allow_nans_condition=config.allow_nans_condition,
            embed_v2_meta_dim=meta_dim,
            sensor_embedder_config=sensor_embedder_config,
            sensors=config.sensors,
            as_vit=config.as_vit,
            linear_temporal_attention=config.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=config.temporal_attn_legacy_scaling_bug,
        )
    elif config.architecture == "dit-l_reg_hpx6":
        return dit.DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=16,
            num_layers=24,
            attention_head_dim=64,
            level_in=_dit_level_in(config),
            dense_decoder=config.dense_decoder,
            attention_backend=config.attention_backend,
            add_calendar=config.add_calendar,
            level_model=5,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            drop_path_uniform=config.drop_path_uniform,
            dropout=config.p_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            time_length=config.time_length,
            temporal_attention=config.dit_temporal_attention,
            embed_v2=embed_v2,
            embed_v2_meta_dim=meta_dim,
            obs_hpx_level=config.obs_hpx_level,
            sensor_embedder_config=sensor_embedder_config,
            sensors=config.sensors,
            qk_rms_norm=config.qk_rms_norm,
            compile_dit=config.compile_dit,
            allow_nans_condition=config.allow_nans_condition,
            as_vit=config.as_vit,
            linear_temporal_attention=config.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=config.temporal_attn_legacy_scaling_bug,
            temporal_causal_window=config.temporal_causal_window,
            fused_temporal_attn=config.fused_temporal_attn,
        )
    elif config.architecture == "dit-5B" or config.architecture.startswith(
        "dit-5B-d128"
    ):
        # Every dit-5B-d128* variant shares this body; depth and model level come from the
        # config.models registries. head_dim 128 matches the te backend's kernel block size.
        heads, head_dim = (16, 96) if config.architecture == "dit-5B" else (12, 128)
        return dit.DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            # suggested by cursor based on lit review
            # Hidden/Depth ratio: 40-60
            # Don't exceed depth of ~36-40 for models <10B
            num_attention_heads=heads,
            num_layers=ARCHITECTURES[config.architecture].num_layers,
            attention_head_dim=head_dim,
            level_in=_dit_level_in(config),
            level_model=ARCHITECTURES[config.architecture].level_model,
            dense_decoder=config.dense_decoder,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            drop_path_uniform=config.drop_path_uniform,
            dropout=config.p_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            time_length=config.time_length,
            temporal_attention=config.dit_temporal_attention,
            qk_rms_norm=config.qk_rms_norm,
            compile_dit=config.compile_dit,
            allow_nans_condition=config.allow_nans_condition,
            embed_v2=embed_v2,
            embed_v2_meta_dim=meta_dim,
            obs_hpx_level=config.obs_hpx_level,
            sensor_embedder_config=sensor_embedder_config,
            sensors=config.sensors,
            emb_channels=config.emb_channels,
            noise_channels=config.noise_channels,
            as_vit=config.as_vit,
            linear_temporal_attention=config.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=config.temporal_attn_legacy_scaling_bug,
            temporal_causal_window=config.temporal_causal_window,
            fused_temporal_attn=config.fused_temporal_attn,
            attention_backend=config.attention_backend,
            add_calendar=config.add_calendar,
            backbone_pixel_attention=config.backbone_pixel_attention,
            pixel_attn_impl=config.pixel_attn_impl,
            drop_path_zero_first_n_blocks=config.drop_path_zero_first_n_blocks,
        )
    elif config.architecture == "dit-l-lev4":
        # similar flops as dit-5B, but shifted towards MLP - should increase model flop utilization
        return dit.DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=24,
            num_layers=32,
            attention_head_dim=128,
            level_in=_dit_level_in(config),
            dense_decoder=config.dense_decoder,
            attention_backend=config.attention_backend,
            add_calendar=config.add_calendar,
            level_model=4,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            drop_path_uniform=config.drop_path_uniform,
            dropout=config.p_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            time_length=config.time_length,
            temporal_attention=config.dit_temporal_attention,
            qk_rms_norm=config.qk_rms_norm,
            compile_dit=config.compile_dit,
            allow_nans_condition=config.allow_nans_condition,
            embed_v2=embed_v2,
            embed_v2_meta_dim=meta_dim,
            obs_hpx_level=config.obs_hpx_level,
            sensor_embedder_config=sensor_embedder_config,
            sensors=config.sensors,
            emb_channels=config.emb_channels,
            noise_channels=config.noise_channels,
            as_vit=config.as_vit,
            linear_temporal_attention=config.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=config.temporal_attn_legacy_scaling_bug,
        )
    elif config.architecture == "dit-10B_reg_hpx6":
        return dit.DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=16,
            num_layers=40,
            attention_head_dim=128,
            level_in=_dit_level_in(config),
            dense_decoder=config.dense_decoder,
            attention_backend=config.attention_backend,
            add_calendar=config.add_calendar,
            level_model=5,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            drop_path_uniform=config.drop_path_uniform,
            dropout=config.p_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            time_length=config.time_length,
            temporal_attention=config.dit_temporal_attention,
            embed_v2=embed_v2,
            embed_v2_meta_dim=meta_dim,
            obs_hpx_level=config.obs_hpx_level,
            sensor_embedder_config=sensor_embedder_config,
            sensors=config.sensors,
            qk_rms_norm=config.qk_rms_norm,
            compile_dit=config.compile_dit,
            allow_nans_condition=config.allow_nans_condition,
            as_vit=config.as_vit,
            linear_temporal_attention=config.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=config.temporal_attn_legacy_scaling_bug,
        )
    elif config.architecture == "dit-l_reg_hpx6_per_sensor":
        return dit.DiT(
            in_channels=config.condition_channels,
            out_channels=config.out_channels,
            num_attention_heads=16,
            num_layers=24,
            attention_head_dim=64,
            level_in=_dit_level_in(config),
            dense_decoder=config.dense_decoder,
            attention_backend=config.attention_backend,
            add_calendar=config.add_calendar,
            level_model=5,
            label_dim=config.label_dim,
            label_dropout=config.label_dropout,
            legacy_label_bias=config.legacy_label_bias,
            obs_config=config.obs_config,
            drop_path=config.drop_path,
            drop_path_uniform=config.drop_path_uniform,
            dropout=config.p_dropout,
            group_norm_eps=config.group_norm_eps,
            use_gains=config.pos_emb_gains,
            time_length=config.time_length,
            temporal_attention=config.dit_temporal_attention,
            embed_v2=embed_v2,
            embed_v2_meta_dim=meta_dim,
            obs_hpx_level=config.obs_hpx_level,
            sensor_embedder_config=sensor_embedder_config,
            sensors=config.sensors,
            compile_dit=config.compile_dit,
            qk_rms_norm=config.qk_rms_norm,
            allow_nans_condition=config.allow_nans_condition,
            as_vit=config.as_vit,
            emb_channels=config.emb_channels,
            noise_channels=config.noise_channels,
            linear_temporal_attention=config.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=config.temporal_attn_legacy_scaling_bug,
        )
    else:
        raise NotImplementedError(config.architecture)
