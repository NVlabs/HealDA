# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json


@dataclasses.dataclass(frozen=True)
class Arch:
    """Geometry and depth of one backbone architecture.

    ``level_in`` is where observations are ingested; ``level_model`` is the coarser level
    the transformer blocks -- and the per-pixel obs cross-attention packing -- run at.
    ``num_layers`` is set only for the families ``get_model`` builds off this table.
    """

    level_in: int
    level_model: int
    num_layers: int | None = None


ARCHITECTURES: dict[str, Arch] = {
    "dit-b_reg_hpx6": Arch(level_in=6, level_model=5),
    "dit-test": Arch(level_in=6, level_model=5),
    "dit-l_reg_hpx6": Arch(level_in=6, level_model=5),
    "dit-l_reg_hpx6_per_sensor": Arch(level_in=6, level_model=5),
    "dit-10B_reg_hpx6": Arch(level_in=6, level_model=5),
    "dit-l-lev4": Arch(level_in=6, level_model=4),
    "dit-5B": Arch(level_in=6, level_model=5, num_layers=32),
    "dit-5B-d128": Arch(level_in=6, level_model=5, num_layers=32),
    "dit-5B-d128-2L": Arch(level_in=6, level_model=5, num_layers=2),
    "dit-5B-d128-16L-lev6": Arch(level_in=6, level_model=6, num_layers=16),
    "dit-5B-d128-18L-lev6": Arch(level_in=6, level_model=6, num_layers=18),
}


def _filter_to_dataclass_fields(d: dict, cls) -> dict:
    """Filter dict to only include fields defined in the dataclass."""
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in valid_fields}


# Checkpoints predating the presets hold a count of the ranked set instead of a name.
def ir_preset_name(value: str | int | None) -> str | None:
    return f"ir{value}" if isinstance(value, int) else value


@dataclasses.dataclass(frozen=True)
class ObsConfig:
    use_obs: bool = False
    innovation_type: str = "none"
    context_start: int = -21  # start/end in hours
    context_end: int = 3
    use_infrared: bool = False
    use_infrared_pca: bool = False
    use_airs_pca: bool = False
    use_conv: bool = False
    conv_uv_in_situ_only: bool = False
    conv_gps_level1_only: bool = False
    # Drop NCEP-restricted aircraft (AMDAR/ACARS/TAMDAR). Public GDAS omits them;
    # NNJA reanalysis PrepBUFR includes them and otherwise inflates conv volume.
    drop_restricted_aircraft: bool = False
    conv_min_pressure_hpa: float | None = None
    use_conv_level_stats: bool = False
    conv_level_channels: bool = False
    # Optional list of global observation channel IDs to drop
    # (these correspond to GLOBAL_CHANNEL_ID in the unified obs schema).
    # When None, no channels are explicitly dropped by ID.
    drop_obs_channel_ids: list[int] | None = None
    # Optional sensors to exclude from loading entirely (e.g. ("airs-pca",)).
    # Used for leave-one-sensor-out sensitivity studies at inference.
    drop_sensors: tuple[str, ...] = ()
    # PrepBUFR report types to deny, e.g. tuple(range(240, 261)) for AMVs. Applied after
    # the archives are merged, so it reaches AMVs from the dedicated SATWND archive too.
    drop_report_types: tuple[int, ...] = ()
    # ObsFamily names to deny, e.g. ("amv", "scatterometer"). Expanded to report types.
    drop_obs_families: tuple[str, ...] = ()
    # Read satellites from the NNJA archive instead of the UFS replay, pairing them with
    # conventional obs. Default conventional source is still UFS; set use_nnja_conv to
    # pull PrepBUFR + GPS-RO from the NNJA parquet archives instead. Channel vocabulary
    # is `nnja_combined`, so checkpoints are not interchangeable with a GSI-vocabulary run.
    use_nnja_sat: bool = False
    # With use_nnja_sat, also pull PrepBUFR + GPS-RO from NNJA instead of UFS replay.
    # Still expanded to the same 92-channel conv-plevel vocabulary CombinedObsLoader expects.
    use_nnja_conv: bool = False
    # Add the dedicated NNJA SATWND/AMV archive to the conventional u/v channels.
    # Overlapping PrepBUFR SATWND report types are suppressed by the loader.
    use_nnja_satwnd: bool = False
    # Training-only row dropout for SATWND, ASCAT, and aircraft wind reports.
    nnja_wind_dropout: float = 0.0
    # Training-only dropout of surface-pressure observations, independent of the wind
    # rate. Drops the ps channel only; the row's other variables survive.
    nnja_surface_pressure_dropout: float = 0.0
    # Treat ASCAT as the only scatterometer
    ascat_only_scatterometer: bool = False
    # How every NNJA random drop applies: "row" per observation, "sample" all-or-nothing
    # for the whole sample. Sample scope needs LoopConfig.fix_time_parallel_rng.
    nnja_dropout_scope: str = "row"
    # (sensor, platform, raw channel id, probability) drop rules. Below 1 they are a
    # training-only regulariser; at 1 they are a denial and hold at inference too.
    nnja_platform_channel_dropout: tuple[tuple[str, str, int, float], ...] = ()
    # None loads `sensors_nnja.DEFAULT_SENSORS`; a tuple names sensors instead.
    nnja_sensors: tuple[str, ...] | None = None
    # An `ir_spectral.IR_CHANNEL_PRESETS` name, narrowing the IR sounders and leaving microwave
    # alone; None keeps every channel the archive publishes.
    nnja_ir_channels: str | None = "ir32"
    # The archive is unthinned, unlike the GSI-thinned replay: 64 costs 494M rows a sample
    # against 31M. None is full resolution and is only for measuring what thinning removes.
    nnja_thin_nside: int | None = 64
    # GPS-RO receiving satellites (BUFR SAIDs) to keep: "full" or the 2024-derived "legacy".
    nnja_gpsro_saids: str = "legacy"
    # Load surface wind types 280/281/287
    nnja_surface_winds: bool = False
    # Largest PrepBUFR quality marker the conventional loader keeps; None disables the cut.
    nnja_max_quality_mark: int | None = 2
    # HEALPix level of SATWND spatial thinning; see NNJASatwndLoader for the full key.
    nnja_satwnd_thin_hpx_level: int = 5

    def __post_init__(self):
        # JSON round-trips tuples as lists; canonicalize so equality holds.
        object.__setattr__(self, "drop_sensors", tuple(self.drop_sensors))
        object.__setattr__(self, "drop_report_types", tuple(self.drop_report_types))
        object.__setattr__(self, "drop_obs_families", tuple(self.drop_obs_families))
        if self.nnja_sensors is not None:
            object.__setattr__(self, "nnja_sensors", tuple(self.nnja_sensors))
        try:
            channel_dropout = tuple(
                (str(sensor), str(platform), int(channel), float(probability))
                for sensor, platform, channel, probability in (
                    self.nnja_platform_channel_dropout
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "nnja_platform_channel_dropout entries must be "
                "(sensor, platform, channel, probability)"
            ) from error
        object.__setattr__(self, "nnja_platform_channel_dropout", channel_dropout)
        object.__setattr__(
            self, "nnja_ir_channels", ir_preset_name(self.nnja_ir_channels)
        )
        # Checkpoints written before the rename hold the old spellings.
        old = {"ufs2024": "legacy", "all": "full"}.get(self.nnja_gpsro_saids)
        if old is not None:
            object.__setattr__(self, "nnja_gpsro_saids", old)
        # nnja_combined's conv block is built from the per-pressure-level channels, so the
        # loader would silently disagree with the vocabulary if conv stayed unlevelled.
        if self.use_nnja_sat and not self.conv_level_channels:
            raise ValueError("use_nnja_sat requires conv_level_channels=True")
        if self.use_nnja_conv and not self.use_nnja_sat:
            raise ValueError("use_nnja_conv requires use_nnja_sat=True")
        if self.use_nnja_satwnd and not self.use_nnja_conv:
            raise ValueError("use_nnja_satwnd requires use_nnja_conv=True")
        if not 0.0 <= self.nnja_wind_dropout <= 1.0:
            raise ValueError(
                f"nnja_wind_dropout must be in [0, 1], got {self.nnja_wind_dropout}"
            )
        if self.nnja_dropout_scope not in ("row", "sample"):
            raise ValueError(
                f'nnja_dropout_scope must be "row" or "sample", got '
                f"{self.nnja_dropout_scope!r}"
            )
        if not 0.0 <= self.nnja_surface_pressure_dropout <= 1.0:
            raise ValueError(
                "nnja_surface_pressure_dropout must be in [0, 1], got "
                f"{self.nnja_surface_pressure_dropout}"
            )
        if self.nnja_gpsro_saids not in ("full", "legacy"):
            raise ValueError(
                f'nnja_gpsro_saids must be "full" or "legacy", got '
                f"{self.nnja_gpsro_saids!r}"
            )
        if self.nnja_max_quality_mark is not None and not (
            0 <= self.nnja_max_quality_mark <= 15
        ):
            raise ValueError(
                "nnja_max_quality_mark must be None or in [0, 15], got "
                f"{self.nnja_max_quality_mark}"
            )
        if self.nnja_wind_dropout and not self.use_nnja_conv:
            raise ValueError("nnja_wind_dropout requires use_nnja_conv=True")
        if self.nnja_surface_pressure_dropout and not self.use_nnja_conv:
            raise ValueError(
                "nnja_surface_pressure_dropout requires use_nnja_conv=True"
            )
        if self.nnja_platform_channel_dropout and not self.use_nnja_sat:
            raise ValueError("nnja_platform_channel_dropout requires use_nnja_sat=True")
        seen_dropout_channels = set()
        for (
            sensor,
            platform,
            channel,
            probability,
        ) in self.nnja_platform_channel_dropout:
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "nnja platform/channel dropout probability must be in [0, 1], "
                    f"got {probability} for {(sensor, platform, channel)}"
                )
            key = (sensor, platform, channel)
            if key in seen_dropout_channels:
                raise ValueError(f"duplicate nnja platform/channel dropout rule: {key}")
            seen_dropout_channels.add(key)


@dataclasses.dataclass(frozen=True)
class ModelSensorConfig:
    sensor_id: int
    nchannel: int
    platform_ids: tuple[int, ...]  # Use tuple since frozen=True

    def __post_init__(self):
        object.__setattr__(self, "platform_ids", tuple(self.platform_ids))


@dataclasses.dataclass(frozen=True)
class SensorEmbedderConfig:
    """Sensor embedding configuration. Used per sensor."""

    embed_dim: int = 32  # initial tokenization dimension
    fusion_dim: int = 512  # sensor fusion dimension
    use_channel_platform_embedding_table: bool = False
    tokenizer_type: str = "concat"  # "concat" or "film"
    use_fused_mlp: bool = False
    features_v2: bool = False  # expanded features

    # Rows in the backbone tokenizer's channel embedding table. Sizes a checkpointed weight.
    max_global_channels: int = 1024

    # FiLM tokenizer embedding widths (obs-type / channel / platform tables).
    obs_type_embed_dim: int = 4
    channel_embed_dim: int = 4
    platform_embed_dim: int | None = None
    film_hidden_dim: int | None = None

    # Backbone per-block observation cross-attention (PixelCrossAttention).
    backbone_pixel_attn_num_heads: int | None = None
    pixel_attn_kv_heads: int = 1
    pixel_attn_head_dim: int = 32

    @property
    def meta_dim(self) -> int:
        if self.features_v2:
            from healda.observations.preprocessing.features_v2 import N_FEATURES
        else:
            from healda.observations.preprocessing.features import N_FEATURES
        return N_FEATURES

    @property
    def tokenizer_meta_dim(self) -> int:
        """Metadata width the FiLM obs tokenizer's first linear receives. Equals
        ``meta_dim``: features_v2 emits the conv/sat-split layout directly."""
        return self.meta_dim


@dataclasses.dataclass
class ModelConfigV1:
    architecture: str = "dit-l_reg_hpx6"
    label_dim: int = 0
    out_channels: int = 1
    condition_channels: int = 0
    obs_hpx_level: int = 6
    time_length: int = 1
    label_dropout: float = 0.0
    legacy_label_bias: bool = (
        False  # For loading old checkpoints with trained label bias
    )

    obs_config: ObsConfig = ObsConfig()

    p_dropout: float = 0.0
    drop_path: float = 0.0
    drop_path_uniform: bool = False
    drop_path_zero_first_n_blocks: int = 0
    group_norm_eps: float = 1e-6
    pos_emb_gains: bool = False

    # dit settings
    dit_temporal_attention: bool = False
    compile_dit: bool = False
    qk_rms_norm: bool = False
    embed_v2: bool = False
    allow_nans_condition: bool = False
    emb_channels: int | None = None
    noise_channels: int | None = None
    """
    the number of channels to use for the noise and label embedding, defaults to 4 * inner_dim.
    """
    as_vit: bool = False  # run DiT without noise/label conditioning
    # Spatial self-attention: "diffusers" (HF split-QKV), "te" (TransformerEngine), or
    # "disk-<order>:<radius_deg>" for a geodesic disk per query. The disk names the operation,
    # not the kernel, so a different implementation can be swapped in without a config change.
    # Old checkpoints spell it "flex-disk-*"; see dit.normalize_attention_backend.
    attention_backend: str = "diffusers"
    # Override DiT ``level_in``. None uses ARCHITECTURES[architecture].level_in.
    level_in: int | None = None
    # False skips building patch_decode entirely, for wrappers that decode the raw latent
    # themselves (see Hpx256LatlonModel with refine_kind="graphcast").
    dense_decoder: bool = True
    # False for wrappers that supply their own calendar at a finer grid (latlon_decode).
    add_calendar: bool = True
    linear_temporal_attention: bool = True
    # Temporal-attention implementation: 0 einsum, 2 CuTe DSL kernels.
    fused_temporal_attn: int = 0
    # When True, linear temporal attention scales by sequence length instead of head
    temporal_attn_legacy_scaling_bug: bool = False
    # Sliding causal lookback (in frames) for temporal attention; None = unbounded.
    temporal_causal_window: int | None = None
    # Per-block cross-attention from pixel latents to packed observation tokens.
    backbone_pixel_attention: bool = False
    # Pixel-attention backend: 1 Triton (baseline), 2 CuTe DSL.
    pixel_attn_impl: int = 1

    # obs encoder settings
    sensor_embedder_config: SensorEmbedderConfig | None = dataclasses.field(
        default_factory=SensorEmbedderConfig
    )
    sensors: dict[str, ModelSensorConfig] | None = None

    def dumps(self):
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def loads(cls, s):
        d = json.loads(s)

        # Filter out fields that aren't in the current model config definition
        d = _filter_to_dataclass_fields(d, cls)

        if isinstance(d.get("obs_config"), dict):
            d["obs_config"] = ObsConfig(
                **_filter_to_dataclass_fields(d["obs_config"], ObsConfig)
            )

        if isinstance(d.get("sensor_embedder_config"), dict):
            embed_cfg = d["sensor_embedder_config"]
            # Backwards compat: old checkpoints had sensors/sensor_config nested inside
            # Move it to top-level ModelConfigV1.sensors
            nested_sensors = embed_cfg.pop("sensors", None) or embed_cfg.pop(
                "sensor_config", None
            )
            if nested_sensors and "sensors" not in d:
                d["sensors"] = nested_sensors
            # Filter to only known fields
            d["sensor_embedder_config"] = SensorEmbedderConfig(
                **_filter_to_dataclass_fields(embed_cfg, SensorEmbedderConfig)
            )

        # Handle sensors dict at top level
        if isinstance(d.get("sensors"), dict):
            d["sensors"] = {
                k: ModelSensorConfig(**v) if isinstance(v, dict) else v
                for k, v in d["sensors"].items()
            }

        return cls(**d)
