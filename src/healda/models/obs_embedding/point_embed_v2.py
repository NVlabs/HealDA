# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import torch

import healda.utils.profiling
from healda.observations.types import (
    UnifiedObservation,
    split_by_sensor,
)
from healda.config.models import SensorEmbedderConfig, ModelSensorConfig
from healda.varlen import lengths_to_idx
from healda.models.obs_embedding.scatter_infill_aggregators import ScatterAggregator

try:
    from healda.models.obs_embedding.fused_film_triton import (
        fused_film_tokenizer_triton,
    )
except (ImportError, OSError):
    fused_film_tokenizer_triton = None


def _prod(shape):
    out = 1
    for s in shape:
        out *= s
    return out


GLOBAL_MAX_CHANNELS = 1024
GLOBAL_MAX_PLATFORM = 1024


def _default_film_hidden_dim(out_dim: int) -> int:
    return out_dim * 2 if out_dim <= 64 else out_dim


class ObsTokenizer(torch.nn.Module):
    """Tokenizes individual observations using metadata + measurement + embedding tables into feature tokens.

    This creates intermediate token representations that will be aggregated and projected to final embeddings.

    TODO(deprecated): legacy "concat" tokenizer. The current models use
    ``ObsTokenizerFiLM`` (``tokenizer_type="film"``). Remove this class and
    ``use_channel_platform_embedding_table`` once the config default and the
    remaining concat loops/tests are migrated off it.

    Args:
        meta_dim: Dimension of static metadata features
        out_dim: Output token dimension
        n_embed: Size of observation type embedding table
        nchannel: Max number of channels (for channel embedding table)
        nplatform: Max number of platforms (for platform embedding table)
        embed_dim: Dimension of observation type embeddings
        use_channel_platform_embedding_table: Use channel and platform embedding tables
    """

    def __init__(
        self,
        meta_dim: int,
        out_dim: int,
        n_embed: int = 1024,
        nchannel: int = 1024,
        nplatform: int = 1024,
        embed_dim: int = 4,
        use_channel_platform_embedding_table: bool = True,
        use_fused_mlp: bool = False,
    ):
        super().__init__()

        if nchannel > GLOBAL_MAX_CHANNELS or nplatform > GLOBAL_MAX_PLATFORM:
            raise ValueError(
                f"nchannel {nchannel} or nplatform {nplatform} is greater than the global max {GLOBAL_MAX_CHANNELS} or {GLOBAL_MAX_PLATFORM}"
            )

        self.use_channel_platform_embedding_table = use_channel_platform_embedding_table
        if self.use_channel_platform_embedding_table:
            self.channel_embedding = torch.nn.Embedding(GLOBAL_MAX_CHANNELS, embed_dim)
            self.platform_embedding = torch.nn.Embedding(GLOBAL_MAX_PLATFORM, embed_dim)
        self.embed_table = torch.nn.Embedding(n_embed, embed_dim)

        mlp_in_dim = (
            1
            + meta_dim
            + embed_dim * (3 if self.use_channel_platform_embedding_table else 1)
        )
        mlp_out_dim = out_dim - 1
        hidden_dim = out_dim * 2 if out_dim <= 32 else out_dim

        self.use_fused_mlp = use_fused_mlp
        self.single_active = nchannel != 8  # conv has 8 channels
        self.meta_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, mlp_out_dim),
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Legacy checkpoints saved a derived platform remap buffer on the tokenizer.
        state_dict.pop(prefix + "platform_id_map", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, obs: UnifiedObservation) -> torch.Tensor:
        """
        Tokenize observations into feature tokens.

        Args:
            obs: UnifiedObservation containing observations and metadata

        Returns:
            (nobs, out_dim)
        """
        obs_type_id = obs.obs_type

        embed_vec = self.embed_table(obs_type_id)
        if self.use_channel_platform_embedding_table:
            channel_emb = self.channel_embedding(obs.local_channel)
            platform_emb = self.platform_embedding(obs.local_platform)

        x_in = torch.cat(
            [
                obs.obs.unsqueeze(-1),
                obs.float_metadata,
                embed_vec,
                *(
                    [channel_emb, platform_emb]
                    if self.use_channel_platform_embedding_table
                    else []
                ),
            ],
            dim=-1,
        )
        mlp_out = self.meta_mlp(x_in)
        encoded = torch.cat([obs.obs.unsqueeze(-1), mlp_out], dim=-1)
        return encoded


class ObsTokenizerFiLM(torch.nn.Module):
    """FiLM-style observation tokenizer: map each scalar observation to a token.

    Each observation is a single scalar measurement (a brightness temperature, a
    PCA latent of one, a wind component, ...) plus metadata describing it
    (location/time features, which instrument channel, which platform). This
    module turns that into an ``out_dim``-vector token, one per observation.

    FiLM (Feature-wise Linear Modulation) keeps the raw measurement as the signal
    and lets the metadata *modulate* it with a per-feature scale and shift::

        conditioning = cat(metadata, obs_type_emb, channel_emb[, platform_emb])
        alpha, beta  = cond_mlp(conditioning).chunk(2)   # 2 * out_dim -> two out_dim vectors
        token        = alpha * obs + beta                # broadcast scalar obs over out_dim

    Rationale: empirically the model leans heavily on the raw measurement and
    largely ignores metadata, so giving metadata a competing slot in a shared
    feature space (the older concat tokenizer) wastes capacity. FiLM preserves the
    strong raw-obs signal while still letting metadata steer it via ``alpha``/``beta``.

    Embedding tables:
      - ``embed_table``: observation *type* embedding (which kind of obs).
      - ``channel_embedding``: instrument *channel* embedding.
      - ``platform_embedding`` (optional): satellite/platform embedding.
    All are looked up per observation and concatenated into the conditioning.

    Separate conv/sat first-linear head: conventional (in-situ / GNSS) and
    satellite observations have different metadata semantics. To let the first
    ``cond_mlp`` linear specialize per family without duplicating the whole MLP, the
    metadata is pre-expanded (on the data side) into
    ``[shared, sat_private, conv_private]`` with the off-family block zeroed -- so a
    single plain first linear effectively learns separate conv/sat weights. The
    expansion is the featurization layer's job (``features_v2.compute_unified_metadata``);
    this module just consumes the wider metadata vector.

    ``use_fused_mlp`` runs the whole tokenizer (embedding gather + conditioning +
    2-layer MLP + FiLM) in a single Triton kernel; the pure-PyTorch path below is a
    readable reference that produces identical results.
    """

    def __init__(
        self,
        meta_dim: int,
        out_dim: int,
        n_embed: int = 1024,
        nchannel: int = 1024,
        nplatform: int = 1024,
        obs_type_embed_dim: int = 4,
        channel_embed_dim: int | None = None,
        platform_embed_dim: int | None = None,
        use_fused_mlp: bool = True,
        use_global_channel_platform_ids: bool = False,
        hidden_dim: int | None = None,
        max_global_channels: int = GLOBAL_MAX_CHANNELS,
    ):
        super().__init__()
        self.out_dim = out_dim
        if channel_embed_dim is None:
            channel_embed_dim = obs_type_embed_dim
        if platform_embed_dim is None:
            platform_embed_dim = 0
        self.use_platform_embedding = platform_embed_dim > 0
        self.obs_type_embed_dim = obs_type_embed_dim
        self.channel_embed_dim = channel_embed_dim
        self.platform_embed_dim = platform_embed_dim
        self.max_global_channels = max_global_channels
        self.embed_table = torch.nn.Embedding(n_embed, obs_type_embed_dim)
        self.channel_embedding = torch.nn.Embedding(
            max_global_channels, channel_embed_dim
        )
        self.platform_embedding = (
            torch.nn.Embedding(GLOBAL_MAX_PLATFORM, platform_embed_dim)
            if self.use_platform_embedding
            else None
        )
        if use_fused_mlp and fused_film_tokenizer_triton is None:
            raise ImportError("use_fused_mlp=True requires triton implementation")
        self.use_fused_mlp = use_fused_mlp
        self.use_global_channel_platform_ids = use_global_channel_platform_ids
        if hidden_dim is None:
            hidden_dim = _default_film_hidden_dim(out_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim

        cond_dim = (
            meta_dim + obs_type_embed_dim + channel_embed_dim + platform_embed_dim
        )

        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2 * out_dim),
        )

    # --- Pure-PyTorch reference path (used when use_fused_mlp=False) -----------
    # The two helpers below + the else-branch of forward() spell out the tokenizer
    # math step by step. With use_fused_mlp=True they are bypassed:
    # fused_film_tokenizer_triton fuses the embedding gathers, the 2-layer
    # conditioning MLP, and the final alpha*obs+beta FiLM into one Triton kernel
    # (no intermediate activations materialized), producing identical results. The
    # reference path is kept for readability, CPU runs, and correctness checks.

    def _channel_and_platform_ids(
        self, obs: UnifiedObservation
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Channel/platform ids are already mapped to local indices in the data
        # transform; the global ids travel alongside.
        channel_ids = (
            obs.global_channel
            if self.use_global_channel_platform_ids
            else obs.local_channel
        )
        platform_ids = (
            None
            if not self.use_platform_embedding
            else (
                obs.global_platform
                if self.use_global_channel_platform_ids
                else obs.local_platform
            )
        )
        return channel_ids, platform_ids

    def _build_conditioning(
        self,
        obs: UnifiedObservation,
        channel_ids: torch.Tensor,
        platform_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        embed_vec = self.embed_table(obs.obs_type)
        chan_emb = self.channel_embedding(channel_ids)
        conditioning_parts = [obs.float_metadata, embed_vec, chan_emb]
        if self.use_platform_embedding:
            if platform_ids is None:
                raise ValueError("platform embedding requires platform ids")
            conditioning_parts.append(self.platform_embedding(platform_ids))
        return torch.cat(conditioning_parts, dim=-1)

    def forward(self, obs: UnifiedObservation) -> torch.Tensor:
        channel_ids, platform_ids = self._channel_and_platform_ids(obs)
        if self.use_fused_mlp:
            return fused_film_tokenizer_triton(
                obs.obs,
                obs.float_metadata,
                obs.obs_type,
                channel_ids,
                platform_ids if self.use_platform_embedding else None,
                self.embed_table,
                self.channel_embedding,
                self.platform_embedding,
                self.cond_mlp[0],
                self.cond_mlp[1],
                self.cond_mlp[3],
                eps=self.cond_mlp[1].eps,
            )

        # Pure-PyTorch reference path (identical results to the fused kernel).
        conditioning = self._build_conditioning(obs, channel_ids, platform_ids)
        ab = self.cond_mlp(conditioning)
        alpha, beta = ab.chunk(2, dim=-1)
        return alpha * obs.obs.unsqueeze(-1) + beta


# Sensor fusion module
class UniformFusion(torch.nn.Module):
    """
    Uniform weighting across all sensors with normalization for number of sensors.

    Simple averaging with 1/sqrt(N) scaling to maintain variance.
    """

    def __init__(self, fusion_dim: int = 256):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.norm = torch.nn.LayerNorm(self.fusion_dim)

    def forward(
        self, projected: torch.Tensor, sensor_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            projected: (num_sensors, ..., fusion_dim)
            sensor_ids: (num_sensors,) - not used, for API consistency
        Returns:
            (..., fusion_dim)
        """
        num_sensors = projected.shape[0]

        projected = self.norm(projected)
        return projected.sum(dim=0) / math.sqrt(num_sensors)


class SensorEmbedder(torch.nn.Module):
    """Unified sensor embedding for any observation source (satellite, conventional, etc.).

    Pipeline:
      1. Per-obs tokenization via MLP
      2. Scatter aggregation
      3. Final projection to output_dim

    Args:
        platform_ids: List of global platform IDs for this sensor
        sensor_embed_dim: Internal feature dimension
        output_dim: Final output dimension of a sensor embedding
        meta_dim: Dimension of static metadata features
        hpx_level: HEALPix grid level
        n_embed: Size of observation type embedding table
        embed_dim: Dimension of observation type embeddings
        nchannel: Max number of channels
        use_checkpoint: Apply gradient checkpointing
    """

    def __init__(
        self,
        platform_ids: list[int],
        sensor_embed_dim: int = 32,
        output_dim: int = 256,
        meta_dim: int = 32,
        hpx_level: int = 6,
        # Embedding table config
        n_embed: int = 1024,  # Large sparse table for observation types
        embed_dim: int = 4,
        nchannel: int = 1024,  # Max channels
        use_checkpoint: bool = False,
        use_channel_platform_embedding_table: bool = True,
        use_fused_mlp: bool = False,
        tokenizer_type: str = "concat",
    ):
        super().__init__()
        nplatform = max(len(platform_ids), 1)

        self.sensor_embed_dim = sensor_embed_dim
        self.output_dim = output_dim
        self.hpx_level = hpx_level
        self.npix = 12 * 4**hpx_level
        self.use_checkpoint = use_checkpoint
        self.nchannel = nchannel
        self.nplatform = nplatform

        if tokenizer_type == "film":
            self.obs_tokenizer = ObsTokenizerFiLM(
                meta_dim=meta_dim,
                out_dim=sensor_embed_dim,
                n_embed=n_embed,
                nchannel=nchannel,
                nplatform=nplatform,
                obs_type_embed_dim=embed_dim,
                use_fused_mlp=use_fused_mlp,
            )
        else:
            self.obs_tokenizer = ObsTokenizer(
                meta_dim=meta_dim,
                out_dim=sensor_embed_dim,
                n_embed=n_embed,
                nchannel=nchannel,
                nplatform=nplatform,
                embed_dim=embed_dim,
                use_channel_platform_embedding_table=use_channel_platform_embedding_table,
                use_fused_mlp=use_fused_mlp,
            )

        # Aggregation setup - outputs (nbatch, npix, output_dim)
        self.scatter_infill_aggregator = ScatterAggregator(
            in_dim=sensor_embed_dim,
            out_dim=output_dim,
            nchannel=nchannel,
            nplatform=nplatform,
            npix=self.npix,
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Legacy checkpoints saved a derived platform remap buffer on the sensor embedder.
        state_dict.pop(prefix + "platform_id_map", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def aggregate(
        self,
        embedded_obs: torch.Tensor,
        obs: UnifiedObservation,
        batch_idx: torch.Tensor,
        nbatch: int,
    ) -> torch.Tensor:
        """
        Aggregate observations to spatial grid and project to output dimension.

        Args:
            embedded_obs: (nobs, sensor_embed_dim) tokenized observations
            obs: UnifiedObservation
            batch_idx: (nobs,) batch index for each obs
            nbatch: product of batch dimensions

        Returns:
            (nbatch, npix, output_dim) aggregated spatial grid
        """
        channel = obs.local_channel
        local_platform = obs.local_platform

        # Build combined bucket ID
        bucket_id = local_platform * self.nchannel + channel
        return self.scatter_infill_aggregator(
            obs_features=embedded_obs,
            batch_idx=batch_idx,
            pix=obs.pix,
            bucket_id=bucket_id,
            nbatch=nbatch,
        )

    def _forward(self, obs: UnifiedObservation):
        batch_dims = obs.batch_dims  # () if offsets is None, (B, T) otherwise
        nbatch = _prod(batch_dims)  # 1 if batch_dims==(), B*T otherwise

        embedded_obs = self.obs_tokenizer(obs)

        if obs.lengths is not None:
            batch_idx = lengths_to_idx(obs.lengths, output_size=obs.obs.shape[0]) % max(
                nbatch, 1
            )
        else:
            batch_idx = torch.zeros(
                obs.obs.shape[0], dtype=torch.long, device=obs.obs.device
            )

        # Aggregator handles empty batches internally to keep all parameters in the computation graph
        output = self.aggregate(
            embedded_obs, obs, batch_idx, nbatch
        )  # (nbatch, npix, output_dim)
        nbatch, npix, output_dim = output.shape

        if len(batch_dims) == 0:
            output = output.view(npix, output_dim)
        else:
            output = output.view(*batch_dims, npix, output_dim)

        return output

    @healda.utils.profiling.nvtx(enabled=False)
    def forward(self, obs: UnifiedObservation) -> torch.Tensor:
        """
        Embed observations from a single sensor onto a spatial grid.

        Args:
            obs: UnifiedObservation for a single sensor. Observations are flattened
                 across batch/time dimensions; `obs.lengths` defines the structure.

        Returns:
            If lengths=None: (npix, output_dim) - single spatial grid in the
                same pixel order used by `obs.pix`
            If lengths present: (*lengths.shape, npix, output_dim) - spatial
                grid in the same pixel order used by `obs.pix`
            e.g., (batch, time, npix, output_dim) if lengths is (batch, time)
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, obs, use_reentrant=False
            )
        else:
            return self._forward(obs)

    def set_spatial_sharding_group(self, group: torch.distributed.ProcessGroup):
        """Set the spatial sharding group for the embedder."""
        self.scatter_infill_aggregator.set_spatial_sharding_group(group)


class MultiSensorObsEmbedding(torch.nn.Module):
    """Multi-sensor observation embedding.

    Args:
        sensor_embedder_config: Config with embedding hyperparameters
        sensors: Dict mapping sensor names to ModelSensorConfig
        hpx_level: HEALPix grid level for all sensors
        use_checkpoint: Apply gradient checkpointing
    """

    def __init__(
        self,
        sensor_embedder_config: SensorEmbedderConfig,
        sensors: dict[str, ModelSensorConfig],
        hpx_level: int,
        use_checkpoint: bool = True,
        compile: bool = True,
    ):
        super().__init__()

        # Store config values
        self.sensors = sensors
        self.sensor_names = list(self.sensors.keys())
        self.sensor_ids = [cfg.sensor_id for cfg in self.sensors.values()]
        self.fusion_dim = sensor_embedder_config.fusion_dim
        self.use_channel_platform_embedding_table = (
            sensor_embedder_config.use_channel_platform_embedding_table
        )
        self.hpx_level = hpx_level
        self.npix = 12 * 4**hpx_level

        embed_cfg = sensor_embedder_config
        # Separate embedders for each sensor.
        self.embedder = torch.nn.ModuleDict(
            {
                str(sensor_cfg.sensor_id): SensorEmbedder(
                    sensor_embed_dim=embed_cfg.embed_dim,
                    meta_dim=embed_cfg.meta_dim,
                    output_dim=self.fusion_dim,
                    hpx_level=self.hpx_level,
                    nchannel=sensor_cfg.nchannel,
                    platform_ids=sensor_cfg.platform_ids,
                    use_checkpoint=use_checkpoint,
                    use_channel_platform_embedding_table=self.use_channel_platform_embedding_table,
                    use_fused_mlp=embed_cfg.use_fused_mlp,
                    tokenizer_type=embed_cfg.tokenizer_type,
                )
                for sensor_cfg in self.sensors.values()
            }
        )

        self.sensor_fusion = UniformFusion(fusion_dim=self.fusion_dim)

        self.output_norm = torch.nn.LayerNorm(self.fusion_dim)

        self.register_buffer(
            "sensor_ids_tensor", torch.tensor(self.sensor_ids, dtype=torch.int32)
        )
        if compile:
            self._forward_inner = torch.compile(
                self._forward_inner, dynamic=True, options={"shape_padding": False}
            )

    def _forward_inner(
        self, obs_by_sensor: dict[int, UnifiedObservation]
    ) -> torch.Tensor:
        sensor_embeddings = []

        for sensor_id_str, embedder in self.embedder.items():
            sensor_id = int(sensor_id_str)
            sensor_obs: UnifiedObservation = obs_by_sensor[sensor_id]
            output = embedder(sensor_obs)  # (b, t, x, c)
            sensor_embeddings.append(output)

        sensor_embeddings = torch.stack(
            sensor_embeddings, dim=0
        )  # (num_sensors, b, t, x, c)

        # Fuse sensors
        num_sensors, b, t, x, c = sensor_embeddings.shape
        sensor_embeddings_flat = sensor_embeddings.view(num_sensors, b * t * x, c)
        fused_flat = self.sensor_fusion(
            sensor_embeddings_flat, self.sensor_ids_tensor
        )  # (b*t*x, fusion_dim)

        out = fused_flat.view(b, t, x, self.fusion_dim)  # (b, t, x, fusion_dim)
        out = self.output_norm(out)
        out = out.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        return out

    def set_spatial_sharding_group(self, group: torch.distributed.ProcessGroup):
        """Set the spatial sharding group for the aggregator."""
        for embedder in self.embedder.values():
            embedder.set_spatial_sharding_group(group)

    @healda.utils.profiling.nvtx(enabled=False)
    def forward(self, obs: UnifiedObservation) -> torch.Tensor:
        if obs.batch_dims is None:
            raise ValueError(
                f"offset batch dimensions must be (batch, time) in MultiSensorObsEmbedding, got {obs.batch_dims}"
            )

        obs_by_sensor = split_by_sensor(obs, self.sensor_ids)
        return self._forward_inner(obs_by_sensor)
