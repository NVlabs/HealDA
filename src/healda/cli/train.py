# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train HealPix Data Assimilation model"""

import dataclasses
import functools
import json
import logging
import os
import warnings

import healda.config.environment as config
import healda.config.models
import earth2grid.healpix as H
import healda.models
import healda.utils.profiling
import healda.training.loop
import matplotlib.pyplot as plt
import torch
import torch.distributed
import torch.utils
import torch.utils.data
from healda.utils import distributed as dist
from healda.config.models import (
    ModelSensorConfig,
    ObsConfig,
    SensorEmbedderConfig,
    ir_preset_name,
)
from healda.training.providers import DATASET_PROVIDERS, dataset_provider
from healda.utils.signals import finish_before_quitting
from healda.utils.parsing import parse_args, parse_dict
from healda.datasets.base import BatchInfo, NormalizationStats, TimeUnit, VariableConfig
from healda.datasets.da.tasks import (
    TASK_CONFIGS,
    TrainingTaskConfig,
    build_training_dataset,
    collate,
    get_sensors_for_config,
)
from healda.datasets.da.state_stats import (
    PredictionNormalizations,
    get_batch_info,
    load_channel_stats,
    make_batch_info,
)
from healda.config.variables import VARIABLE_CONFIGS, encode_channels
from healda.observations.system import ObsFilters
from healda.observations.loaders import combined
from healda.observations.sensors import (
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
)
from healda.datasets.da.transform import (
    collate as collate_v2,
    TransformV2,
)
from healda.observations import types
from healda.datasets.prefetch_map import prefetch_map
from healda.datasets.samplers import RestartableDistributedSampler
from healda.training import step as training_step
from healda.training.channel_weights import load_channel_weights
from healda.training import loop
from healda.utils.visualization import visualize
from healda.training import distributed_checkpoint
from healda.models import dit
import itertools

warnings.filterwarnings(
    "ignore",
    message=r"torch\.get_autocast_gpu_dtype",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast",
)


logger = logging.getLogger(__name__)


MAX_CLASSES = 1024


def build_sensor_config(
    sensor_names: list[str], use_nnja_sat: bool = False
) -> dict[str, ModelSensorConfig]:
    """
    Args:
        sensor_names: List of sensor names to include in the model
        use_nnja_sat: Read identities from the NNJA vocabulary rather than the GSI one

    Returns:
        dict mapping sensor_name to ModelSensorConfig
    """
    if use_nnja_sat:
        return {
            name: ModelSensorConfig(
                sensor_id=combined.SENSOR_NAME_TO_ID[name],
                nchannel=combined.channel_count(name),
                platform_ids=combined.platform_ids(name),
            )
            for name in sensor_names
        }
    return {
        name: ModelSensorConfig(
            sensor_id=SENSOR_NAME_TO_ID[name],
            nchannel=SENSOR_CONFIGS[name].channels,
            platform_ids=tuple(
                PLATFORM_NAME_TO_ID[p] for p in SENSOR_CONFIGS[name].platforms
            ),
        )
        for name in sensor_names
    }


@dataclasses.dataclass(frozen=True)
class GradientClipSchedule:
    start_clip: float = 1.0
    end_clip: float = 1.0
    nimg: int = 1


@dataclasses.dataclass(frozen=True)
class LatlonDecode:
    """Decode the backbone latent onto the 0.25 degree grid through GraphCastDecoder.

    Its presence on a loop is what turns the 0.25 degree path on; these are the only
    knobs it has left.
    """

    # Level the bilinear baseline interpolates on. None is the backbone's own level, so
    # the four children of a cell share one parent; the mesh level gives each its own.
    base_level: int | None = 7
    # Nearest mesh nodes per target.
    k: int = 4
    # Width the backbone hands the decoder, not the channel count of the loss target.
    decode_channels: int = 128
    # Run the decoder's output projection outside autocast, in fp32. Inference sets it.
    fp32_output: bool = False


def _migrate_latlon_fields(d: dict) -> dict:
    """Fold a pre-LatlonDecode loop.json onto the current field."""
    if not isinstance(d.get("latlon_decode"), bool):
        return d
    # Flat fields the latlon decode used to carry; loads() would reject the unknown names.
    legacy_fields = (
        "latlon_refine",
        "latlon_refine_k",
        "latlon_refine_base_level",
        "latlon_refine_duplicate_xyz",
        "latlon_refine_hidden",
        "latlon_refine_rounds",
        "decode_channels",
        "geo_statics",
        "lat_harmonics",
    )
    d = dict(d)
    on = d.pop("latlon_decode")
    old = {k: d.pop(k, None) for k in legacy_fields}
    refine = old["latlon_refine"] or "graphcast"
    if on and refine != "graphcast":
        # Raise rather than silently decode through a tail this version no longer has.
        raise ValueError(
            f"checkpoint was trained with latlon_refine={refine!r}, which is no longer "
            "reachable; score it with a build that still has that tail"
        )
    d["latlon_decode"] = (
        LatlonDecode(
            base_level=old["latlon_refine_base_level"],
            k=old["latlon_refine_k"] if old["latlon_refine_k"] is not None else 4,
            decode_channels=(
                old["decode_channels"] if old["decode_channels"] is not None else 128
            ),
        )
        if on
        else None
    )
    return d


@dataclasses.dataclass
class TrainingLoop(loop.TrainingLoopBase):
    """
    valid_samples_per_season: the number of samples to use when making season
        average plots
    """

    regression: bool = False
    valid_min_samples: int = 128

    # loss options
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_min: float = 0.02
    sigma_max: float = 80.0
    noise_distribution: str = "log_normal"
    loss_type: str = "mse"
    huber_delta: float = 0.1
    multistep: bool = False

    # data loader options
    dataloader_num_workers: int = 3
    dataloader_prefetch_factor: int = 8
    prefetch_to_gpu: bool = True

    # parallelism settings
    time_parallel: int = 1
    space_parallel: int = 1
    # Passed to DistributedDataParallel only when set away from these defaults.
    ddp_bucket_cap_mb: int | None = None
    ddp_gradient_as_bucket_view: bool = False
    ddp_static_graph: bool = False
    fsdp: bool = False
    # All-gathers params in bf16 (matching bf16 execution) while still reducing
    # grads in fp32; may reduce FSDP latency.
    fsdp_bf16_params: bool = False
    check_batch: bool = False

    # data options
    dataset: str = "era5_74ch"
    # Options: "full" predicts the normalized target state; "residual" predicts
    # the normalized target-input increment. None uses the task default.
    prediction_mode: str | None = None
    train_years: list[int] | None = (
        None  # Optional: specific years for ERA5 training split
    )
    test_years: list[int] | None = (
        None  # Optional: specific years for ERA5 validation/test split
    )

    ## which datasets to include

    use_labels: bool = False  # lag labels / obs window labels
    label_dropout: float = 0.0
    legacy_label_bias: bool = (
        False  # For loading old checkpoints with trained label bias
    )
    obs_config: ObsConfig = ObsConfig()

    # video model configuration
    time_length: int = 1  # Number of frames per video
    window_stride: int = 1
    time_step: int = 1  # Time step between frames in hours

    # Temporal loss weighting: ratio of first-frame weight to last-frame weight, invariant
    # to the mean-1 normalization. 1.0 = uniform/off. Lower values upweight later frames,
    # which have more causal context and is what matters at inference). Power controls the
    # shape of the curve, where power == 1.0 is linear and power < 1 creates a steep-early/flat-late curve.
    temporal_loss_weight_start: float = 1.0
    temporal_loss_power: float = 1.0

    # Per-channel (pressure-level / per-variable) loss weighting config (from channel_weights.py)
    channel_weight_config: str | None = None

    # Zero the loss (and rescale for the dilution that would otherwise cause) outside
    # each masked channel's valid domain (sst over land, etc. -- state_masks.CHANNEL_DOMAIN),
    # and invert any model-space channel transform (tcw's residual) for physical
    # metrics/visualization. HPX64 grids only today (hpx_loss_rescale_mask).
    use_masked_domain_loss: bool = False

    condition_on_target: bool = False
    condition_on_target_dropout: float = (
        0.5  # probability of an input frame being entirely dropped
    )

    # model configuration
    model_channels: int = 256
    opt: str = "adamw"
    # AdamW kernel impl: "auto" picks per GPU arch (fused on Hopper/sm_9x,
    # foreach on Blackwell/sm_10x where fused AdamW is ~3x slower). Override
    # with "fused" or "foreach" to force.
    adam_impl: str = "auto"
    adam_eps: float = 1e-8
    adam_beta2: float = 0.95
    group_norm_eps: float = 1e-6
    lr_obs: float = 1e-4
    weight_decay: float = 0.1
    weight_decay_biases: bool = True
    drop_path: float = 0.0
    drop_path_uniform: bool = False
    p_dropout: float = 0.0
    architecture: str = "dit-l_reg_hpx6"
    gradient_checkpointing: int = 0
    # 0: gradient_checkpointing applies to every block. N > 0: only the last N.
    gradient_checkpointing_last_n: int = 0
    pos_emb_gains: bool = False
    compile_dit: bool = False  # Enable torch.compile for DiT _forward_DiT
    dit_temporal_attention: bool = True  # if False then full attention
    dit_temporal_attention_causal: bool = True  # if False then full attention
    # Number of frames each frame can attend to (including itself) in temporal
    # attention; only used when dit_temporal_attention_causal=True. None means full context.
    dit_temporal_attention_window: int | None = None
    # Temporal-attention body: 0 einsum, 2 CuTe DSL kernels. Configurations the CuTe DSL kernels
    # cannot serve warn and fall back to einsum.
    fused_temporal_attn: int = 0
    # Pixel-attention backend: 1 Triton (baseline), 2 CuTe DSL.
    pixel_attn_impl: int = 1
    dit_qk_rms_norm: bool = False
    emb_channels: int | None = None
    noise_channels: int | None = None

    # When True, apply a custom gradient clipping schedule that
    # linearly decays the clip value from 1.0 → 0.015 over the
    # first 50k images, then keeps it at 0.015 afterwards.
    gradient_clip_schedule: GradientClipSchedule | None = None

    sensor_embedder_config: SensorEmbedderConfig | None = SensorEmbedderConfig()

    linear_temporal_attention: bool = True  # True to drop softmax in temporal attention
    # Set True to reproduce scaling bug of legacy ckpts
    temporal_attn_legacy_scaling_bug: bool = False

    # Spatial self-attention backend: "diffusers" | "fused_qkv" | "te".
    attention_backend: str = "diffusers"
    # Per-block observation cross-attention (pixel latents attend to packed obs).
    backbone_pixel_attention: bool = False
    drop_path_zero_first_n_blocks: int = 0

    # None keeps the HPX output. Set it to decode to 0.25 degrees, which also implies
    # HPX256 in, a fine calendar, and static surface fields + xyz + solar zenith as
    # conditioning.
    latlon_decode: LatlonDecode | None = None

    finetune_from: str = ""
    finetune_optimizer: bool = False
    freeze_transformer_blocks: bool = False
    freeze_decoder: bool = False
    freeze_spatial: bool = False
    freeze_pos_embedding: bool = False

    # change defaults for parameter norm logging
    log_parameter_norm: bool = False
    log_parameter_grad_norm: bool = False

    # Constant attributes
    grid = H.Grid(6, H.HEALPIX_PAD_XY)

    def __post_init__(self):
        super().__post_init__()
        if self.multistep:
            raise ValueError(
                "The multistep training option has been removed. "
                "Please set multistep=False in your configuration."
            )
        self._device_mesh = None
        self._train_sampler = None
        self._test_sampler = None
        prediction_mode = self.resolved_prediction_mode
        if prediction_mode not in {"full", "residual"}:
            raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")
        task = self.task
        if task is not None:
            task.validate_prediction_mode(prediction_mode)
        elif prediction_mode == "residual":
            raise ValueError("prediction_mode='residual' requires a registered task")

    @property
    def task_name(self) -> str:
        """The registered task this run reads: the ``_latlon`` variant under latlon_decode.

        That variant carries its own statistics, so resolving from the bare name would
        z-score and denormalise the target with two different sets. Not cached: the preset
        builders assign `loop.dataset` after __post_init__ has already read the task.
        """
        if not self.latlon_decode:
            return self.dataset
        latlon = f"{self.dataset}_latlon"
        if latlon not in TASK_CONFIGS:
            # Falling back would read HPX64 targets into a 721x1440 decoder.
            raise ValueError(
                f"latlon_decode needs a '{latlon}' task; {self.dataset!r} has none"
            )
        return latlon

    @property
    def task(self) -> TrainingTaskConfig | None:
        return TASK_CONFIGS.get(self.task_name)

    @property
    def resolved_prediction_mode(self) -> str:
        if self.prediction_mode is not None:
            return self.prediction_mode
        task = self.task
        if task is not None:
            return task.default_prediction_mode
        return "full"

    @property
    def variable_config(self) -> VariableConfig:
        # The recipe is the single source of truth for the state channel layout.
        # The fallback is only for datasets registered outside the task table.
        task = self.task
        if task is not None:
            return VARIABLE_CONFIGS[task.target.variable_config]
        return VARIABLE_CONFIGS["era5_74ch"]

    @functools.cached_property
    def channel_loss_weights(self) -> torch.Tensor | None:
        """Mean-normalized per-channel loss weights (None when disabled)."""
        name = self.channel_weight_config
        if not name or name == "uniform":
            return None
        weights = load_channel_weights(
            name, self.batch_info.channels, self.variable_config
        )
        dist.print0(f"[channel_weights] applying '{name}' to the training loss")
        return torch.from_numpy(weights)

    @property
    def _pipeline_pixel_order(self) -> str:
        """The one pixel order the transform, the packing and the backbone all run in."""
        return dit.pipeline_token_order(self.attention_backend)

    @functools.cached_property
    def _latlon_area_weight(self) -> torch.Tensor:
        """``(1, 1, 1, NLAT, 1)`` cos(lat), normalised to mean 1.

        An equiangular grid oversamples the poles, so an unweighted mean is not global.
        """
        lat = torch.linspace(90.0, -90.0, 721)
        weight = torch.cos(torch.deg2rad(lat)).clamp_min(0)
        return (weight / weight.mean()).view(1, 1, 1, 721, 1)

    @functools.cached_property
    def loss_weights(self) -> torch.Tensor | None:
        return training_step.build_loss_weights(
            self.time_length // self.time_size,
            channel_loss_weights=self.channel_loss_weights,
            weight_start=self.temporal_loss_weight_start,
            time_size=self.time_size,
            time_rank=self.time_rank,
            power=self.temporal_loss_power,
            device=self.device,
        )

    @functools.cached_property
    def channel_mask(self) -> torch.Tensor | None:
        if not self.use_masked_domain_loss:
            return None
        from healda.datasets.da import state_masks

        dist.print0(
            "[channel_mask] applying masked-domain loss (state_masks.CHANNEL_DOMAIN)"
        )
        # Materialise on the training device once instead of repeated H2D copies.
        if self.latlon_decode:
            # Built on the same cos(lat) weight the loss uses, so the valid fraction it
            # divides by matches the reduction. The HPX mask is a different grid and rank.
            mask = state_masks.loss_rescale_mask(
                self.batch_info.channels, self._latlon_area_weight
            )
        else:
            mask = state_masks.hpx_loss_rescale_mask(
                self.batch_info.channels,
                file_name=state_masks.HPX64,
                pixel_order=self._pipeline_pixel_order,
            )
        return mask.to(self.device)

    @property
    def background_enabled(self) -> bool:
        return self.task.has_background if self.task is not None else False

    @property
    def n_background_channels(self) -> int:
        if not self.background_enabled:
            return 0
        task = self.task
        if task is not None:
            return sum(
                len(encode_channels(VARIABLE_CONFIGS[state.variable_config]))
                for state in task.inputs
            )
        return self.out_channels

    def _background_channel_positions(self, input_name: str) -> dict[str, int] | None:
        # The background tensor is every task input's channels concatenated in
        # declaration order. This returns {channel_name: column index in that
        # concatenated background} for the named input
        offset = 0
        for state in self.task.inputs:
            channels = encode_channels(VARIABLE_CONFIGS[state.variable_config])
            if state.name == input_name:
                return {name: offset + i for i, name in enumerate(channels)}
            offset += len(channels)
        return None

    @functools.cached_property
    def residual_base_index(self) -> torch.Tensor | None:
        # Position of each target channel within the concatenated background,
        # located by name in the residual_against input. None = no residual base.
        task = self.task
        if task is None or task.residual_input is None:
            return None
        position = self._background_channel_positions(task.residual_input.name)
        if position is None:
            return None
        target_channels = encode_channels(VARIABLE_CONFIGS[task.target.variable_config])
        missing = [c for c in target_channels if c not in position]
        if missing:
            raise ValueError(
                f"{task.name}: residual input {task.residual_input.name!r} "
                f"missing target channels {missing}"
            )
        return torch.tensor([position[c] for c in target_channels], dtype=torch.long)

    @property
    def background_matches_target(self) -> bool:
        task = self.task
        if task is None or not task.inputs:
            return True
        return all(
            state.variable_config == task.target.variable_config
            for state in task.inputs
        )

    @functools.cached_property
    def _skill_channel_indices(self):
        # (target_pos, background_pos, name) for channels shared by the target and
        # the skill baseline (skill_input, else residual input, else first input).
        # Partial overlap is fine: a baseline need not cover every target channel.
        task = self.task
        if task is None or not task.has_background:
            return []
        baseline_input = task.skill_input or task.residual_input or task.inputs[0]
        bg_index = self._background_channel_positions(baseline_input.name) or {}
        return [
            (ti, bg_index[name], name)
            for ti, name in enumerate(self.batch_info.channels)
            if name in bg_index
        ]

    @functools.cached_property
    def prediction_normalizations(self) -> PredictionNormalizations:
        task = self.task
        if task is not None and task.has_background:
            channels = encode_channels(VARIABLE_CONFIGS[task.target.variable_config])
            input_stats = [
                load_channel_stats(
                    state.stats_file,
                    encode_channels(VARIABLE_CONFIGS[state.variable_config]),
                    state.source,
                )
                for state in task.inputs
            ]
            residual_input = task.residual_input
            residual_source = (
                residual_input.source if residual_input is not None else ""
            )
            residual = (
                load_channel_stats(task.residual_stats_file, channels, residual_source)
                if self.resolved_prediction_mode == "residual"
                else None
            )
            return PredictionNormalizations(
                target=self.target_channel_stats,
                background=NormalizationStats(
                    center=torch.cat(
                        [torch.as_tensor(stats.center) for stats in input_stats]
                    ),
                    scales=torch.cat(
                        [torch.as_tensor(stats.scales) for stats in input_stats]
                    ),
                ),
                residual=residual,
            )
        # No background (obs-only): predict the full target state.
        return PredictionNormalizations(
            target=self.target_channel_stats,
            background=self.target_channel_stats,
            residual=None,
        )

    # Materialised on the training device once instead of repeated H2D copies.
    @functools.cached_property
    def target_channel_stats(self) -> NormalizationStats:
        return self.batch_info.normalization.to(self.device)

    @functools.cached_property
    def background_channel_stats(self) -> NormalizationStats:
        return self.prediction_normalizations.background.to(self.device)

    @functools.cached_property
    def residual_channel_stats(self) -> NormalizationStats | None:
        stats = self.prediction_normalizations.residual
        return None if stats is None else stats.to(self.device)

    @functools.cached_property
    def batch_info(self) -> BatchInfo:
        # batch_info must match the dataset's channel layout so the model's
        # out_channels line up during training and checkpoint loading.
        # ERA5 uses 74 channels including sst/sic.
        task = self.task
        if task is not None:
            return make_batch_info(
                VARIABLE_CONFIGS[task.target.variable_config],
                task.target.stats_file,
                task.target.source,
                time_step=self.time_step,
            )

        if self.dataset in DATASET_PROVIDERS:
            return dataset_provider(self.dataset).get_batch_info()

        return get_batch_info(
            config=self.variable_config,
            time_step=self.time_step,
            time_unit=TimeUnit.HOUR,
        )  # 74 channels

    def resume_from_state(
        self,
        resume_state_dump,
        optimizer=True,
        require_all=True,
        wandb=False,
        iterator_state=True,
    ):
        if not self.fsdp:
            super().resume_from_state(
                resume_state_dump, optimizer, require_all, wandb, iterator_state
            )
            self._load_wandb_id()
        else:
            metadata = distributed_checkpoint.load(
                resume_state_dump,
                self.net,
                self.optimizer if optimizer else None,
                require_all=require_all,
            )
            if iterator_state:
                self.epoch_idx = metadata.get("epoch_idx", 0)
                self.samples_processed_this_epoch_per_rank = metadata.get(
                    "samples_processed_this_epoch_per_rank", 0
                )

            if wandb:
                self.wandb_id = metadata.get("wandb_id")

        if self._train_sampler is not None:
            self._train_sampler.restart(
                self.epoch_idx, self.samples_processed_this_epoch_per_rank
            )

        dist.print0(f"Loaded checkpoint from {resume_state_dump}.")

    def _save_wandb_id(self):
        if self.wandb_id is not None:
            with open(os.path.join(self.run_dir, "wandb_id"), "w") as f:
                f.write(self.wandb_id)

    def _load_wandb_id(self):
        try:
            with open(os.path.join(self.run_dir, "wandb_id")) as f:
                self.wandb_id = f.read()
        except FileNotFoundError:
            pass

    @finish_before_quitting
    def _save(self, path, optimizer=True):
        metadata = dict(
            epoch_idx=self.epoch_idx,
            samples_processed_this_epoch_per_rank=self.samples_processed_this_epoch_per_rank,
            wandb_id=self.wandb_id,
            model_config=json.loads(self.model_config.dumps()),
        )
        distributed_checkpoint.save(
            path,
            self.net,
            optimizer=self.optimizer if optimizer else None,
            metadata=metadata,
        )

    def save_training_state(self, cur_nimg):
        state_filename = self._state_checkpoint_handler.get_path(cur_nimg)
        dist.print0(f"Saving checkpoint to {state_filename}")
        if not self.fsdp:
            if dist.get_rank() != 0:
                return
            self._save_wandb_id()
            return super().save_training_state(cur_nimg)
        else:
            self._save(state_filename, optimizer=True)

        # TODO save wandb id
        dist.print0(f"Checkpoint saved to {state_filename}")

    def save_network_snapshot(self, cur_nimg):
        filename = self._snapshot_checkpoint_handler.get_path(cur_nimg)
        dist.print0(f"Saving network snapshot to {filename}")
        if not self.fsdp:
            if dist.get_rank() != 0:
                return
            super().save_network_snapshot(cur_nimg)
        else:
            self._save(filename, optimizer=False)

    @property
    def out_channels(self):
        return len(self.batch_info.channels)

    @property
    def model_parallel(self) -> int:
        return self.time_parallel * self.space_parallel

    @property
    def batch_gpu_total(self) -> int:
        if self.model_parallel > 1:
            return self.batch_size // self._device_mesh.shape[0]
        else:
            return super().batch_gpu_total

    @property
    def data_rank(self):
        """Rank along the data-parallel dimension (dim 0 of the mesh)."""
        return self._device_mesh.get_local_rank(0)

    @property
    def data_world_size(self):
        """Size of the data-parallel dimension."""
        return self._device_mesh.shape[0]

    @property
    def time_rank(self):
        return self._device_mesh.get_local_rank(1)

    @property
    def time_size(self):
        return self._device_mesh.shape[1]

    @property
    def space_rank(self):
        return self._device_mesh.get_local_rank(2)

    @property
    def space_size(self):
        return self._device_mesh.shape[2]

    @property
    def data_parallel_group(self) -> torch.distributed.ProcessGroup:
        return self._device_mesh.get_group(0)

    @property
    def model_parallel_group(self) -> torch.distributed.ProcessGroup:
        return self._device_mesh.get_group(1)

    @property
    def fsdp_mesh(self):
        # fully_shard (FSDP2) wants a 1D/2D DeviceMesh. With no time parallelism
        # the data group spans the whole world, so shard over that 1D group. With
        # time parallelism the 3D [data, time, space] mesh degenerates to a 2D
        # [data, time] HSDP mesh -- space is an activation-parallel group, not a
        # parameter-sharding dim, so it is excluded here.
        if self.space_parallel > 1:
            raise ValueError(
                "fsdp with space_parallel > 1 is not wired yet: got data/time/space "
                f"mesh {tuple(self._device_mesh.shape)}"
            )
        if self.time_parallel == 1:
            return self._device_mesh["data"]
        return self._device_mesh["data", "time"]

    def setup(self):
        if (
            self.obs_config.nnja_dropout_scope in ("sample", "frame")
            and self.time_parallel > 1
            and not self.fix_time_parallel_rng
        ):
            raise ValueError(
                f'obs dropout_scope="{self.obs_config.nnja_dropout_scope}" needs '
                "fix_time_parallel_rng, or the time-parallel ranks of a sample draw "
                "different dropout verdicts"
            )
        if dist.get_world_size() % self.model_parallel != 0:
            raise ValueError(
                "world size must be divisible by time_parallel * space_parallel: "
                f"{dist.get_world_size()} % {self.model_parallel} != 0"
            )
        # Always build the full [data, time, space] mesh, with sizes collapsing to 1 when dims are unused
        self._device_mesh = torch.distributed.init_device_mesh(
            "cuda",
            [
                dist.get_world_size() // self.model_parallel,
                self.time_parallel,
                self.space_parallel,
            ],
            mesh_dim_names=["data", "time", "space"],
        )
        super().setup()
        # Only wire the space/time groups when those dims are real: a
        # size-1 group would otherwise route single-GPU forward through the
        # shard_t / all_to_all path (dit gates on `_parallel_group is not None`).
        if self.time_parallel > 1:
            self.net.set_time_parallel_group(self._device_mesh.get_group(1))
        if self.space_parallel > 1:
            self.net.set_domain_parallel_group(self._device_mesh.get_group(2))
        if self.model_parallel > 1:
            dist.print0("setting up model parallelism")
            dist.print0(f"{self.batch_gpu_total=}")
        self.net.gradient_checkpointing = self.gradient_checkpointing
        self.net.gradient_checkpointing_last_n = self.gradient_checkpointing_last_n

    @healda.utils.profiling.nvtx
    @finish_before_quitting
    def step_optimizer(self, cur_nimg):
        """Optionally apply a scheduled gradient clipping value, then
        delegate to the base implementation for LR scheduling and optimizer step.
        """
        if self.gradient_clip_schedule is not None:
            sched = self.gradient_clip_schedule
            n = max(0, min(cur_nimg, sched.nimg))
            if sched.nimg > 0:
                frac = n / sched.nimg
            else:
                frac = 1.0

            self.gradient_clip_max_norm = (
                sched.start_clip + (sched.end_clip - sched.start_clip) * frac
            )

        super().step_optimizer(cur_nimg)

    @property
    def _transform_options(self):
        return healda.models.transform_options_for(self.model_config)

    def get_dataset(
        self,
        train: bool,
        years: list[int] | None = None,
        extra_obs_filters: ObsFilters = ObsFilters(),
    ):
        """``years`` picks the span; ``train`` stays free to mean "augment like training".

        Only inference passes ``years`` and ``extra_obs_filters``. ``train_years`` is still honoured when ``train``,
        since an arm sets it for its own split and validates with ``train=False``.
        """
        # Registered gridded-state tasks go through the single factory.
        task = self.task
        if task is not None:
            if years is not None:
                split = years
            elif self.train_years is not None and train:
                split = self.train_years
            else:
                split = "train" if train else "val"
            # task_name, not self.dataset: the same name `task` resolves, so the loader's
            # statistics and the ones used to denormalise cannot diverge.
            dataset_name = self.task_name
            ds = build_training_dataset(
                dataset_name,
                split=split,
                time_length=self.time_length,
                frame_step=self.window_stride,
                model_rank=self.time_rank,
                model_world_size=self.time_size,
                obs_config=self.obs_config,
                transform_options=self._transform_options,
                training=train,
                extra_obs_filters=extra_obs_filters,
            )
            dist.print0(
                f"[setup] dataset={dataset_name!r} split={split} "
                f"-> {type(ds).__name__} (len={len(ds)}); "
                f"target={task.target.name} inputs={[state.name for state in task.inputs]}; "
                f"time_length={self.time_length} frame_step={self.window_stride} "
                f"time_step={self.time_step}h time_rank={self.time_rank}/{self.time_size}; "
                f"use_obs={self.obs_config.use_obs} "
                f"obs_pixel_hpx_level={self._transform_options.observation_hpx_level} "
                f"sensors={get_sensors_for_config(self.obs_config)}"
            )
            return ds

        # Anything not in the task table comes from a registered provider:
        # obs-to-state DA when the config asks for obs, else mask training.
        provider = dataset_provider(self.dataset)
        if extra_obs_filters != ObsFilters():
            raise ValueError(
                f"dataset {self.dataset!r} takes no extra observation filters"
            )

        if self.obs_config.use_obs:
            return provider.ObsDataset(
                time_length=self.time_length,
                frame_step=self.window_stride,
                model_rank=self.time_rank,
                model_world_size=self.time_size,
                obs_config=self.obs_config,
                transform_options=self._transform_options,
                training=train,
            )

        return provider.get_dataset(
            dataset=self.dataset,
            split={True: "train", False: "test"}[train],
            time_length=self.time_length,
            frame_step=self.window_stride,
            model_rank=self.time_rank,
            model_world_size=self.time_size,
            sensors=get_sensors_for_config(self.obs_config),
            obs_context=(
                self.obs_config.context_start,
                self.obs_config.context_end,
            ),
        )

    def _create_dataloader(
        self,
        dataset,
        sampler,
        batch_size,
        num_workers=None,
        prefetch_factor=None,
        pin_memory=True,
    ):
        """Helper to create a DataLoader with common settings."""
        if num_workers is None:
            num_workers = self.dataloader_num_workers
        # Mask training emits legacy batches and needs collate. The dense
        # static+background condition path emits TransformV2 batches, even when
        # observations are disabled for no-obs DA ablations.
        collate_fn = collate if self.mask_training else collate_v2

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            in_order=True,
            drop_last=True,
        )

    def _device_stage_for_dataset(self, dataset):
        # RandomMasker and ObservationMasker already normalize, transpose, and reorder
        # on CPU; their device stage must preserve those values and only move tensors.
        if self.mask_training:
            return self._move_mask_batch_to_device

        # The same TransformV2 owns the CPU and device stages.
        return functools.partial(self._device_transform, transform=dataset.transform)

    def _get_loader(self, dataset, batch_size, train: bool = True):
        workers = self.dataloader_num_workers
        prefetch_factor = self.dataloader_prefetch_factor
        if not train and self.model_parallel == 1 and workers != 0:
            workers = 1
            prefetch_factor = 4

        if isinstance(dataset, torch.utils.data.IterableDataset):
            # Iterable datasets don't use samplers
            loader = self._create_dataloader(
                dataset, sampler=None, batch_size=batch_size
            )

        else:
            # RestartableDistributedSampler supports checkpointing the dataloader
            # position so training resumes where it left off.
            sampler = RestartableDistributedSampler(
                dataset,
                num_replicas=self.data_world_size,
                rank=self.data_rank,
                shuffle=True,
                seed=self.seed,
            )
            sampler.set_epoch(0)
            loader = self._create_dataloader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=workers,
                prefetch_factor=prefetch_factor,
            )
            if self.space_rank > 0:
                loader = itertools.repeat(None)

            # Store sampler reference for checkpointing
            if train:
                self._train_sampler = sampler
            else:
                self._test_sampler = sampler

        # transferring the obs data from cpu -> gpu can be slow, so
        # running it in a separate thread using prefetch_map improves utilization
        # queue_size=1 is sufficient for pipelining, assuming device_transform time is smaller than training time
        if self.prefetch_to_gpu:
            loader = prefetch_map(
                loader,
                self._device_stage_for_dataset(dataset),
                queue_size=1,
            )
        return loader

    def _stage_dict_batch(self, batch):
        return batch

    def _broadcast_batch(self, raw_batch):
        """Broadcast batch tensors across space parallel ranks and select local subdomain."""
        subdomain = None
        if self.space_size > 1:
            space_group = self._device_mesh.get_group(2)
            subdomain = dit.my_subdomain(
                space_group, self.grid.level, "cuda", self.batch_gpu
            )

            for key in ("timestamp", "second_of_day", "day_of_year", "labels"):
                torch.distributed.broadcast(
                    raw_batch[key], group=space_group, group_src=0
                )

            # Spatial fields are broadcast whole, then narrowed to this rank's
            # subdomain. background is optional (no-background recipes omit it).
            spatial_keys = ["condition", "target"]
            if "background" in raw_batch:
                spatial_keys.append("background")
            for key in spatial_keys:
                raw_batch[key] = raw_batch[key].contiguous()
                torch.distributed.broadcast(
                    raw_batch[key], group=space_group, group_src=0
                )
            for key in spatial_keys:
                raw_batch[key] = subdomain.select_from_global(raw_batch[key])

        raw_batch["subdomain"] = subdomain
        return raw_batch

    def _empty_device_batch(self):
        # Nonzero space ranks receive the global batch by broadcast instead of loading it.
        return types.empty_batch(
            batch_gpu=self.batch_gpu,
            out_channels=self.out_channels,
            condition_channels=self.model_config.condition_channels,
            time_length=self.time_length // self.time_parallel,
            x_size=self.grid.shape[0],
            device=self.device,
            background_channels=self.n_background_channels
            if self.background_enabled
            else None,
        )

    def _move_mask_batch_to_device(self, batch: types.Batch | None):
        if batch is None:
            return self._empty_device_batch()
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def _device_transform(self, batch: types.Batch | None, transform: TransformV2):
        """Transformations to occur on device in a separate thread. including device movement"""
        if batch is None:
            return self._empty_device_batch()
        return transform.device_transform(batch, device=self.device)

    def get_data_loaders(self, batch_gpu):
        dataset = self.get_dataset(train=True)
        train_loader = self._get_loader(dataset, batch_size=batch_gpu, train=True)
        test_dataset = self.get_dataset(train=False)
        test_loader = self._get_loader(test_dataset, batch_size=batch_gpu, train=False)

        self._test_dataset = test_dataset
        return dataset, train_loader, test_loader

    def _ensure_batch_consistency(self, timestamp):
        gather_list = [timestamp.clone() for _ in range(self.time_size)]
        # Get the model parallel group from the device mesh (dimension 1)
        model_group = self._device_mesh.get_group(1)
        torch.distributed.all_gather(gather_list, timestamp, group=model_group)
        out = torch.cat(gather_list, dim=1)  # (b,t_local)->(b,t_local*model_world_size)
        if not torch.all(torch.diff(out) == self.time_step * 3600):
            times = out.cpu().numpy().astype("datetime64[s]")
            raise ValueError(
                f"{times} not in sequential order. This means the model parallelism is not setup correctly."
            )

    def _step(
        self,
        *,
        train=True,
        plot_image=False,
        target: torch.Tensor,
        condition,
        second_of_day,
        day_of_year,
        unified_obs=None,
        labels=None,
        background=None,
        return_both=False,
        timestamp,
        subdomain=None,
        **batch,
    ):
        with healda.utils.profiling.nvtx_range("broadcast_batch"):
            broadcast_batch = dict(
                target=target,
                condition=condition,
                timestamp=timestamp,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
                labels=labels,
                subdomain=subdomain,
            )
            if background is not None:
                broadcast_batch["background"] = background
            raw_batch = self._broadcast_batch(broadcast_batch)
            target = raw_batch["target"]
            condition = raw_batch["condition"]
            timestamp = raw_batch["timestamp"]
            second_of_day = raw_batch["second_of_day"]
            day_of_year = raw_batch["day_of_year"]
            labels = raw_batch["labels"]
            background = raw_batch.get("background")
            subdomain = raw_batch["subdomain"]

        # check that all timestamps within model parallel group are in order (corresponding to sharded t)
        if self.check_batch and self.model_parallel > 1:
            self._ensure_batch_consistency(timestamp)

        step_out = training_step.run_model_step(
            batch=training_step.StepInput(
                target=target,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
                timestamp=timestamp,
                unified_obs=unified_obs,
                labels=labels,
                background=background,
                subdomain=subdomain,
            ),
            prediction_mode=self.resolved_prediction_mode,
            target_stats=self.target_channel_stats,
            residual_stats=self.residual_channel_stats,
            background_stats=self.background_channel_stats,
            residual_base_index=self.residual_base_index,
            is_causal=self.dit_temporal_attention_causal,
            huber_delta=self.huber_delta,
            loss_type=self.loss_type,
            loss_weights=self.loss_weights,
            forward_fn=self.ddp,
            channel_mask=self.channel_mask,
            physical_channels=self.batch_info.channels,
            physical_config_name=self.variable_config.name,
            spatial_loss_weight=(
                self._latlon_area_weight.to(target.device)
                if self.latlon_decode
                else None
            ),
        )

        train_tag = "train" if train else "test"

        mse = step_out.mse
        huber_loss = step_out.huber_loss

        self.log_metric(f"Loss/{train_tag}_mse", mse)
        self.log_metric(f"Loss/{train_tag}_huber", huber_loss)

        physical_prediction = step_out.physical_prediction
        physical_target = step_out.physical_target
        # Metrics only, so keep them off the graph.
        error = physical_prediction.detach() - physical_target.detach()
        # channel_mask rescales by 1/valid-fraction, so it enters error and error² once each.
        physical_error = (
            error if self.channel_mask is None else error * self.channel_mask
        )
        physical_mse = error * physical_error

        channels = self.batch_info.channels
        # When time_length > 1 we report metrics for the final time step only;
        # that is the value used as a forecast initial condition.
        is_last_rank = self.time_rank == self.time_size - 1

        # Weighted like the loss, and rank-generic: a fixed dim=(0, 2) reduction assumes
        # one spatial axis and would leave longitude on a (B, C, T, nlat, nlon) error.
        spatial = (
            self._latlon_area_weight.to(physical_mse.device)
            if self.latlon_decode
            else None
        )
        rmse_per_channel = training_step.last_frame_rmse_per_channel(
            physical_mse, spatial
        )
        huber_per_channel = training_step._reduce_weighted(huber_loss, spatial)[
            :, -1
        ].mean(dim=-1)
        # Signed counterpart to rmse_per_channel, through the same reduction so the two
        # are always on one domain: RMSE cannot separate an offset from a wider spread.
        bias_per_channel = training_step._reduce_weighted(physical_error, spatial)[
            :, -1
        ].mean(dim=-1)

        # WARNING: all ranks must call these (the tick flush does a collective
        # reduction); `disable` skips the value update on non-final ranks while
        # still registering the keys so the collective stays balanced.
        self.log_multiple_metrics(
            [f"rmse/{ch}/{train_tag}" for ch in channels],
            rmse_per_channel,
            disable=not is_last_rank,
        )
        self.log_multiple_metrics(
            [f"huber/{ch}/{train_tag}" for ch in channels],
            huber_per_channel,
            should_print=False,
            disable=not is_last_rank,
        )
        self.log_multiple_metrics(
            [f"bias/{ch}/{train_tag}" for ch in channels],
            bias_per_channel,
            should_print=False,
            disable=not is_last_rank,
        )

        # Skill over shared channels: 1 - rmse(prediction) / rmse(background).
        shared_skill_channels = (
            self._skill_channel_indices if background is not None else []
        )
        if shared_skill_channels:
            target_pos, background_pos, names = zip(*shared_skill_channels, strict=True)
            t_idx = torch.tensor(target_pos, device=physical_target.device)
            b_idx = torch.tensor(background_pos, device=physical_target.device)

            physical_background = self.background_channel_stats.denormalize(background)
            background_error = physical_target.index_select(
                1, t_idx
            ) - physical_background.index_select(1, b_idx)
            background_mse = background_error**2
            if self.channel_mask is not None:
                background_mse *= self.channel_mask.index_select(1, t_idx)
            background_rmse = training_step.last_frame_rmse_per_channel(
                background_mse, spatial
            )
            prediction_rmse = rmse_per_channel.index_select(0, t_idx)

            skill_per_channel = 1.0 - prediction_rmse / background_rmse.clamp_min(1e-12)
            self.log_multiple_metrics(
                [f"skill/{name}/{train_tag}" for name in names],
                skill_per_channel,
                should_print=False,
                disable=not is_last_rank,
            )

        if plot_image and is_last_rank and dist.get_rank() == 0:
            # Fewer channels at 0.25 degree; the figures are slow to draw.
            plot_idx = range(len(channels))
            if self.latlon_decode:
                want = {"tas", "uas", "Q700"}
                plot_idx = [i for i, ch in enumerate(channels) if ch in want] or [0]
            for c in plot_idx:
                channel = channels[c]
                # plot the final time step prediction
                for name, field in zip(
                    ["prediction", "target"], [physical_prediction, physical_target]
                ):
                    fig = plt.figure()
                    display_field = field[0, c, -1].cpu()
                    visualize(
                        display_field,
                        hpxpad=not self.latlon_decode,
                        title=channel,
                    )
                    self.writer.add_figure(
                        f"sample/{channel}/{name}", fig, global_step=self.cur_nimg
                    )
        loss = step_out.loss

        if train:
            self.log_metric("loss", loss, frequency="step")

        if return_both:
            return mse, huber_loss
        else:
            return loss

    def train_step(self, **batch):
        return self._step(train=True, **batch)

    def test_step(self, **batch):
        return self._step(train=False, **batch)

    @classmethod
    def loads(cls, s):
        fields = json.loads(s)
        # remove any old fields from some older checkpoints

        # TODO: remove
        # parse_dict type-checks against the current annotation before ObsConfig is built,
        # so a checkpoint's stale value has to be migrated while it is still a dict.
        obs = fields.get("obs_config")
        if isinstance(obs, dict) and "nnja_ir_channels" in obs:
            obs["nnja_ir_channels"] = ir_preset_name(obs["nnja_ir_channels"])

        fields = _migrate_latlon_fields(fields)
        return parse_dict(cls, fields)

    @property
    def mask_training(self) -> bool:
        """Mask training: no real obs; the (NaN-)masked target state is the condition.

        This only applies when there is no background to condition on. Background
        recipes always use the dense static+background condition; there,
        ``use_obs`` only toggles obs ingestion and obs cross-attention.
        """
        if self.background_enabled:
            return False
        return not self.obs_config.use_obs

    @property
    def allow_nans_condition(self) -> bool:
        """Coupled to mask training: only the masked-state condition carries NaNs
        (at unobserved pixels). Real-obs DA uses a dense static condition.
        """
        return self.mask_training

    @property
    def resolved_hpx_in_level(self) -> int:
        """HEALPix level the model ingests at.

        latlon_decode fixes this at 8 regardless of the architecture registry, so read this
        rather than ARCHITECTURES[...].level_in wherever the ingest level actually matters.
        """
        if self.latlon_decode:
            return 8
        return healda.config.models.ARCHITECTURES[self.architecture].level_in

    @property
    def obs_hpx_level(self) -> int:
        """HEALPix level observations are ingested at (== DiT ``level_in``)."""
        return healda.config.models.ARCHITECTURES[self.architecture].level_in

    @property
    def backbone_pixel_attention_hpx_level(self) -> int:
        """HEALPix level the backbone packs obs at; must equal DiT.level_model.

        Distinct from ``obs_hpx_level`` (the obs encoder level): the backbone
        transformer runs at the coarser ``level_model``.
        """
        return healda.config.models.ARCHITECTURES[self.architecture].level_model

    @property
    def effective_dit_temporal_attention(self) -> bool:
        # A single-frame model should use spatial attention only. Leaving the
        # temporal block enabled at T=1 creates a degenerate one-token temporal
        # gate, which is not a meaningful video attention path.
        return self.dit_temporal_attention and self.time_length > 1

    def get_network(self) -> torch.nn.Module:
        net = healda.models.get_model(self.model_config)
        if not self.latlon_decode:
            return net
        from healda.models.hpx256_latlon import Hpx256LatlonModel

        return Hpx256LatlonModel(
            net,
            out_channels=self.out_channels,
            hpx_level=self.resolved_hpx_in_level,
            decode_k=self.latlon_decode.k,
            decode_base_level=self.latlon_decode.base_level,
            decode_fp32_output=self.latlon_decode.fp32_output,
            fine_calendar=True,
            # Derived, not defaulted: the wrapper owns a second copy of the geometry and has
            # to build it in the same order the backbone's tokens are in.
            pixel_order=self._pipeline_pixel_order,
        )

    @property
    def model_config(self) -> healda.config.models.ModelConfigV1:
        out_channels = self.out_channels
        label_dim = MAX_CLASSES if self.use_labels else 0
        temporal_attention = self.effective_dit_temporal_attention

        condition_channels = 0
        if self.latlon_decode:
            # The wrapper builds the condition itself (IFS statics + calendar + geo), and the
            # DiT emits decode_channels for the tail rather than the 0.25 degree loss target.
            from healda.models.hpx256_latlon import CALENDAR_CHANNELS, GEO_CHANNELS
            from healda.datasets.era5_statics import N_ERA5_STATICS

            if self.background_enabled or self.mask_training:
                # Hpx256LatlonModel.forward discards the condition run_model_step assembled,
                # so widening the DiT for it would size pos_embed for a tensor that never
                # arrives; mask training would silently become a no-op.
                raise NotImplementedError(
                    "latlon_decode builds its own condition: background and mask training "
                    "cannot reach the backbone"
                )
            condition_channels = N_ERA5_STATICS + CALENDAR_CHANNELS + GEO_CHANNELS
            out_channels = self.latlon_decode.decode_channels
        elif self.mask_training:
            # mask training: the masked target state is fed in as the condition
            condition_channels += out_channels
        else:
            # obs DA: condition is the static fields; obs enter via cross-attention
            condition_channels += 2  # orog and lfrac static variables
        if self.background_enabled:
            condition_channels += self.n_background_channels

        sensor_embedder_config = None
        sensors_dict = None
        if self.obs_config.use_obs:
            sensor_names = get_sensors_for_config(self.obs_config)
            sensors_dict = build_sensor_config(
                sensor_names, use_nnja_sat=self.obs_config.use_nnja_sat
            )
            sensor_embedder_config = self.sensor_embedder_config
            # Out-of-range global ids index the channel embedding silently, so refuse a
            # table too small for the vocabulary rather than train against wrapped rows.
            # Only the backbone tokenizer keys on global ids; the scatter path embeds
            # per-sensor local ids and sizes no table by max_global_channels.
            if (
                self.obs_config.use_nnja_sat
                and self.backbone_pixel_attention
                and sensor_embedder_config is not None
                and sensor_embedder_config.max_global_channels < combined.NCHANNEL
            ):
                raise ValueError(
                    f"the NNJA vocabulary has {combined.NCHANNEL} channels but "
                    f"max_global_channels is "
                    f"{sensor_embedder_config.max_global_channels}"
                )

        return healda.models.ModelConfigV1(
            architecture=self.architecture,
            condition_channels=condition_channels,
            obs_hpx_level=self.obs_hpx_level,
            out_channels=out_channels,
            time_length=self.time_length if temporal_attention else 1,
            label_dim=label_dim,
            label_dropout=self.label_dropout,
            legacy_label_bias=self.legacy_label_bias,
            obs_config=self.obs_config,
            p_dropout=self.p_dropout,
            drop_path=self.drop_path,
            drop_path_uniform=self.drop_path_uniform,
            group_norm_eps=self.group_norm_eps,
            pos_emb_gains=self.pos_emb_gains,
            dit_temporal_attention=temporal_attention,
            sensor_embedder_config=sensor_embedder_config,
            sensors=sensors_dict,
            qk_rms_norm=self.dit_qk_rms_norm,
            allow_nans_condition=self.allow_nans_condition,
            compile_dit=self.compile_dit,
            emb_channels=self.emb_channels,
            noise_channels=self.noise_channels,
            linear_temporal_attention=self.linear_temporal_attention,
            temporal_attn_legacy_scaling_bug=self.temporal_attn_legacy_scaling_bug,
            temporal_causal_window=self.dit_temporal_attention_window,
            fused_temporal_attn=self.fused_temporal_attn,
            pixel_attn_impl=self.pixel_attn_impl,
            attention_backend=self.attention_backend,
            level_in=self.resolved_hpx_in_level if self.latlon_decode else None,
            # The graphcast tail decodes the raw latent, so patch_decode would be dead weight.
            dense_decoder=not (self.latlon_decode is not None),
            # Hpx256LatlonModel adds the calendar at the fine grid; adding it at the coarse
            # tokens too would condition on time twice.
            add_calendar=not self.latlon_decode,
            backbone_pixel_attention=self.backbone_pixel_attention,
            drop_path_zero_first_n_blocks=self.drop_path_zero_first_n_blocks,
        )

    def _setup_networks(self):
        torch.manual_seed(self.seed)
        net = self.get_network()
        net.train()
        net.requires_grad_(True)
        net.to(self.device)

        if self.freeze_spatial:
            net.requires_grad_(False)
            for name, module in net.named_modules():
                if name.endswith("temporal_attn"):
                    dist.print0(f"Unfreezing {name}")
                    module.requires_grad_(True)

        if self.freeze_transformer_blocks:
            for block in net.transformer_blocks:
                block.requires_grad_(False)

        if self.freeze_pos_embedding:
            net.pos_embed.pos_embed.requires_grad_(False)

        if self.freeze_decoder:
            if net.patch_decode is None:
                raise ValueError(
                    "freeze_decoder needs a patch decoder, but dense_decoder=False built "
                    "none; the graphcast tail is the decoder on this arm."
                )
            net.patch_decode.requires_grad_(False)

        self.net = net
        if self.fsdp:
            mp_policy = None
            if self.fsdp_bf16_params:
                from torch.distributed.fsdp import MixedPrecisionPolicy

                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                )
            net.fully_shard(self.fsdp_mesh, mp_policy=mp_policy)
            self.ddp = self.net
        elif dist.get_world_size() > 1:
            ddp_kwargs = {}
            if self.ddp_bucket_cap_mb is not None:
                ddp_kwargs["bucket_cap_mb"] = self.ddp_bucket_cap_mb
            if self.ddp_gradient_as_bucket_view:
                ddp_kwargs["gradient_as_bucket_view"] = True
            if self.ddp_static_graph:
                ddp_kwargs["static_graph"] = True
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.device],
                broadcast_buffers=False,
                **ddp_kwargs,
            )
        else:
            self.ddp = self.net

    def get_optimizer(self, named_parameters):
        named_params = list(named_parameters)
        # Separate the obs embedding and transformer parameters to apply
        # lower learning rate to the obs embedding
        obs_param_prefix = "embed_v2_patch"

        def _get_param_groups(params, lr):
            if self.weight_decay_biases:
                return ({"params": params, "lr": lr, "base_lr": lr},)

            weights, biases = [], []

            for param in params:
                if param.ndim > 1:
                    weights.append(param)
                else:
                    biases.append(param)

            return [
                {
                    "params": weights,
                    "lr": lr,
                    "base_lr": lr,
                    "weight_decay": self.weight_decay,
                },
                {"params": biases, "lr": lr, "base_lr": lr, "weight_decay": 0.0},
            ]

        xfmr_params, obs_params = [], []
        for name, param in named_params:
            if name.startswith(obs_param_prefix):
                obs_params.append(param)
            else:
                xfmr_params.append(param)

        param_groups = []
        if xfmr_params:
            param_groups.extend(_get_param_groups(xfmr_params, self.lr))
        if obs_params:
            param_groups.extend(_get_param_groups(obs_params, self.lr_obs))

        adam_kwargs = self._select_adam_kwargs()
        if self.opt == "adamw":
            return torch.optim.AdamW(
                param_groups,
                betas=(0.9, self.adam_beta2),
                eps=self.adam_eps,
                weight_decay=self.weight_decay,
                **adam_kwargs,
            )
        else:
            return torch.optim.Adam(
                param_groups,
                betas=(0.9, self.adam_beta2),
                eps=self.adam_eps,
                **adam_kwargs,
            )

    def _select_adam_kwargs(self):
        choice = self.adam_impl
        if choice == "foreach":
            return {"foreach": True}
        # fused is now faster on B300 too.
        return {"fused": True}

    def get_loss_fn(self):
        return None

    @staticmethod
    def print_network_info(net, device):
        num_params = sum(p.numel() for p in net.parameters())
        logger.info(f"Number of parameters: {num_params}")

    def validate(self, net=None):
        if self.valid_min_samples <= 0:
            return
        # show plots for a single batch
        if net is None:
            net = self.net
        net.eval()

        for batch_num, batch in enumerate(self.valid_loader):
            if batch_num * self.batch_size >= self.valid_min_samples:
                break
            batch = self._stage_dict_batch(batch)
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
                ):
                    self.test_step(plot_image=batch_num == 0, return_both=True, **batch)


@dataclasses.dataclass
class CLI:
    name: str = ""
    output_dir: str = config.CHECKPOINT_ROOT
    finetune_from: str = ""
    resume_dir: str = ""
    loop: TrainingLoop = dataclasses.field(
        default_factory=lambda: TrainingLoop(
            total_ticks=250,
            steps_per_tick=1200,
            state_dump_ticks=1,
            snapshot_ticks=None,  # don't save snapshots
            batch_size=32,
            batch_gpu=4,
            lr=1e-4,
            lr_min=0.0,
            lr_rampup_img=32 * 1000,
            flat_imgs=0,
            decay_imgs=300_000 * 32,
            opt="adamw",
            dataloader_num_workers=8,
            dataloader_prefetch_factor=100,
            gradient_clip_max_norm=1.0,
        )
    )


warnings.filterwarnings(action="ignore", message="Cannot do a zero-copy NCHW to NHWC.")


class _Loops(dict):
    """Arm registry. Raises on a duplicate name rather than silently overwriting.

    An arm name ties a checkpoint, a wandb run and an analysis directory together, so
    two definitions of one name is never what was meant.
    """

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"training loop {key!r} is already registered")
        super().__setitem__(key, value)


LOOPS = _Loops()


# UFS-replay observations, HPX64 ERA5 target.
UFS_HPX64 = "v2-videoDA-convMinP1hPa-plevelNorm-plevelConvChannels"
LOOPS[UFS_HPX64] = TrainingLoop(
    fix_time_parallel_rng=True,
    architecture="dit-5B",
    attention_backend="te",
    backbone_pixel_attention=True,
    compile_dit=True,
    dit_qk_rms_norm=True,
    emb_channels=128,
    noise_channels=128,
    time_length=8,
    time_step=6,
    time_parallel=8,
    fsdp=False,
    batch_size=4,
    batch_gpu=1,
    lr=2e-4,
    lr_min=0.0,
    flat_imgs=0,
    decay_imgs=3_000_000,
    loss_type="huber",
    weight_decay=0.05,
    weight_decay_biases=False,
    drop_path=0.1,
    drop_path_uniform=True,
    drop_path_zero_first_n_blocks=4,
    gradient_clip_max_norm=1.0,
    gradient_clip_schedule=GradientClipSchedule(end_clip=0.015, nimg=50_000),
    total_ticks=375,
    steps_per_tick=2000,
    snapshot_ticks=None,
    state_dump_ticks=1,
    valid_min_samples=32,
    dataloader_num_workers=8,
    dataloader_prefetch_factor=10,
    obs_config=ObsConfig(
        use_obs=True,
        context_start=-3,
        use_infrared_pca=True,
        use_airs_pca=True,
        use_conv=True,
        conv_uv_in_situ_only=True,
        conv_gps_level1_only=True,
        conv_min_pressure_hpa=1.0,
        use_conv_level_stats=True,
        conv_level_channels=True,
        drop_obs_channel_ids=[],
    ),
    sensor_embedder_config=SensorEmbedderConfig(
        tokenizer_type="film",
        features_v2=True,
        use_fused_mlp=True,
        use_channel_platform_embedding_table=True,
        channel_embed_dim=16,
        platform_embed_dim=8,
        backbone_pixel_attn_num_heads=64,
        pixel_attn_kv_heads=2,
    ),
)

# NNJA obs, 104-channel HPX64 ERA5 target.
NNJA_HPX64 = "v2-videoDA-nnja-nnjaConv-104ch-windDrop50"
LOOPS[NNJA_HPX64] = dataclasses.replace(
    LOOPS[UFS_HPX64],
    dataset="era5_104ch",
    channel_weight_config="step-cloudLandW-10hpa",
    use_masked_domain_loss=True,
    architecture="dit-5B-d128",
    time_parallel=4,
    fsdp=False,
    fix_time_parallel_rng=False,
    state_dump_ticks=2,
    dataloader_num_workers=4,
    gradient_clip_schedule=GradientClipSchedule(end_clip=0.02, nimg=50_000),
    obs_config=dataclasses.replace(
        LOOPS[UFS_HPX64].obs_config,
        use_nnja_sat=True,
        use_nnja_conv=True,
        use_nnja_satwnd=True,
        conv_uv_in_situ_only=False,
        # Training-only dropout: 50% of ASCAT, aircraft and SATWND wind rows, and 20% of
        # MetOp-B AMSU-A channels 3 and 6.
        nnja_wind_dropout=0.5,
        nnja_platform_channel_dropout=(
            ("amsua", "metop-b", 3, 0.2),
            ("amsua", "metop-b", 6, 0.2),
        ),
        ascat_only_scatterometer=True,
    ),
    sensor_embedder_config=dataclasses.replace(
        LOOPS[UFS_HPX64].sensor_embedder_config,
        max_global_channels=2048,
    ),
)

# 0.25 degree model: 18-layer backbone with a lat-lon decode.
NNJA_LATLON = "v2-videoDA-nnja-nnjaConv-104ch-windDrop50-latlon025-18L-baseL7-aga-tp8-dp018-obsAll"
LOOPS[NNJA_LATLON] = dataclasses.replace(
    LOOPS[NNJA_HPX64],
    architecture="dit-5B-d128-18L-lev6",
    latlon_decode=LatlonDecode(),
    attention_backend="disk-nest:10.46",
    time_parallel=8,
    drop_path=0.1818,
    fused_temporal_attn=2,
    pixel_attn_impl=2,
    ddp_bucket_cap_mb=100,
    ddp_gradient_as_bucket_view=True,
    dataloader_prefetch_factor=3,
    state_dump_ticks=4,
    gradient_clip_schedule=GradientClipSchedule(end_clip=0.025, nimg=50_000),
    obs_config=dataclasses.replace(
        LOOPS[NNJA_HPX64].obs_config,
        nnja_gpsro_saids="full",
        nnja_surface_winds=True,
        ascat_only_scatterometer=False,
    ),
)
# TODO: increase NNJA conv quality mark to 3

# Short names.
LOOPS["v2-ufs-hpx64"] = LOOPS[UFS_HPX64]
LOOPS["v2-nnja-hpx64"] = LOOPS[NNJA_HPX64]
LOOPS["v2-nnja-latlon-final"] = LOOPS[NNJA_LATLON]

# End-to-end single-GPU training smoke test.
LOOPS["debug-single-gpu"] = dataclasses.replace(
    LOOPS[NNJA_HPX64],
    architecture="dit-5B-d128-2L",
    fsdp=False,
    time_parallel=1,
    space_parallel=1,
    time_length=1,
    batch_size=1,
    gradient_checkpointing=0,
    dataloader_num_workers=0,
    compile_dit=False,
    compile_optimizer=False,
    print_steps=1,
    total_ticks=1,
    steps_per_tick=50,
    state_dump_ticks=1,
    valid_min_samples=1,
    test_with_single_batch=True,
)


SMOKE_SUFFIX = "-smoke"


def as_smoke(loop: TrainingLoop) -> TrainingLoop:
    """100 steps of a production config, otherwise unchanged."""
    world = dist.get_world_size()
    # setup() wants world divisible by time_parallel * space_parallel, so shrink to the
    # largest time_parallel that still divides rather than to the world size alone.
    time_parallel = min(loop.time_parallel, max(1, world // loop.space_parallel))
    while time_parallel > 1 and world % (time_parallel * loop.space_parallel):
        time_parallel -= 1
    checkpointing = {}
    if time_parallel < loop.time_parallel:
        # Fewer time ranks, so more frames on each: pay the activation memory back.
        # Both fields, or the last_n selection has nothing to apply.
        checkpointing = dict(gradient_checkpointing=1, gradient_checkpointing_last_n=5)
    return dataclasses.replace(
        loop,
        time_parallel=time_parallel,
        batch_size=max(1, world // (time_parallel * loop.space_parallel)),
        batch_gpu=1,
        **checkpointing,
        total_ticks=2,
        steps_per_tick=50,
        print_steps=5,
        snapshot_ticks=None,
        state_dump_ticks=None,
    )


def main():
    cli = parse_args(CLI, convert_underscore_to_hyphen=False)
    dist.init()

    if dist.get_rank() == 0:
        logging.basicConfig(level=logging.INFO)
        healda.training.loop.logger.setLevel(level=logging.DEBUG)

    name = cli.name.removesuffix(SMOKE_SUFFIX)
    try:
        dist.print0(f"Using {name=} preset.")
        loop = LOOPS[name]
    except KeyError:
        dist.print0("Using --loop command line arguments")
        loop = cli.loop

    if cli.name.endswith(SMOKE_SUFFIX):
        loop = as_smoke(loop)
        dist.print0(f"Smoke run: {loop.batch_size=} {loop.time_parallel=}")

    loop.run_dir = os.path.join(cli.output_dir, cli.name)
    if cli.finetune_from:
        loop.finetune_from = cli.finetune_from
    loop.setup()
    dist.print0("Training with:", loop)

    if dist.get_rank() == 0:
        config.print_config()

    new_training = True
    # attempt resuming from output-dir, and then try the resume_dir CLI
    # this behavoir makes it easy to submit multiple segments of the run using
    # the same CLI arguments
    resume_dirs_in_priority = [loop.run_dir, cli.resume_dir]
    for rundir in resume_dirs_in_priority:
        try:
            loop.resume_from_rundir(rundir, require_all=False)
            new_training = False
            break
        except FileNotFoundError:
            pass

    if new_training and loop.finetune_from:
        loop.resume_from_state(
            loop.finetune_from,
            optimizer=loop.finetune_optimizer,
            require_all=False,
            iterator_state=False,
        )

    if new_training:
        loop.wandb_id = None
        dist.print0("Starting new training")

        # TODO save cur nimg in the checkpoint
        try:
            loop.cur_nimg = loop._cur_nimg_start
        except AttributeError:
            pass

    loop.setup_wandb(name=cli.name)
    loop.train()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
