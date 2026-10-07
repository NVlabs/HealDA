# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a trained checkpoint over a date range of local data and write the analysis.

Reads the observations and targets the checkpoint's recipe names from local storage,
loads a checkpoint written by ``healda-train``, and writes the analysis zarr. The
checkpoint's ``loop.json`` supplies the geometry. Launch under ``torchrun -m healda.cli.inference_local``.
"""

import contextlib
import dataclasses
import functools
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.distributed
import xarray as xr
import zarr
from tqdm import tqdm

import healda.utils.distributed as dist
from healda.utils import parsing
from healda.utils.parsing import a, Help
from healda.datasets import samplers
from healda.datasets.da import state_masks, state_transforms
from healda.observations.sensors import conv_var_global_ids
from healda.datasets.da.hourly_latlon_dataset import NLAT as LATLON_NLAT
from healda.datasets.da.hourly_latlon_dataset import NLON as LATLON_NLON
from healda.datasets.da.transform import collate as collate_v2
from healda.datasets.prefetch_map import prefetch_map
from healda.training import distributed_checkpoint
from healda.inference import (
    read_training_loop,
    set_inference_recipe,
    to_physical,
)
from healda.config.models import ObsConfig
from healda.observations.system import (
    DENIALS_CSV,
    ChannelDenial,
    ObsFilters,
    inference_filters,
    report_types_for,
)
from healda.inference_output import (
    AsyncZarrWriter,
    StageTimer,
    open_output_store,
    provenance_attrs,
    record_run,
    store_slots,
)


def scoring_times(z06_z18_inits: bool, time_frequency, split) -> pd.DatetimeIndex:
    year = 2022 if split == "test" else 2021
    start_date = f"{year}-01-01-00" if not z06_z18_inits else f"{year}-01-01-06"
    return pd.date_range(start_date, f"{year}-12-31-12", freq=time_frequency)


def inference_obs_config(trained: ObsConfig, args) -> ObsConfig:
    """The recipe to score with: ``trained``, plus explicit recipe overrides.

    A flag left at its default keeps what training used. Dropped sensors union with the
    trained ones (the sensor list also builds the model's vocabulary); every other removal
    is an extra filter (``inference_filters_from_args``).
    """
    overrides = {
        name: value
        for name, value in (
            ("conv_uv_in_situ_only", args.conv_uv_in_situ_only),
            ("conv_gps_level1_only", args.conv_gps_level1_only),
            ("drop_restricted_aircraft", args.drop_restricted_aircraft),
            ("nnja_balloon_drift", args.balloon_drift),
            ("nnja_withhold_stations", args.withhold_stations),
            ("conv_min_pressure_hpa", args.conv_min_pressure_hpa),
            ("ascat_only_scatterometer", args.ascat_only_scatterometer),
        )
        if value is not None
    }
    if args.drop_sensors:
        overrides["drop_sensors"] = tuple(
            sorted(set(trained.drop_sensors) | set(args.drop_sensors))
        )
    return dataclasses.replace(trained, **overrides)


def inference_filters_from_args(args) -> ObsFilters:
    """``inference_filters`` with the CLI's removals and opt-outs."""
    return inference_filters(
        denials_csv=None if args.no_denials else (args.denials or DENIALS_CSV),
        extra_denials=tuple(
            ChannelDenial(*spec.split(":")) for spec in args.drop_platform_channels
        ),
        drop_source_flagged=not args.keep_source_flagged,
        channel_ids=args.drop_obs_channel_ids,
        report_types=sorted(
            set(args.drop_report_types) | report_types_for(args.drop_obs_families)
        ),
    )


def find_matching_indices(targets, available):
    indices = available.get_indexer(targets)
    valid = indices != -1
    return indices[valid], available[indices[valid]]


def enumerate_to_dict(array):
    return {int(val): i for i, val in enumerate(array)}


def _join(values) -> str:
    return " ".join(map(str, values or ())) or "none"


@dataclasses.dataclass
class InferenceArgs:
    """Command line for this script; see ``--help`` for the rendered text.

    A ``bool | None`` field is three-valued: unset leaves the loop's own setting
    alone, where a plain ``False`` would override it.
    """

    checkpoint_path: a[str, Help("Path to the training checkpoint")] = ""
    output_path: a[str, Help("Output zarr file path")] = "pretrained_output.zarr"
    dataset: a[
        str | None, Help("Registered task to read, overriding the loop's own")
    ] = None
    num_samples: a[int, Help("Number of samples to process (-1 for all)")] = -1
    batch_gpu: a[int, Help("Batch size per GPU")] = 1

    split: a[str, Help("test (2022) or train (2021)")] = "test"
    start_date: a[str | None, Help("First scoring time, YYYY-MM-DD-HH")] = None
    end_date: a[
        str | None,
        Help("Last scoring time, YYYY-MM-DD-HH. Defaults to --start-date's year"),
    ] = None
    time_frequency: a[str, Help("Spacing to sample times from")] = "12h"
    z06_18_inits: a[bool, Help("Use 06z and 18z inits instead of 00z/12z")] = False

    time_length: a[
        int | None,
        Help(
            "Override the loop's frame count, e.g. 16 to extend an 8-frame model. "
            "Temporal attention is RoPE-based, so longer-than-trained windows load fine"
        ),
    ] = None
    time_parallel: a[
        int | None,
        Help(
            "Override the loop's time-parallel size; frames shard time_length/time_parallel per GPU"
        ),
    ] = None
    no_compile: a[bool, Help("Disable torch.compile of the DiT (debugging)")] = False
    fsdp: a[
        bool,
        Help(
            "Enable FSDP parameter sharding. Data must then divide evenly across "
            "data-parallel ranks or inference hangs at the end"
        ),
    ] = False

    conv_uv_in_situ_only: a[
        bool | None, Help("Keep conventional UV channels in-situ only")
    ] = None
    conv_gps_level1_only: a[bool | None, Help("Keep conventional GPS level-1 only")] = (
        None
    )
    balloon_drift: a[
        bool | None,
        Help(
            "Place radiosonde levels at their drifted position and time (XDR/YDR/HRDR)"
        ),
    ] = None
    no_uv_sonde_height_fill: a[
        bool,
        Help(
            "Do not give radiosonde/pibal wind reports (PrepBUFR types 220/221) that "
            "carry no height the standard-atmosphere height of their pressure; the "
            "loader then drops those winds"
        ),
    ] = False
    withhold_stations: a[
        bool | None,
        Help(
            "Drop the held-out radiosonde and land stations (nnja_conventional.is_withheld)"
        ),
    ] = None
    drop_restricted_aircraft: a[
        bool | None,
        Help("Drop NCEP-restricted aircraft (AMDAR/ACARS/TAMDAR) from NNJA PrepBUFR"),
    ] = None
    ascat_only_scatterometer: a[
        bool | None, Help("Treat ASCAT as the only scatterometer")
    ] = None
    conv_min_pressure_hpa: a[
        float | None, Help("Drop conventional obs above this pressure (hPa)")
    ] = None

    drop_report_types: a[
        list[int], Help("PrepBUFR report types to deny, e.g. 240..260 for AMVs")
    ] = dataclasses.field(default_factory=list)
    drop_obs_families: a[
        list[str], Help("ObsFamily names to deny, e.g. amv scatterometer")
    ] = dataclasses.field(default_factory=list)
    drop_platform_channels: a[
        list[str],
        Help(
            "Deny NNJA satellite sensor:platform:channel, e.g. amsua:metop-b:3, always"
        ),
    ] = dataclasses.field(default_factory=list)
    profile: a[
        bool, Help("Print per-rank stage timings and writer emit times (synchronizes)")
    ] = False
    extend: a[
        bool,
        Help(
            "Write into the existing store at --output-path, growing its time axis. "
            "Refused if it holds another checkpoint or observing system"
        ),
    ] = False
    no_denials: a[
        bool, Help("Skip the packaged dated channel denials (observations/denials.csv)")
    ] = False
    denials: a[
        str | None,
        Help("Dated channel denials CSV to apply instead of the packaged one"),
    ] = None
    keep_source_flagged: a[
        bool, Help("Keep CrIS/IASI/ATMS footprints whose provider quality flag is set")
    ] = False
    drop_sensors: a[
        list[str],
        Help(
            "Sensors to exclude from loading (airs-pca iasi-pca cris-fsr-pca) for "
            "leave-one-out sensitivity studies"
        ),
    ] = dataclasses.field(default_factory=list)
    drop_obs_channel_ids: a[
        list[int],
        Help(
            "Global observation channel IDs to exclude; conventional only on NNJA runs, "
            "where satellite channels are denied with --drop-platform-channels"
        ),
    ] = dataclasses.field(default_factory=list)
    drop_conv_vars: a[
        list[str],
        Help(
            "Conventional sub-channels to drop by name/platform "
            "(gps gps_angle gps_t gps_q ps/pres q t u v uv), resolved to global channel "
            "IDs. 'gps' drops all GPS, 'uv' drops u+v"
        ),
    ] = dataclasses.field(default_factory=list)
    keep_obs_dropout: a[
        bool,
        Help(
            "Score with the training-time obs dropout left on, as every dated run did "
            "before 2026-09-17. Only to reproduce an analysis written back then: it "
            "throws away half the wind observations"
        ),
    ] = False
    allow_partial_load: a[
        bool,
        Help(
            "Load the checkpoint non-strictly. Any parameter the checkpoint does not "
            "supply stays randomly initialised and the run still completes, so use this "
            "only when the mismatch is understood"
        ),
    ] = False

    clamp_bounds: a[
        bool,
        Help(
            "Clip bounded channels to their physical range (humidity, tcw, tcwv, sd, swvl "
            ">= 0; sic and cloud cover in [0, 1]); see state_transforms.PHYSICAL_BOUNDS"
        ),
    ] = True
    bf16_output_head: a[
        bool,
        Help(
            "Run the latlon decoder's output projection under bf16 autocast, which "
            "rounds every analysis to bf16 before de-normalisation, as stores written "
            "before 2026-10-04 did"
        ),
    ] = False
    stats_path: a[
        str | None,
        Help(
            "If set, write a netCDF of per-field RMSE+MAE for every channel and "
            "predicted frame, per analysis time (dims: time, field, frame). Enables "
            "full-field timeseries and richer obs-impact analysis. Independent of zarr"
        ),
    ] = None

    no_write_zarr: a[
        bool,
        Help(
            "Skip the analysis zarr; compute and log RMSE only, for metric-only sweeps"
        ),
    ] = False
    zarr_pool_size: a[int, Help("Write-thread count")] = 8
    no_zarr_sharding: a[
        bool,
        Help("Write one object per analysis instead of packing 28 into one"),
    ] = False
    zstd_level: a[
        int,
        Help("Compress the analysis zarr with zstd at this level (0 = raw)"),
    ] = 0
    mantissa_bits: a[
        int,
        Help(
            "Round each analysis value to this many float32 mantissa bits, to nearest, "
            "before it is written (23 = off). Per-time metrics are computed before "
            "rounding"
        ),
    ] = 15

    CHOICES = {"split": ("test", "train")}

    def __post_init__(self):
        if not self.checkpoint_path:
            raise SystemExit("--checkpoint-path is required")
        for name, allowed in self.CHOICES.items():
            if getattr(self, name) not in allowed:
                raise SystemExit(
                    f"--{name.replace('_', '-')} must be one of {', '.join(allowed)}"
                )
        if not 1 <= self.mantissa_bits <= 23:
            raise SystemExit("--mantissa-bits must be in 1..23")
        try:
            pd.Timedelta(self.time_frequency)
        except ValueError:
            raise SystemExit(
                "--time-frequency must be a fixed step such as 6h"
            ) from None


def parse_args() -> InferenceArgs:
    return parsing.parse_args(InferenceArgs)


# Key fields logged per batch (one RMSE line per frame). The stats netCDF
# (--stats-path) records all channels; this is just a readable sanity check.
LOG_FIELDS = [
    "Z500",
    "U100",
    "T100",
    "T850",
    "tas",
    "uas",
    "U500",
    "Q700",
]


def _area_weights(latlon_decode, device):
    """Normalised cos(lat) over the flat spatial axis, or None for an equal-area grid.

    Returned flat and already summing to 1, so `_space_mean` is a single einsum-free
    weighted sum with no per-call normalisation.
    """
    if not latlon_decode:
        return None
    lat = torch.linspace(90.0, -90.0, LATLON_NLAT, device=device, dtype=torch.float32)
    w = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    # The pole rows are a half-cell tall and cos() sends them to ~0 anyway; clamping only
    # guards the float error that can make cos(90 deg) slightly negative.
    w = w[:, None].expand(LATLON_NLAT, LATLON_NLON).reshape(-1)
    return w / w.sum()


def _domain_weights(channels, area_weights, latlon_decode, device):
    """Per-channel spatial weights, area weights restricted to each channel's domain.

    A masked channel carries a constant fill outside its domain, so scoring it over the
    whole grid dilutes the real error by exactly sqrt(domain area fraction). score_latlon.py
    weights by the domain; matching it here keeps the two statistics on one denominator.
    """
    from healda.datasets.da.state_masks import CHANNEL_DOMAIN, state_mask

    if not latlon_decode or not any(c in CHANNEL_DOMAIN for c in channels):
        return None
    rows = []
    for name in channels:
        w = area_weights
        domain = CHANNEL_DOMAIN.get(name)
        if domain is not None:
            mask = torch.as_tensor(
                np.asarray(state_mask(domain)).reshape(-1),
                device=device,
                dtype=torch.float32,
            )
            w = area_weights * mask
            w = w / w.sum()
        rows.append(w)
    return torch.stack(rows)[:, None, :]  # (C, 1, X), broadcasts over (B, C, T, X)


def _space_mean(x, weights):
    """Mean of x over its last (spatial) axis: plain for equal-area, else area-weighted."""
    if weights is None:
        return x.mean(dim=-1)
    return (x * weights).sum(dim=-1)


def _gather_frames(frame_shard, group, mp_size):
    """All-gather a (B, C, T_local) tensor over the time/model-parallel group and
    concat along the frame axis -> (B, C, T_full). No-op when mp_size == 1."""
    if mp_size == 1:
        return frame_shard
    parts = [torch.empty_like(frame_shard) for _ in range(mp_size)]
    torch.distributed.all_gather(parts, frame_shard, group=group)
    return torch.cat(parts, dim=2)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_in_temporary_directory(tmpdir)


def _run_in_temporary_directory(tmpdir):
    args = parse_args()
    # What the checkpoint trained with, not the preset as it reads today.
    loop = read_training_loop(args.checkpoint_path)

    dataset_name = args.dataset or loop.dataset

    # Initialize distributed
    dist.init(timeout_infinite=True)
    dist.print0("Inference configuration:")
    dist.print0(f"  Checkpoint: {args.checkpoint_path}")
    dist.print0(f"  Dataset: {dataset_name}")
    dist.print0(f"  Number of samples: {args.num_samples}")
    dist.print0(f"  Output: {args.output_path}")
    dist.print0(f"  Split: {args.split}")

    # Removals beyond the recipe go into extra filters; the recipe itself is never reduced.
    drop_channel_ids = list(args.drop_obs_channel_ids)
    if args.drop_conv_vars:
        drop_channel_ids += conv_var_global_ids(args.drop_conv_vars)
    args.drop_obs_channel_ids = drop_channel_ids
    loop.obs_config = inference_obs_config(loop.obs_config, args)
    extra_filters = inference_filters_from_args(args)
    if not loop.obs_config.use_nnja_sat:
        # Denials and provider flags act on NNJA satellites; a UFS recipe has none.
        extra_filters = dataclasses.replace(
            extra_filters, channel_denials=(), drop_source_flagged=False
        )

    # Optional temporal-window overrides: run a model over more (or fewer) frames
    # than it was trained on. Safe because temporal attention uses RoPE, not a
    # learned per-frame embedding.
    if args.time_length is not None:
        loop.time_length = args.time_length
    if args.time_parallel is not None:
        loop.time_parallel = args.time_parallel

    # Override settings for inference
    loop.run_dir = tmpdir
    loop.dataset = dataset_name
    loop.batch_size = args.batch_gpu * dist.get_world_size()
    loop.batch_gpu = args.batch_gpu
    loop.dataloader_num_workers = 6
    loop.dataloader_prefetch_factor = 6
    loop.fsdp = args.fsdp
    if args.no_compile:
        loop.compile_dit = False

    # Keep FSDP settings to match checkpoint structure

    # Setup the loop (creates network with proper parallelism/fsdp). The scoring dataset
    # is built below, so setup()'s train/val loaders are unused and their years may be absent.
    loop.setup_datasets = False
    set_inference_recipe(
        loop,
        fill_uv_sonde_heights=not args.no_uv_sonde_height_fill,
        fp32_output_head=not args.bf16_output_head,
    )
    loop.setup()

    dist.print0(f"Loading checkpoint from {args.checkpoint_path}")
    dist.print0(f"  FSDP: {loop.fsdp}")
    dist.print0(f"  Model parallel: {loop.model_parallel}")
    dist.print0(
        f"  Temporal: time_length={loop.time_length} time_parallel={loop.time_parallel} "
        f"(frames/GPU={loop.time_length // loop.time_parallel})"
    )
    dist.print0(f"  Obs config: {loop.obs_config}")

    # Load checkpoint. Strict: require_all=False leaves any unmatched parameter
    # randomly initialised and the run still completes, producing a plausible zarr.
    distributed_checkpoint.load(
        args.checkpoint_path,
        loop.net,
        optimizer=None,
        require_all=not args.allow_partial_load,
    )
    loop.net.eval()

    # Get dataset
    dist.print0(f"Loading {dataset_name} dataset...")
    dataset_train = args.split == "train"
    span_years = None
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None
    if start_date is not None:
        if end_date is None:
            end_date = pd.Timestamp(year=start_date.year, month=12, day=31, hour=12)
        if end_date < start_date:
            raise ValueError(
                f"--end-date ({end_date}) must not be earlier than "
                f"--start-date ({start_date})"
            )
        # Via `years`, not by asking for the training split: that flag is also what turns
        # on the training obs dropout, which scoring must not have.
        span_years = list(range(start_date.year, end_date.year + 1))
        if args.keep_obs_dropout:
            loop.train_years = span_years
            dataset_train = True
    dataset = loop.get_dataset(
        train=dataset_train, years=span_years, extra_obs_filters=extra_filters
    )

    # Compute final times for each window
    dist.print0("Computing final timesteps for windowed dataset...")
    dt = dataset.times[1] - dataset.times[0]
    final_times = dataset.times[: len(dataset)] + (dataset.time_length - 1) * dt
    dist.print0(f"Computed {len(final_times)} final timesteps")

    # Sample times to score
    if start_date is not None:
        scoring_times_all = pd.date_range(
            start_date, end_date, freq=args.time_frequency
        )
    else:
        scoring_times_all = scoring_times(
            args.z06_18_inits, args.time_frequency, args.split
        )

    if args.num_samples != -1:
        min_samples = max(args.num_samples, dist.get_world_size())
        tasks = samplers.subsample(scoring_times_all, min_samples=min_samples)
    else:
        tasks = list(range(len(scoring_times_all)))
    subsampled = scoring_times_all[tasks]

    # Match to dataset indices
    dataset_idx, matched_times = find_matching_indices(subsampled, final_times)
    dist.print0(f"Found {len(dataset_idx)} valid samples out of {len(tasks)} requested")

    dp_size = torch.distributed.get_world_size(group=loop.data_parallel_group)
    if loop.fsdp and loop.time_parallel == 1 and len(dataset_idx) % dp_size != 0:
        raise ValueError(
            f"FSDP+time_parallel=1 needs samples divisible by dp_size or the param "
            f"all-gather hangs: {len(dataset_idx)} % {dp_size} != 0. Drop --fsdp."
        )

    # Split samples across data-parallel replicas: ranks in the same model-parallel
    # group get identical samples (they jointly run one forward); different dp
    # replicas get disjoint samples.
    gpu_tasks = samplers.distributed_split(dataset_idx, group=loop.data_parallel_group)
    dist.print0(f"Tasks: Length: {len(gpu_tasks)} on rank {dist.get_rank()}.")

    # Create mapping from dataset index to output index
    dataset_idx_to_output = enumerate_to_dict(dataset_idx)

    # Create dataloader
    batch_size = min(args.batch_gpu, len(gpu_tasks))
    # Build dataloader kwargs conditionally based on num_workers
    dataloader_kwargs = {
        "dataset": dataset,
        "sampler": gpu_tasks,
        "pin_memory": True,
        "batch_size": batch_size,
        "collate_fn": collate_v2,
        "num_workers": loop.dataloader_num_workers,
    }
    # Only add multiprocessing params if num_workers > 0
    if loop.dataloader_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = loop.dataloader_prefetch_factor
        dataloader_kwargs["multiprocessing_context"] = "spawn"

    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

    # Apply device transform. The same TransformV2 owns the CPU and device stages.
    device_transform = functools.partial(
        loop._device_transform, transform=dataset.transform
    )
    if loop.time_parallel == 1:
        # One GPU holds the whole window: each prefetched batch is ~18.6 GiB at 70M
        # observations, enough to run out of memory with two queued.
        dataloader = map(device_transform, dataloader)
    else:
        dataloader = prefetch_map(dataloader, device_transform, queue_size=2)

    channels = loop.batch_info.channels

    if loop.latlon_decode:
        grid = (("lat", "lon"), (LATLON_NLAT, LATLON_NLON))
    else:
        grid = (("cells",), (49152,))
    # Above dp 1 the interval each replica owns can straddle a 28-deep boundary, so
    # two writers would own one object. Depth 1 is the only safe layout there.
    packed = dp_size == 1 and not args.no_zarr_sharding
    time_chunk = 28 if packed else 1
    dist.print0(f"  Zarr: dp_size={dp_size} time_chunk={time_chunk}")

    group = None
    time_step = pd.Timedelta(args.time_frequency)
    slots = None
    if not args.no_write_zarr:
        if dist.get_rank() == 0:
            provenance = provenance_attrs(loop.obs_config, extra_filters)
            provenance["mantissa_bits"] = args.mantissa_bits
            provenance["clamp_bounds"] = args.clamp_bounds
            group = open_output_store(
                args.output_path,
                channels,
                matched_times,
                time_step,
                extend=args.extend,
                identity={
                    "checkpoint": str(args.checkpoint_path),
                    "obs_config": provenance["obs_config"],
                    "extra_obs_filters": provenance["extra_obs_filters"],
                },
                frames=loop.time_length,
                time_chunk=time_chunk,
                grid=grid,
                zstd_level=args.zstd_level,
                attrs=provenance,
            )

        if dist.get_world_size() > 1:
            torch.distributed.barrier()
        group = zarr.open_group(args.output_path, mode="r+")
        slots = store_slots(group, matched_times, time_step)

    if args.no_write_zarr:
        dist.print0("Skipping zarr output (--no-write-zarr). Starting inference...")
    else:
        dist.print0("Setup output zarr file. Starting inference...")

    # Per-field stats accumulators (rank 0): all channels x all frames, per sample.
    stats_enabled = bool(args.stats_path)
    # Per-time metrics also stream into the store, so a crash keeps them.
    store_metrics = not args.no_write_zarr
    stats_rmse, stats_mae, stats_time = [], [], []

    # Each rank owns the frame shard [mp_rank*T_local : (mp_rank+1)*T_local].
    # mp_last writes the zarr; mp_rank==0 is its dp replica's stats/log writer.
    mp_group = loop.model_parallel_group
    mp_size = loop.model_parallel
    mp_rank = torch.distributed.get_rank(group=mp_group) if mp_size > 1 else 0
    mp_last = mp_rank == mp_size - 1
    dp_rank = torch.distributed.get_rank(group=loop.data_parallel_group)
    dp_size = torch.distributed.get_world_size(group=loop.data_parallel_group)
    if mp_last and not args.no_write_zarr:
        writer = AsyncZarrWriter(
            group,
            channels,
            workers=args.zarr_pool_size,
            profile=args.profile,
            mantissa_bits=args.mantissa_bits,
        )
    else:
        writer = None
    writer_context = (
        contextlib.closing(writer) if writer is not None else contextlib.nullcontext()
    )

    # Inference loop
    with writer_context, torch.no_grad():
        area_weights = _area_weights(loop.latlon_decode, loop.device)
        metric_weights = _domain_weights(
            channels, area_weights, loop.latlon_decode, loop.device
        )
        domain_scored = metric_weights is not None
        if metric_weights is None:
            metric_weights = area_weights

        # The target is loaded in model space, so era5_104ch predicts `tcw - tcwv`;
        # denormalising alone would leave that residual in the zarr.
        phys_channels = list(loop.batch_info.channels)
        phys_config = loop.variable_config.name
        transformed = sorted(state_transforms.TRANSFORMS.get(phys_config, ()))

        # Masked channels are unconstrained by the loss outside their domain, so a
        # model trained with the masked loss gets the loader's fill re-imposed there.
        restore = loop.use_masked_domain_loss
        masked = (
            [c for c in state_masks.CHANNEL_DOMAIN if c in phys_channels]
            if restore
            else []
        )

        dist.print0(
            f"  Physical-space inverse: {phys_config} -> {transformed or 'none'}"
        )
        dist.print0(f"  Domain fill: {', '.join(masked) or 'not re-imposed'}")
        dist.print0(f"  Physical bounds clamp: {'on' if args.clamp_bounds else 'off'}")

        def _to_physical(state):
            return to_physical(
                state, loop, restore_fill=restore, clamp=args.clamp_bounds
            )

        dist.print0(
            f"  Spatial reduction: {'cos(lat)-weighted' if area_weights is not None else 'unweighted (equal-area)'}"
            + (", masked channels scored on their domain" if domain_scored else "")
        )

        timer = StageTimer(args.profile, dist.get_rank())
        for k, batch in enumerate(
            tqdm(dataloader, disable=dist.get_rank() != 0, desc="Inference")
        ):
            timer.mark("data")
            # Extract inputs
            target = batch["target"]
            condition = batch["condition"]
            second_of_day = batch["second_of_day"]
            day_of_year = batch["day_of_year"]
            # Popped, not read: tokenizing releases float_metadata only if nothing else holds it.
            unified_obs = batch.pop("unified_obs", None)
            timestamp = batch["timestamp"]
            labels = batch.get("labels")

            # Flatten (B, C, T, lat, lon) to a single spatial axis so both grids reach the
            # reduction and the writer at one rank. Row-major, matching setup_zarr_output.
            if target.ndim == 5:
                target = target.flatten(3)
            b, c, t, x = target.shape
            noise_labels = torch.zeros([b], device=loop.device)

            # Run model
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unified_obs = loop.net.tokenize_observations(unified_obs)
                prediction = loop.net(
                    condition,
                    noise_labels=noise_labels,
                    class_labels=labels,
                    second_of_day=second_of_day,
                    day_of_year=day_of_year,
                    unified_obs=unified_obs,
                    timestamp=timestamp,
                    is_causal=loop.dit_temporal_attention_causal,
                )
            timer.mark("forward")
            pred = prediction.out
            if pred.ndim == 5:
                pred = pred.flatten(3)

            batch_dataset_indices = gpu_tasks[k * batch_size : (k + 1) * batch_size]
            output_indices = [
                dataset_idx_to_output[int(idx)] for idx in batch_dataset_indices
            ]
            # Inverted here rather than in the writer, which only knows how to
            # denormalise: a residual channel needs a sibling channel to become physical.
            pred_scaled = _to_physical(pred)
            target_scaled = _to_physical(target)
            timer.mark("physical")

            err = pred_scaled - target_scaled  # (B, C, T_local, X)

            # cos(lat)-weighted, since equiangular cells shrink toward the poles. Uniform
            # on equal-area HPX, so both grids report the same physical quantity.
            squared_error = _gather_frames(
                _space_mean(err.pow(2), metric_weights), mp_group, mp_size
            )
            abs_error = (
                _gather_frames(
                    _space_mean(err.abs(), metric_weights), mp_group, mp_size
                )
                if stats_enabled or store_metrics
                else None
            )  # both (B, C, T_full)
            timer.mark("gather")
            if writer is not None:
                writer.submit(
                    [slots[i] for i in output_indices],
                    pred_scaled,
                    metrics=(
                        squared_error.sqrt().cpu().numpy(),
                        abs_error.cpu().numpy(),
                    )
                    if store_metrics
                    else None,
                )
            timer.mark("submit")

            # Per-frame RMSE for a few key fields: sqrt(mean over batch & space).
            # Streaming sanity check; under dp>1 this is rank 0's sample shard only.
            if dist.get_rank() == 0:
                rmse_per_field_frame = squared_error.mean(dim=0).sqrt()  # (C, T)
                log_idx = [channels.index(f) for f in LOG_FIELDS if f in channels]
                host = rmse_per_field_frame[log_idx].detach().cpu()  # one sync
                for row, field in zip(
                    host.tolist(), (f for f in LOG_FIELDS if f in channels)
                ):
                    fmt = ".4e" if field.upper().startswith("Q") else ".4f"
                    vals = " ".join(f"T{ti}={v:{fmt}}" for ti, v in enumerate(row))
                    print(f"RMSE {field} {vals}")

            # Stats: one writer per dp replica accumulates its own disjoint samples.
            if stats_enabled and mp_rank == 0:
                stats_rmse.append(squared_error.sqrt().cpu().numpy())  # (B, C, T)
                stats_mae.append(abs_error.cpu().numpy())
                stats_time.append(
                    np.array(
                        [np.datetime64(matched_times[oi]) for oi in output_indices]
                    )
                )

            # Release this batch before the next one is built (~27 GiB unsharded).
            del batch, condition, target, unified_obs, prediction, pred
            del pred_scaled, target_scaled, err, squared_error, abs_error
            timer.mark("stats")
            timer.step()

    # Every writer drains before rank 0 consolidates.
    if dist.get_world_size() > 1:
        torch.distributed.barrier()

    if dist.get_rank() == 0 and not args.no_write_zarr:
        group = zarr.open_group(args.output_path, mode="r+")
        if "run_id" in group:
            entry = provenance_attrs(loop.obs_config, extra_filters)
            entry.pop("obs_config", None)
            entry.pop("extra_obs_filters", None)
            entry.update(
                first=str(matched_times[0]),
                last=str(matched_times[-1]),
                mantissa_bits=args.mantissa_bits,
            )
            run = record_run(group, slots, entry)
            dist.print0(f"Recorded run {run}: {len(slots)} times")
        zarr.consolidate_metadata(group.store)

    # Write a single per-field stats netCDF. Only mp_rank==0 of each dp replica
    # accumulates stats (its disjoint samples); final all-gather over the dp axis
    # collects every shard on dp_rank 0 before the write.
    if stats_enabled and mp_rank == 0:
        local = (stats_rmse, stats_mae, stats_time)  # lists of per-batch arrays
        if dp_size > 1:
            gathered = [None] * dp_size
            torch.distributed.all_gather_object(
                gathered, local, group=loop.data_parallel_group
            )
        else:
            gathered = [local]

        rmse_chunks = [a for g in gathered for a in g[0]]
        if dp_rank == 0 and rmse_chunks:
            rmse = np.concatenate(rmse_chunks, axis=0)  # (N, C, T)
            mae = np.concatenate([a for g in gathered for a in g[1]], axis=0)
            times = np.concatenate([a for g in gathered for a in g[2]], axis=0)  # (N,)
            n_frames = rmse.shape[2]
            dt_hours = float(dt / np.timedelta64(1, "h"))
            frame_offset = (np.arange(n_frames) - (n_frames - 1)) * dt_hours
            ds = xr.Dataset(
                {
                    "rmse": (("time", "field", "frame"), rmse.astype("float32")),
                    "mae": (("time", "field", "frame"), mae.astype("float32")),
                },
                coords={
                    "time": pd.to_datetime(times),
                    "field": list(channels),
                    "frame": np.arange(n_frames),
                    "frame_offset_hours": ("frame", frame_offset),
                },
            ).sortby("time")
            provenance = provenance_attrs(loop.obs_config, extra_filters)
            ds.attrs.update(
                obs_config=provenance["obs_config"],
                extra_obs_filters=provenance["extra_obs_filters"],
                drop_conv_vars=_join(args.drop_conv_vars),
                split=args.split,
                note="frame -1 == analysis (T7); rmse/mae are per-cell over HPX output grid",
            )
            os.makedirs(
                os.path.dirname(os.path.abspath(args.stats_path)), exist_ok=True
            )
            ds.to_netcdf(args.stats_path)
            print(f"wrote per-field stats -> {args.stats_path} {dict(ds.sizes)}")

    # Final barrier
    dist.print0("Inference completed.")
    if dist.get_world_size() > 1:
        torch.distributed.barrier()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
