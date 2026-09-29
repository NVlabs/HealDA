# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import datetime
import functools
import healda.utils.profiling
import cftime
from earth2grid import healpix

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from healda.datasets.base import (
    NormalizationStats,
    VariableConfig,
)
from healda.datasets.da import state_transforms
from healda.observations.preprocessing import features, features_v2
from healda.observations.loaders import threads
from healda.observations import types
from healda.datasets.da.state_stats import (
    get_batch_info,
)
from healda.config.variables import encode_channels
from healda.observations.sensors import (
    NPLATFORMS,
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
)
from healda.config.variables import VARIABLE_CONFIGS
from healda.datasets import static_data
from healda.observations.packing import pack_observations_by_pixel
import warnings

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors",
)

_GLOBAL_MAX_PLATFORM = 1024

# Obs table column -> key in the tensor dict the model consumes.
_OBS_TENSOR_COLUMNS = {
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Observation": "observation",
    "Global_Channel_ID": "global_channel_id",
    "Sat_Zenith_Angle": "sat_zenith_angle",
    "Sol_Zenith_Angle": "sol_zenith_angle",
    "local_channel_id": "local_channel_id",
    "Height": "height",
    "Pressure": "pressure",
    "Scan_Angle": "scan_angle",
    "Absolute_Obs_Time": "absolute_obs_time",
    "Platform_ID": "platform_id",
    "Observation_Type": "observation_type",
}

# Integer columns, where a null cannot be carried as NaN.
_OBS_FILL_ZERO = frozenset({"Platform_ID", "Observation_Type"})

# Column names required by the encode function
ENCODE_REQUIRED_COLUMNS = [
    "Latitude",
    "Longitude",
    "Absolute_Obs_Time",
    "Platform_ID",
    "Observation_Type",
    "Observation",
    "Global_Channel_ID",
    "Sat_Zenith_Angle",
    "Sol_Zenith_Angle",
    "sensor_id",
    "local_channel_id",
]

# Optional column names that are checked for existence in encode function
ENCODE_OPTIONAL_COLUMNS = [
    "Height",
    "Pressure",
    "Scan_Angle",
]

# All column names (required + optional)
ENCODE_ALL_COLUMNS = ENCODE_REQUIRED_COLUMNS + ENCODE_OPTIONAL_COLUMNS


def _cftime_to_timestamp(time: cftime.DatetimeGregorian) -> float:
    return datetime.datetime(
        *cftime.to_tuple(time), tzinfo=datetime.timezone.utc
    ).timestamp()


def _frames_are_consecutive_slices(frames: list[np.ndarray]) -> bool:
    """Whether `frames` are `base[0..n-1]`, in order, of one C-contiguous `frames[0].base`.

    The DA datasets read a sample's state as one (t, c, x) array and hand out `state[t]` per
    frame, so stacking them rebuilds a buffer that already exists, wasting 40ms at HPX256.
    """
    first = frames[0]
    # An array that owns its buffer has no `base`, so there is nothing to reuse.
    if not isinstance(first.base, np.ndarray) or not first.base.flags["C_CONTIGUOUS"]:
        return False
    base = first.base
    # `base` must be exactly the stack: one slice per frame, each laid out as that slice.
    if base.shape != (len(frames),) + first.shape or first.strides != base.strides[1:]:
        return False
    # Same buffer and shape still allows repeats or a shuffle, so check each frame's
    # address, and its strides, since a transposed view starts in the right place.
    starts = (base.ctypes.data + index * first.nbytes for index in range(len(frames)))
    return all(
        frame.shape == first.shape
        and frame.strides == first.strides
        and frame.ctypes.data == start
        for frame, start in zip(frames, starts)
    )


_PIXEL_ORDERS = {
    "hpxpadxy": healpix.HEALPIX_PAD_XY,
    "nest": healpix.NEST,
}


def reorder_from_nest(x, target_order: str):
    x = torch.as_tensor(x)
    target = _PIXEL_ORDERS[target_order]
    if target is healpix.NEST:
        return x
    return healpix.reorder(x, healpix.NEST, target)


def _compute_second_of_day(time: cftime.datetime):
    day_start = time.replace(hour=0, minute=0, second=0)
    return (time - day_start) / datetime.timedelta(seconds=1)


def _compute_day_of_year(time: cftime.datetime):
    day_start = time.replace(hour=0, minute=0, second=0)
    year_start = day_start.replace(month=1, day=1)
    return (time - year_start) / datetime.timedelta(seconds=86400)


def _compute_timestamp(time: cftime.datetime):
    return int(_cftime_to_timestamp(time))


def zscore_static(values) -> torch.Tensor:
    """Normalize a static field by its own spatial mean/std"""
    x = torch.as_tensor(values).float()
    return (x - x.mean()) / x.std()


def fixed_static_condition(
    variable_config,
    hpx_level: int = 6,
    source: str = "ufs",
    pixel_order: str = "hpxpadxy",
) -> torch.Tensor:
    """Static condition from a fixed static source, ``ufs`` or the ERA5 invariants.

    orog + land fraction, z-scored, as a ``(1, C, 1, X)`` tensor in ``pixel_order``.
    Packed-store datasets build their condition from the target store instead.
    """
    land = static_data.load_lfrac(hpx_level, source=source)
    # Packed stores spell land fraction land_surface; same field.
    fields = {
        "orog": static_data.load_orography(hpx_level=hpx_level, source=source),
        "lfrac": land,
        "land_surface": land,
    }
    arrays = [zscore_static(fields[name]) for name in variable_config.variables_static]
    condition = torch.stack(arrays).float().unsqueeze(1).unsqueeze(0)  # (1, c, 1, x)
    return reorder_from_nest(condition, pixel_order)


def _map_platform_to_local(
    platform: torch.Tensor,
    lengths: torch.Tensor,
    lut_matrix: torch.Tensor,
) -> torch.Tensor:
    """Map global platform IDs to sensor-local IDs."""
    if lut_matrix.numel() == 0:
        return torch.zeros_like(platform)

    # lengths is (S, B, T), platform is (N,), and obs rows are sensor-major.
    n_sensors, n_platforms = lut_matrix.shape  # lut_matrix: (S, P)
    counts = lengths.reshape(n_sensors, -1).sum(dim=1).to(torch.long)  # (S,)
    sensor_idx = torch.repeat_interleave(
        torch.arange(n_sensors, device=platform.device),
        counts,
        output_size=platform.numel(),
    )  # (N,)
    flat = lut_matrix.reshape(-1)  # (S * P,)
    idx = sensor_idx * n_platforms + platform.long().clamp(0, n_platforms - 1)  # (N,)
    return flat[idx].reshape(platform.shape)  # (N,) local platform IDs


class _DeviceCache:
    """Device copies of run-constant transform tensors."""

    __slots__ = ("_entries",)

    def __init__(self):
        self._entries = {}

    def get(self, key, build):
        value = self._entries.get(key)
        if value is None:
            value = self._entries[key] = build()
        return value

    def __reduce__(self):
        return (_DeviceCache, ())


@dataclasses.dataclass(frozen=True)
class TransformOptions:
    # HEALPix level used to assign each observation's pixel id. Backbone attention
    # requires this to match the model token level.
    observation_hpx_level: int = 6
    features_v2: bool = False
    attention_prepack: bool = False
    build_attention_group_map: bool = True
    pixel_order: str = "hpxpadxy"


@dataclasses.dataclass
class TransformV2:
    """Batch transform for normalizing state data and preparing observations for training.

    Two-stage pipeline:
        1. ``transform(times, frames)`` - CPU preprocessing, returns intermediate dict
        2. ``device_transform(batch, device)`` - GPU transfer and featurization

    Stage 1 - ``transform()`` returns dict with:
        - ``target``: Model-space state tensor (B, T, C, X).
        - ``unified_obs``: Tuple of (obs_tensors, lengths_3d).
        - ``condition``: Static conditioning features (1, C_cond, 1, X).
        - ``second_of_day``, ``day_of_year``, ``timestamp``: Time encodings (B, T).

    Intermediate ``unified_obs`` tuple structure:
        - ``obs_tensors``: Dict of 1D tensors (N_obs,) - latitude, longitude,
          observation, global_channel_id, sensor_id, platform_id, etc.
        - ``lengths_3d``: Shape (S, B, T) per-window obs counts in configured
          sensor order. Missing sensors remain present with zero counts.
    Stage 2 - ``device_transform()`` converts ``unified_obs`` to ``UnifiedObservation``:
        - Moves tensors to GPU
        - Computes ``float_metadata`` via ``features.compute_unified_metadata()``
          (encodes lat/lon, time deltas, zenith angles, etc.)
        - Populates per-obs integer fields: pix, local_channel, platform,
          global_platform, obs_type, global_channel
        - Returns ``types.UnifiedObservation`` dataclass ready for model input.

    Sensor grouping: Observations sorted by sensor_id with lengths_3d enabling
    efficient (sensor, batch, time) slicing (see ``split_by_sensor`` in
    ``healda.observations.types``).
    """

    variable_config: VariableConfig = VARIABLE_CONFIGS["era5_74ch"]
    hpx_level: int = 10  # pixel level of the observations
    hpx_level_condition: int = 6
    features_v2: bool = False
    sensors: list[str] = dataclasses.field(default_factory=list)
    background_normalization: NormalizationStats | None = None
    # Explicit target stats. When None, fall back to the era5/ufs CSV stats keyed
    # by ``variable_config.name`` (used by the original era5 path). Background
    # training passes the packed store's own 89-ch stats here instead.
    target_normalization: NormalizationStats | None = None
    # Prebuilt static condition (1, C, 1, X) in pixel_order. When set it replaces
    # the fixed static_data orog/lfrac (e.g. target-store orog/land_surface).
    # orog + land fraction from the target store; None for obs-only users.
    static_condition: torch.Tensor | None = None
    # Pre-pack observations per pixel for backbone obs cross-attention. Requires
    # hpx_level to equal the model's level (pix is used directly as the packing grid).
    attention_prepack: bool = False
    build_attention_group_map: bool = True
    # Shared order for state, static condition, and observation pixel ids.
    pixel_order: str = "hpxpadxy"
    # NNJA numbers sensors in its own order, which agrees with the GSI one only for atms.
    # Must match ObsConfig.use_nnja_sat, or the transform labels rows with ids the loader
    # never emitted.
    use_nnja_sat: bool = False
    _device: "_DeviceCache" = dataclasses.field(
        default_factory=_DeviceCache, init=False, compare=False, repr=False
    )

    def __post_init__(self):
        if self.target_normalization is not None:
            target_stats = self.target_normalization
        else:
            target_stats = get_batch_info(self.variable_config).normalization
        # default np dtype is fp64, which upcasts the state to fp64 during normalization. keep fp32
        self.mean = np.array(target_stats.center, dtype=np.float32)[:, None]
        self.std = np.array(target_stats.scales, dtype=np.float32)[:, None]
        background_stats = self.background_normalization or target_stats
        self.background_mean = np.array(background_stats.center, dtype=np.float32)[
            :, None
        ]
        self.background_std = np.array(background_stats.scales, dtype=np.float32)[
            :, None
        ]
        # For state_transforms.to_model_space: a no-op unless variable_config.name has
        # an entry in state_transforms.TRANSFORMS (currently only "era5_104ch", for the
        # tcw -> tcw-tcwv residual).
        self._channels = encode_channels(self.variable_config)
        self._config_name = self.variable_config.name

    @functools.cached_property
    def _grid(self):
        return healpix.Grid(self.hpx_level, pixel_order=_PIXEL_ORDERS[self.pixel_order])

    @functools.cached_property
    def _sensor_name_to_id(self) -> dict[str, int]:
        if not self.use_nnja_sat:
            return SENSOR_NAME_TO_ID
        # Imported here, not at module scope: nnja_combined pulls in both loaders.
        from healda.observations.loaders import combined

        return combined.SENSOR_NAME_TO_ID

    def _global_platform_ids(self, sensor_name: str) -> tuple[int, ...]:
        if not self.use_nnja_sat:
            return tuple(
                PLATFORM_NAME_TO_ID[p] for p in SENSOR_CONFIGS[sensor_name].platforms
            )
        from healda.observations.loaders import combined

        return combined.platform_ids(sensor_name)

    @functools.cached_property
    def _platform_luts(self) -> dict[int, torch.Tensor]:
        luts: dict[int, torch.Tensor] = {}
        for sensor_name in self.sensors:
            sensor_id = self._sensor_name_to_id[sensor_name]
            platform_ids = self._global_platform_ids(sensor_name)
            lut = torch.zeros(NPLATFORMS, dtype=torch.long)
            for local_platform_id, global_platform_id in enumerate(platform_ids):
                lut[global_platform_id] = local_platform_id
            luts[sensor_id] = lut
        return luts

    def _platform_lut_matrix(self, device) -> torch.Tensor:
        """Return the cached device platform lookup table."""
        return self._device.get(
            ("platform_lut", str(device)),
            lambda: self._build_platform_lut_matrix(device),
        )

    def _build_platform_lut_matrix(self, device) -> torch.Tensor:
        luts = [self._platform_luts[int(s)] for s in self._ordered_sensor_ids.tolist()]
        if not luts:
            return torch.zeros((0, 0), dtype=torch.long, device=device)
        return torch.stack(luts).to(device)

    @functools.cached_property
    def _ordered_sensor_ids(self) -> torch.Tensor:
        if not self.sensors:
            return torch.zeros((0,), dtype=torch.int32)
        return torch.tensor(
            [self._sensor_name_to_id[sensor_name] for sensor_name in self.sensors],
            dtype=torch.int32,
        )

    @staticmethod
    def _sort_by_record_batch(
        table: pa.Table, column_name: str, key_order: dict | None = None
    ) -> pa.Table:
        """Group rows into contiguous blocks by ``column_name`` (assumes each
        record batch is homogeneous in that column).

        ``key_order`` maps a value to its sort rank; pass it to order blocks by
        the model's sensor order rather than by raw ascending sensor_id (the two
        differ, see ``_process_obs``). Without it, blocks sort by raw value.
        """
        record_batches_order = []
        for batch in table.to_batches():
            if batch.num_rows == 0:
                continue
            group_value = batch[column_name][0].as_py()
            rank = key_order[group_value] if key_order is not None else group_value
            record_batches_order.append((rank, batch))

        # in empty case, from_batches will raise an error
        if not record_batches_order:
            return table

        record_batches_order.sort(key=lambda x: x[0])
        return pa.Table.from_batches([batch for _, batch in record_batches_order])

    @staticmethod
    def _append_batch_time_info_chunked(
        table: pa.Table, b: int, t: int, timestamp: int
    ) -> pa.Table:
        """
        Add batch/time indices and target time while maintaining original chunking.
        """
        b_idx_type = pa.int16()
        t_idx_type = pa.int16()
        time_type = pa.int64()

        ref_col = table.column(0)

        b_idx_chunks = []
        t_idx_chunks = []
        time_chunks = []

        for chunk in ref_col.chunks:
            L = len(chunk)
            if L == 0:
                b_idx_chunks.append(pa.array([], type=b_idx_type))
                t_idx_chunks.append(pa.array([], type=t_idx_type))
                time_chunks.append(pa.array([], type=time_type))
                continue

            # directly creating pa array of int16 not supported so use np first
            b_idx_arr = np.full(L, b, dtype=np.int16)
            t_idx_arr = np.full(L, t, dtype=np.int16)
            times_arr = np.full(L, timestamp, dtype=np.int64)

            b_idx_chunks.append(pa.array(b_idx_arr, type=b_idx_type))
            t_idx_chunks.append(pa.array(t_idx_arr, type=t_idx_type))
            time_chunks.append(pa.array(times_arr, type=time_type))

        b_chunked = pa.chunked_array(b_idx_chunks, type=b_idx_type)
        t_chunked = pa.chunked_array(t_idx_chunks, type=t_idx_type)
        time_chunked = pa.chunked_array(time_chunks, type=time_type)

        out = table.append_column("batch_idx", b_chunked)
        out = out.append_column("time_idx", t_chunked)
        out = out.append_column("target_time", time_chunked)
        return out

    @staticmethod
    def _build_observation_lengths_3d(
        obs_table: pa.Table, frame_times, ordered_sensor_ids: torch.Tensor
    ):
        B, T = len(frame_times), len(frame_times[0])

        counts_map = {}  # sensor_id -> (b, t) array of num obs in that sensor, batch, time

        for batch in obs_table.to_batches():
            if batch.num_rows == 0:
                continue

            s_id = int(batch["sensor_id"][0].as_py())
            b_id = int(batch["batch_idx"][0].as_py())
            t_id = int(batch["time_idx"][0].as_py())
            n = batch.num_rows

            if s_id not in counts_map:
                counts_map[s_id] = torch.zeros((B, T), dtype=torch.int32)

            counts_map[s_id][b_id, t_id] += n

        S = int(ordered_sensor_ids.numel())
        if S == 0:
            lengths_3d = torch.zeros((0, B, T), dtype=torch.int32)
            return lengths_3d

        lengths_3d = torch.zeros((S, B, T), dtype=torch.int32)
        for s_local, s_id in enumerate(ordered_sensor_ids.tolist()):
            if s_id in counts_map:
                lengths_3d[s_local] = counts_map[s_id]  # already (B, T) counts

        return lengths_3d

    @healda.utils.profiling.nvtx
    def _process_obs(self, target_times: list[list[cftime.datetime]], frames):
        if not self.sensors:
            raise ValueError("TransformV2 requires configured sensors for obs_v2.")

        # Add batch and time indices to each table before concatenation
        all_obs_with_indices = []
        for b_idx, sample_frames in enumerate(frames):
            for t_idx, frame_dict in enumerate(sample_frames):
                table = frame_dict["obs_v2"]
                table_with_indices = self._append_batch_time_info_chunked(
                    table,
                    b_idx,
                    t_idx,
                    _compute_timestamp(target_times[b_idx][t_idx]),
                )
                all_obs_with_indices.append(table_with_indices)

        obs = pa.concat_tables(all_obs_with_indices)

        # Sort obs into the model's sensor order (_ordered_sensor_ids), which is
        # NOT ascending sensor_id: conv-plevel has the largest id but sits mid-list.
        # lengths_3d uses the model order, and packing recovers each obs's (b, t)
        # by position in lengths, so the rows must match or obs land in the wrong
        # (batch, time) bucket once B*T > 1.
        sensor_rank = {
            int(sid): rank for rank, sid in enumerate(self._ordered_sensor_ids.tolist())
        }
        obs = self._sort_by_record_batch(obs, "sensor_id", key_order=sensor_rank)

        lengths_3d = self._build_observation_lengths_3d(
            obs, target_times, self._ordered_sensor_ids
        )

        return (self._obs_tensors(obs), lengths_3d)

    def _obs_tensors(self, obs: pa.Table) -> dict[str, torch.Tensor]:
        """Materialize the obs columns the model consumes, one tensor each.

        A tensor needs one buffer but columns arrive chunked, so each is gathered with
        `to_numpy`. Copying chunk by chunk into a preallocated array is marginally
        faster on a few chunks and much slower on highly fragmented input, where every
        per-chunk GIL release hands off to a waiting copy thread. The filter upstream
        avoids that by materializing its mask instead of running Acero, and `to_numpy`
        is flat in chunk count if it ever regresses.
        """

        def build(item: tuple[str, str]) -> tuple[str, torch.Tensor]:
            column, key = item
            values = obs[column]
            # to_numpy promotes an integer column with nulls to float64; fill first.
            if column in _OBS_FILL_ZERO:
                values = pc.fill_null(values, 0)
            out = values.to_numpy(zero_copy_only=False)
            # Kind "M" is datetime64: torch has no datetime dtype, so store int64
            # nanoseconds since the epoch, which is what features_v2 reads back. A view,
            # not astype, since datetime64[ns] already is int64 ns.
            if out.dtype.kind == "M":
                out = out.view(np.int64)
            # from_numpy views the buffer; no copy.
            return key, torch.from_numpy(out)

        pool = threads.thread_pool()
        tensors = dict(pool.map(build, _OBS_TENSOR_COLUMNS.items()))
        tensors["target_time_sec"] = torch.from_numpy(obs["target_time"].to_numpy())
        return tensors

    def _get_packed_field(self, frames, key: str) -> torch.Tensor:
        """
        Normalize + reorder in `device_transform` instead: at high-res (HPX256), takes >200ms w/ 1 torch CPU thread
        vs few ms on the GPU
        """
        # b*t frames, each (c, x)
        all_state = [f[key] for sample in frames for f in sample]
        if _frames_are_consecutive_slices(all_state):
            stacked = torch.as_tensor(all_state[0].base)  # (b*t, c, x), no copy
        else:
            stacked = torch.stack([torch.as_tensor(state) for state in all_state])
        shape = (len(frames), -1) + tuple(all_state[0].shape)
        return stacked.reshape(shape).float()

    def _const_to_device(self, name: str, values, device) -> torch.Tensor:
        """Return a cached device copy of a run-constant array."""
        return self._device.get(
            (name, str(device)), lambda: torch.as_tensor(values).to(device)
        )

    def _device_field(
        self, packed: torch.Tensor, name: str, mean, std, device
    ) -> torch.Tensor:
        """Normalize and reorder one packed field on `device`, (b, t, c, x) to (b, c, t, x)."""
        mean = self._const_to_device(f"{name}_mean", mean, device)  # (c, 1), over x
        std = self._const_to_device(f"{name}_std", std, device)
        moved = packed.to(device, non_blocking=True)
        # In place only when the move copied; `to` returns the input if it is already on `device`,
        # and normalizing that would edit the batch under its caller.
        centred = moved.sub_(mean) if moved is not packed else moved.sub(mean)
        return reorder_from_nest(
            centred.div_(std).permute(0, 2, 1, 3),
            self.pixel_order,
        )

    @healda.utils.profiling.nvtx
    def transform(self, times, frames):
        """
        frames: [[{state: (c, x), obs_v2: Obs}]]
        times: [[cftime]]
        """
        out = {}

        def _apply_time_func(func):
            return torch.from_numpy(np.vectorize(func)(times))

        if "obs_v2" in frames[0][0].keys():
            with healda.utils.profiling.cpu_timing_range("transform.obs"):
                out["unified_obs"] = self._process_obs(times, frames)
        with healda.utils.profiling.cpu_timing_range("transform.state"):
            out["target"] = state_transforms.to_model_space(
                self._get_packed_field(frames, "state"),
                self._channels,
                self._config_name,
                channel_axis=2,
            )
        with healda.utils.profiling.cpu_timing_range("transform.time"):
            out["second_of_day"] = _apply_time_func(_compute_second_of_day).float()
            out["day_of_year"] = _apply_time_func(_compute_day_of_year).float()
            out["timestamp"] = _apply_time_func(_compute_timestamp)
        # target is (b, t, c, x) here; device_transform permutes it to (b, c, t, x).
        b, t, _, _ = out["target"].shape
        if self.static_condition is None:
            raise ValueError(
                "TransformV2 needs static_condition to build a batch; packed stores "
                "pass their own, others can use fixed_static_condition()"
            )
        condition = self.static_condition.float()
        if condition.shape[0] not in (1, b):
            raise ValueError(
                f"condition batch dim {condition.shape[0]} must be 1 or target batch {b}"
            )
        if condition.shape[2] not in (1, t):
            raise ValueError(
                f"condition time dim {condition.shape[2]} must be 1 or target time {t}"
            )
        if "background" in frames[0][0]:
            out["background"] = self._get_packed_field(frames, "background")

        # Shipped unexpanded at (1, c, 1, x) and broadcast in device_transform, which
        # caches it. Expanding here pinned and copied b*t redundant copies per step.
        out["condition"] = condition
        out["labels"] = torch.empty([len(frames), 0])
        return out

    @healda.utils.profiling.nvtx
    def device_transform(self, batch, device):
        """Transforms to the output of .transform that can occur on gpu

        Typically used with the prefetch_map in the main training process.
        """
        batch = batch.copy()
        out = {}
        # Condition expansion uses target's (b, t). Observation-only calls omit both.
        # target is (b, t, c, x); _device_field returns (b, c, t, x).
        target = batch.get("target")
        b, t = (
            (target.shape[0], target.shape[1]) if target is not None else (None, None)
        )

        for key in batch:
            if key == "unified_obs":
                obs_tensors, lengths = batch["unified_obs"]
                out[key] = self._device_transform_unified_obs(
                    obs_tensors, lengths, device
                )
            elif key == "target":
                out[key] = self._device_field(
                    batch[key], "target", self.mean, self.std, device
                )
            elif key == "background":
                out[key] = self._device_field(
                    batch[key],
                    "background",
                    self.background_mean,
                    self.background_std,
                    device,
                )
            elif key == "condition":
                out[key] = self._device_condition(batch[key], b, t, device)
            else:
                out[key] = batch[key].to(device, non_blocking=True)
        return out

    def _device_condition(self, condition, b: int, t: int, device) -> torch.Tensor:
        """Return the run-static condition expanded on the device."""
        key = ("condition", b, t, str(device), condition.shape, condition.dtype)
        return self._device.get(
            key,
            lambda: condition.to(device, non_blocking=True)
            .expand(b, -1, t, -1)
            .contiguous(),
        )

    @healda.utils.profiling.nvtx
    def _device_transform_unified_obs(self, obs_tensors, lengths, device):
        # Move all tensors to device efficiently
        def _to_device(tensor, non_blocking=True):
            if isinstance(tensor, torch.Tensor):
                return tensor.to(device, non_blocking=non_blocking)
            else:
                return torch.from_numpy(tensor).to(device, non_blocking=non_blocking)

        obs_tensors = {key: _to_device(val) for key, val in obs_tensors.items()}

        obs_time_ns = obs_tensors["absolute_obs_time"]
        lat_tensor = obs_tensors["latitude"]
        lon_tensor = obs_tensors["longitude"]
        height_tensor = obs_tensors["height"]
        pressure_tensor = obs_tensors["pressure"]
        scan_angle_tensor = obs_tensors["scan_angle"]
        sat_zenith_tensor = obs_tensors["sat_zenith_angle"]
        sol_zenith_tensor = obs_tensors["sol_zenith_angle"]
        platform_id_tensor = obs_tensors["platform_id"].int()
        obs_type_tensor = obs_tensors["observation_type"].int()
        pix = self._grid.ang2pix(lon_tensor, lat_tensor).int()
        local_channel_id_tensor = obs_tensors["local_channel_id"].int()
        global_channel_id_tensor = obs_tensors["global_channel_id"].int()
        observation_tensor = obs_tensors["observation"]

        # Compute metadata
        if self.features_v2:
            meta = features_v2.compute_unified_metadata(
                obs_tensors["target_time_sec"],
                time=obs_time_ns,
                lon=lon_tensor,
                lat=lat_tensor,
                height=height_tensor,
                pressure=pressure_tensor,
                scan_angle=scan_angle_tensor,
                sat_zenith_angle=sat_zenith_tensor,
                sol_zenith_angle=sol_zenith_tensor,
            )
        else:
            meta = features.compute_unified_metadata(
                obs_tensors["target_time_sec"],
                time=obs_time_ns,
                lon=lon_tensor,
                height=height_tensor,
                pressure=pressure_tensor,
                scan_angle=scan_angle_tensor,
                sat_zenith_angle=sat_zenith_tensor,
                sol_zenith_angle=sol_zenith_tensor,
            )

        lengths = _to_device(lengths)
        if self.attention_prepack:
            # The obs-attention path attends over all sensors together and never
            # reads a sensor-local platform id, so the mapping is dead work. -1 is a
            # fail-fast sentinel: nn.Embedding raises on negative indices.
            # TODO: carry None instead -- the sentinel array is still allocated,
            # reordered by obs_packing and moved to device for no reader.
            local_platform = torch.full_like(platform_id_tensor, -1)
        else:
            # Computed before any reordering, while lengths still describes
            # per-sensor contiguous spans.
            local_platform = _map_platform_to_local(
                platform=platform_id_tensor,
                lengths=lengths,
                lut_matrix=self._platform_lut_matrix(device),
            )

        out = types.UnifiedObservation(
            obs=observation_tensor,
            time=obs_time_ns,
            float_metadata=meta,
            pix=pix,
            local_channel=local_channel_id_tensor,
            local_platform=local_platform,
            obs_type=obs_type_tensor,
            global_channel=global_channel_id_tensor,
            global_platform=platform_id_tensor,
            hpx_level=self.hpx_level,
            lengths=lengths,
        )
        # Backbone pixel cross-attention requires observations packed into
        # per-pixel contiguous groups; the scatter/embed path does not.
        if self.attention_prepack:
            out = pack_observations_by_pixel(
                out,
                pixel_order=self.pixel_order,
                build_group_map=self.build_attention_group_map,
            )
        return out


def collate(obj):
    return obj
