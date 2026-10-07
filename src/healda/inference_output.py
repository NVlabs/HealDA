# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analysis zarr store and asynchronous writer for the local inference CLI."""

import collections
import concurrent.futures
import dataclasses
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
import zarr


def _last_frame(prediction):
    return prediction[:, :, -1, :].float().contiguous()


def _d2h_last_frame(prediction, buf, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # The caching allocator tracks only the allocating stream; recording the copy
        # stream keeps the prediction's memory from being recycled mid-copy.
        prediction.record_stream(stream)
        buf.copy_(_last_frame(prediction), non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
    return ready


def _write_analysis_frame(group, channels, index, frame, pool):
    # The model's spatial axis is flat whatever the grid, so a (time, lat, lon) output
    # needs the slab folded back into 2D. A no-op for HPX.
    for done in [
        pool.submit(
            group[name].__setitem__,
            index,
            frame[:, c, :].reshape(frame.shape[0], *group[name].shape[1:]),
        )
        for c, name in enumerate(channels)
    ]:
        done.result()


PRESSURE_FAMILIES = {
    "Z": ("Geopotential", "m2 s-2", "geopotential", "z"),
    "T": ("Temperature", "K", "air_temperature", "t"),
    "U": ("Eastward wind", "m s-1", "eastward_wind", "u"),
    "V": ("Northward wind", "m s-1", "northward_wind", "v"),
    "W": (
        "Vertical velocity (pressure)",
        "Pa s-1",
        "lagrangian_tendency_of_air_pressure",
        "w",
    ),
    "Q": ("Specific humidity", "kg kg-1", "specific_humidity", "q"),
}

# name: (long_name, units, CF standard_name or None, ERA5 short name)
SINGLE_LEVEL = {
    "uas": ("Eastward wind at 10 m", "m s-1", "eastward_wind", "10u"),
    "vas": ("Northward wind at 10 m", "m s-1", "northward_wind", "10v"),
    "100u": ("Eastward wind at 100 m", "m s-1", "eastward_wind", "100u"),
    "100v": ("Northward wind at 100 m", "m s-1", "northward_wind", "100v"),
    "tas": ("Temperature at 2 m", "K", "air_temperature", "2t"),
    "d2m": ("Dewpoint temperature at 2 m", "K", "dew_point_temperature", "2d"),
    "skt": ("Skin temperature", "K", None, "skt"),
    "pres_msl": (
        "Mean sea level pressure",
        "Pa",
        "air_pressure_at_mean_sea_level",
        "msl",
    ),
    "sp": ("Surface pressure", "Pa", "surface_air_pressure", "sp"),
    "tcw": ("Total column water", "kg m-2", None, "tcw"),
    "tcwv": (
        "Total column water vapour",
        "kg m-2",
        "atmosphere_mass_content_of_water_vapor",
        "tcwv",
    ),
    "tcc": ("Total cloud cover", "1", "cloud_area_fraction", "tcc"),
    "lcc": ("Low cloud cover", "1", None, "lcc"),
    "mcc": ("Medium cloud cover", "1", None, "mcc"),
    "hcc": ("High cloud cover", "1", None, "hcc"),
    "sst": ("Sea surface temperature", "K", "sea_surface_temperature", "sst"),
    "sic": ("Sea ice area fraction", "1", "sea_ice_area_fraction", "siconc"),
    "sd": ("Snow depth", "m of water equivalent", None, "sd"),
    "stl1": ("Soil temperature level 1 (0-7 cm)", "K", None, "stl1"),
    "stl2": ("Soil temperature level 2 (7-28 cm)", "K", None, "stl2"),
    "swvl1": ("Volumetric soil water layer 1 (0-7 cm)", "m3 m-3", None, "swvl1"),
    "swvl2": ("Volumetric soil water layer 2 (7-28 cm)", "m3 m-3", None, "swvl2"),
}


def channel_attrs(name: str) -> dict:
    """CF-style ``long_name``/``units``/``standard_name`` for a channel; {} if unknown."""
    family, level = name[:1], name[1:]
    if family in PRESSURE_FAMILIES and level.isdigit():
        long_name, units, standard_name, era5 = PRESSURE_FAMILIES[family]
        return {
            "long_name": f"{long_name} at {level} hPa",
            "units": units,
            "standard_name": standard_name,
            "pressure_level_hPa": int(level),
            "era5_short_name": era5,
        }
    if name in SINGLE_LEVEL:
        long_name, units, standard_name, era5 = SINGLE_LEVEL[name]
        attrs = {"long_name": long_name, "units": units, "era5_short_name": era5}
        if standard_name:
            attrs["standard_name"] = standard_name
        return attrs
    return {}


def round_mantissa(values: np.ndarray, bits: int) -> np.ndarray:
    """Round float32 `values` in place to `bits` mantissa bits, to nearest; NaN and inf kept."""
    if bits >= 23:
        return values
    drop = 23 - bits
    raw = values.view(np.uint32)
    rounded = (raw + np.uint32(1 << (drop - 1))) & np.uint32(
        ~((1 << drop) - 1) & 0xFFFFFFFF
    )
    finite = np.isfinite(values)
    raw[finite] = rounded[finite]
    return values


def _consecutive(sorted_slots):
    runs = [[sorted_slots[0]]]
    for slot in sorted_slots[1:]:
        if slot == runs[-1][-1] + 1:
            runs[-1].append(slot)
        else:
            runs.append([slot])
    return runs


class StageTimer:
    """Mean wall time per stage, CUDA-synchronized at each mark, printed every `every` steps.

    Synchronizing removes overlap between stages, so the totals attribute time rather than
    reproduce the unprofiled step time.
    """

    def __init__(self, enabled, rank, every=56):
        self.enabled = enabled
        self.rank = rank
        self.every = every
        self.totals = collections.defaultdict(float)
        self.steps = 0
        self.last = time.perf_counter()

    def mark(self, stage):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.totals[stage] += now - self.last
        self.last = now

    def step(self):
        if not self.enabled:
            return
        self.steps += 1
        if self.steps % self.every:
            return
        parts = " ".join(
            f"{k}={1000 * v / self.every:.0f}" for k, v in self.totals.items()
        )
        print(
            f"[profile rank {self.rank}] steps {self.steps} ms/step {parts}", flush=True
        )
        self.totals.clear()


class AsyncZarrWriter:
    """Pinned device-to-host copy, one drain thread, full shards written in the background.

    A filled shard is handed to the emit thread so the drain thread keeps accepting
    analyses; at most `IN_FLIGHT` shards wait on the emit side at once.
    """

    IN_FLIGHT = 2

    def __init__(self, group, channels, workers=4, profile=False, mantissa_bits=23):
        self.profile = profile
        self.mantissa_bits = mantissa_bits
        self.group = group
        self.channels = channels
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.pending = queue.Queue(maxsize=10)
        self.free = collections.defaultdict(queue.Queue)
        self.error = None
        # Shard depth, not `.chunks`: under sharding the chunk is the inner one, so
        # staging to it would re-encode the whole shard on every analysis.
        first = group[channels[0]]
        self.time_chunk = (first.shards or first.chunks)[0]
        self.num_times = group[channels[0]].shape[0]
        self.stage = None
        self.stage_metrics = None
        self.stage_chunk = None
        self.stage_filled = set()
        self.emitter = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.emits = collections.deque()
        self.free_stages = queue.Queue()
        self._copy_stream = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _acquire(self, shape, dtype, pin):
        try:
            return self.free[(tuple(shape), dtype)].get_nowait()
        except queue.Empty:
            return torch.empty(shape, dtype=dtype, pin_memory=pin)

    def _release(self, buffer):
        self.free[(tuple(buffer.shape), buffer.dtype)].put(buffer)

    def _extent(self, chunk):
        start = chunk * self.time_chunk
        return min(start + self.time_chunk, self.num_times) - start

    def _emit(self):
        if self.stage_chunk is None or not self.stage_filled:
            return
        job = (
            self.stage,
            self.stage_metrics,
            self.stage_chunk,
            sorted(self.stage_filled),
        )
        self.stage = None
        self.stage_metrics = None
        self.stage_chunk = None
        self.stage_filled = set()
        emit = self.emitter.submit(self._write_stage, *job)
        emit.add_done_callback(self._record_emit_error)
        self.emits.append(emit)
        while len(self.emits) > self.IN_FLIGHT:
            self.emits.popleft().result()

    def _write_stage(self, stage, metrics, chunk, filled):
        start = chunk * self.time_chunk
        began = time.perf_counter()
        for run in _consecutive(filled):
            round_mantissa(stage[run[0] : run[-1] + 1], self.mantissa_bits)
            _write_analysis_frame(
                self.group,
                self.channels,
                slice(start + run[0], start + run[-1] + 1),
                stage[run[0] : run[-1] + 1],
                self.pool,
            )
        if metrics is not None:
            for run in _consecutive(filled):
                for name, values in zip(METRICS, metrics):
                    self.group[name][start + run[0] : start + run[-1] + 1] = values[
                        run[0] : run[-1] + 1
                    ]
        self.free_stages.put(stage)
        if self.profile:
            print(
                f"[profile writer] emit {len(filled)} times "
                f"{time.perf_counter() - began:.2f} s, queue {self.pending.qsize()}",
                flush=True,
            )

    def _accept(self, output_index, row, metric_rows=None):
        chunk, slot = divmod(output_index, self.time_chunk)
        if self.stage_chunk is not None and chunk != self.stage_chunk:
            self._emit()
        if self.stage is None:
            try:
                self.stage = self.free_stages.get_nowait()
            except queue.Empty:
                self.stage = np.empty((self.time_chunk, *row.shape), row.dtype)
        if metric_rows is not None and self.stage_metrics is None:
            self.stage_metrics = tuple(
                np.full((self.time_chunk, *r.shape), np.nan, np.float32)
                for r in metric_rows
            )
        self.stage_chunk = chunk
        self.stage[slot] = row
        if metric_rows is not None:
            for staged, r in zip(self.stage_metrics, metric_rows):
                staged[slot] = r
        self.stage_filled.add(slot)
        if len(self.stage_filled) == self._extent(chunk):
            self._emit()

    def _serve(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    self._emit()
                else:
                    index, buffer, event, metrics = item
                    try:
                        if event is not None:
                            event.synchronize()
                        rows = buffer.numpy()
                        for position, output_index in enumerate(index):
                            self._accept(
                                int(output_index),
                                rows[position],
                                None
                                if metrics is None
                                else tuple(m[position] for m in metrics),
                            )
                    finally:
                        self._release(buffer)
            except Exception as exc:
                self.error = self.error or exc
            finally:
                self.pending.task_done()
            if item is None:
                return

    def _record_emit_error(self, emit):
        if emit.exception() is not None:
            self.error = self.error or emit.exception()

    def _reraise(self):
        if self.error is not None:
            raise RuntimeError("async zarr write failed") from self.error

    def submit(self, index, prediction, metrics=None):
        """`metrics`: per-time (rmse, mae) arrays shaped (len(index), field, frame),
        written with the analysis chunk they belong to when the store holds them."""
        self._reraise()
        if prediction.is_cuda:
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream()
            shape = (prediction.shape[0], len(self.channels), prediction.shape[-1])
            buffer = self._acquire(shape, torch.float32, pin=True)
            event = _d2h_last_frame(prediction, buffer, self._copy_stream)
        else:
            frame = _last_frame(prediction)
            buffer = self._acquire(frame.shape, frame.dtype, pin=False)
            buffer.copy_(frame)
            event = None
        self.pending.put((index, buffer, event, metrics))

    def close(self):
        self.pending.put(None)
        self.thread.join()
        try:
            for emit in self.emits:
                emit.result()
        except Exception as exc:
            self.error = self.error or exc
        self.emitter.shutdown()
        self.pool.shutdown()
        self._reraise()


def open_output_store(
    output_path, channels, times, step, *, extend, identity, frames=None, **setup
):
    """Create a store on a regular `step` grid spanning `times`, or with `extend` grow the
    existing one to cover them. A store holds one checkpoint and one observing system:
    extending with a different `identity` (checkpoint, obs_config) is refused.
    With `frames`, the store also holds per-time `rmse`/`mae` (time, field, frame),
    one object per `time_chunk` like the analysis.
    Returns the opened group; slots follow from its first time.
    """
    if not (extend and os.path.exists(output_path)):
        grid = pd.date_range(times[0], times[-1], freq=step)
        group = setup_zarr_output(
            output_path, channels, num_times=len(grid), subsampled_times=grid, **setup
        )
        group.attrs.update({**identity, "runs": []})
        _create_run_ids(group, len(grid))
        if frames is not None:
            _create_metric_arrays(
                group, len(grid), channels, frames, setup.get("time_chunk") or 1
            )
        return group

    group = zarr.open_group(output_path, mode="r+")
    for key, value in identity.items():
        if group.attrs.get(key) != value:
            raise ValueError(
                f"{output_path} holds a different {key}; one store per checkpoint and "
                "observing system, so write a new store"
            )
    missing = [c for c in channels if c not in group]
    if missing:
        raise ValueError(f"{output_path} lacks channels {missing}")
    unlabelled = [c for c in channels if "units" not in group[c].attrs]
    for channel in unlabelled:
        group[channel].attrs.update(channel_attrs(channel))
    stored_depth = (group[channels[0]].shards or group[channels[0]].chunks)[0]
    if stored_depth != (setup.get("time_chunk") or 1):
        # Ranks writing one shard concurrently would overwrite each other.
        raise ValueError(
            f"{output_path} holds {stored_depth} times per object; this run writes "
            f"{setup.get('time_chunk') or 1}, so write a new store"
        )
    stored = pd.to_datetime(group["time"][:2], unit="s")
    if len(stored) > 1 and stored[1] - stored[0] != step:
        raise ValueError(f"{output_path} is spaced {stored[1] - stored[0]}, not {step}")
    have = pd.date_range(stored[0], periods=group["time"].shape[0], freq=step)
    if times[0] < have[0]:
        raise ValueError(
            f"{output_path} starts {have[0]}; a window from {times[0]} needs a new store"
        )
    grid = pd.date_range(have[0], max(have[-1], times[-1]), freq=step)
    grid_seconds = grid.to_numpy().astype("datetime64[s]").astype(np.int64)
    if frames is not None and not all(m in group for m in METRICS):
        _create_metric_arrays(
            group, len(have), channels, frames, setup.get("time_chunk") or 1
        )
    grown = len(grid) > len(have) or not np.array_equal(
        group["time"][:], grid_seconds[: len(have)]
    )
    if grown:
        for name in [*channels, "time", "run_id", *(m for m in METRICS if m in group)]:
            group[name].resize((len(grid), *group[name].shape[1:]))
        group["time"][:] = grid_seconds
    if grown or unlabelled:
        # The group's consolidated metadata still holds the old shapes and attributes,
        # and any reopen (every writer rank opens the group by path) reads those.
        zarr.consolidate_metadata(group.store)
        group = zarr.open_group(output_path, mode="r+")
    return group


def _create_run_ids(group, n):
    run_id = group.create_array(
        "run_id",
        shape=(n,),
        chunks=(max(n, 1),),
        dtype=np.int16,
        fill_value=-1,
        dimension_names=["time"],
    )
    run_id.attrs["description"] = (
        "index into the root `runs` attribute; -1 = never written"
    )


METRICS = ("rmse", "mae")


def _create_metric_arrays(group, n, channels, frames, depth):
    for name in METRICS:
        array = group.create_array(
            name,
            shape=(n, len(channels), frames),
            chunks=(depth, len(channels), frames),
            dtype=np.float32,
            fill_value=float("NaN"),
            dimension_names=["time", "field", "frame"],
        )
        array.attrs["fields"] = list(channels)
        array.attrs["description"] = (
            f"area-weighted {name} vs the target per field and frame; "
            "frame -1 is the analysis; NaN = not written"
        )


def store_slots(group, times, step) -> np.ndarray:
    offset = pd.DatetimeIndex(times) - pd.to_datetime(group["time"][0], unit="s")
    if (offset % step != pd.Timedelta(0)).any():
        raise ValueError(f"times off the store's {step} grid")
    return (offset // step).to_numpy().astype(int)


def record_run(group, slots, entry):
    runs = list(group.attrs.get("runs", []))
    run = len(runs)
    group["run_id"][np.asarray(slots)] = run
    group.attrs["runs"] = [*runs, {"run_id": run, **entry}]
    return run


def provenance_attrs(obs_config, extra_filters) -> dict:
    """What produced this output. Written onto the zarr root.

    Two scoring runs can differ only in their observing system, so the resolved
    ObsConfig and the run's extra filters travel with the output.
    """
    try:
        git_revision = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode()
        )
        # An uncommitted fix records a revision that does not contain it, so say so.
        if subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], stderr=subprocess.DEVNULL
        ).strip():
            git_revision += "-dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_revision = ""
    try:
        repo = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        repo = branch = ""
    attrs = {
        "history": " ".join(sys.argv),
        "host": socket.gethostname(),
        "date_created": datetime.now(UTC).isoformat(),
        "git_revision": git_revision,
        "git_worktree": repo,
        "git_branch": branch,
    }
    attrs["obs_config"] = json.dumps(dataclasses.asdict(obs_config), default=str)
    attrs["extra_obs_filters"] = json.dumps(dataclasses.asdict(extra_filters))
    return attrs


def setup_zarr_output(
    output_path,
    channels,
    num_times,
    subsampled_times,
    grid,
    time_chunk=None,
    attrs=None,
    zstd_level=None,
):
    """Setup zarr output structure for inference results.

    Args:
        output_path: Path to output zarr file
        channels: List of channel names
        num_times: Number of time steps
        subsampled_times: pandas DatetimeIndex for the time coordinate
        time_chunk: Depth of one output object. With `zstd_level` set this is the
            SHARD depth and the inner chunk stays 1 analysis deep; without it, it
            is the chunk depth directly.
        grid: Spatial layout as (names, sizes), e.g. (("lat", "lon"), (721, 1440)) or
            (("cells",), (49152,)).
        zstd_level: Compress with zstd at this level. None writes raw.

    Returns:
        Opened zarr group (mode='w')
    """
    group = zarr.open_group(output_path, mode="w")
    group.attrs.update(attrs or {})

    dim_names, dim_sizes = grid

    # Sharding lets one file hold `time_chunk` analyses while a reader still pulls a
    # single one, since the inner chunks are separately compressed and addressable.
    compressors = [zarr.codecs.ZstdCodec(level=zstd_level)] if zstd_level else []
    depth = time_chunk or 1
    shards = (depth, *dim_sizes) if zstd_level and depth > 1 else None
    chunks = (1 if shards else depth, *dim_sizes)

    # Create data arrays for each channel
    for field in channels:
        group.create_array(
            field,
            shape=(num_times, *dim_sizes),
            chunks=chunks,
            shards=shards,
            fill_value=float("NaN"),
            dimension_names=("time", *dim_names),
            dtype="f",
            compressors=compressors,
            attributes=channel_attrs(field),
        )

    # Time coordinate
    times_array = subsampled_times.to_numpy()
    time_v = group.create_array(
        "time",
        dtype=np.int64,
        shape=times_array.shape,
        chunks=times_array.shape,
        dimension_names=["time"],
    )
    time_v[:] = times_array.astype("datetime64[s]").astype(np.int64)
    time_v.attrs["units"] = "seconds since 1970-01-01 00:00:00"
    time_v.attrs["calendar"] = "standard"

    # HPX "cells" is an index rather than a coordinate, so only lat/lon gets values.
    if dim_names == ("lat", "lon"):
        nlat, nlon = dim_sizes
        for name, values in (
            ("lat", np.linspace(90.0, -90.0, nlat, dtype=np.float32)),
            ("lon", np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float32)),
        ):
            coord = group.create_array(
                name,
                dtype=np.float32,
                shape=values.shape,
                chunks=values.shape,
                dimension_names=[name],
            )
            coord[:] = values
            coord.attrs["units"] = "degrees_north" if name == "lat" else "degrees_east"

    zarr.consolidate_metadata(group.store)
    return group
