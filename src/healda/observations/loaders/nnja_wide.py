# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wide NNJA satellite archive loader into the unified long observation schema.

The archive is wide (one row per footprint, one column per channel); the model-facing
schema in ``schema.py`` is long (one row per footprint-channel). Only the
columns the unified schema consumes are read.

Pipeline: read, select, transform (gather and normalize), expand. Selecting before
expanding avoids `channels` times more metadata duplication; it does not avoid reading
channel pages. Channel metadata is a positional lookup, not a hash join: one column is
one channel.
"""

from __future__ import annotations

import asyncio
import functools
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from healda.observations.loaders.archive_row_groups import window_row_groups

from healda.observations.loaders.threads import configure_arrow_pools, thread_pool
from healda.observations.sensors import PLATFORM_NAME_TO_ID
from healda.observations.sensors_nnja import (
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
    SENSOR_OFFSET,
    get_global_channel_id,
)
from healda.observations.sensors_nnja import channel_table as nnja_channel_table
from healda.observations.preprocessing import ir_spectral, scan_geometry
from healda.utils.profiling import cpu_timing_range
from healda.config.environment import nnja_archive
from healda.observations.loaders.nnja_base import (
    GLOBAL_CHANNEL_ID,
    LOCAL_CHANNEL_ID,
    SENSOR_ID,
    NNJAArchiveLoader,
    SampleDropout,
    row_generator,
)


# The archive stores one uint32 hpx2048_nest (NESTED order 11) per footprint; coarser
# parents are hpx2048_nest >> (2 * (11 - target_order)).
ARCHIVE_HPX_ORDER = 11

# BUFR SAID (0-01-007) to the unified platform vocabulary.
SAID_TO_PLATFORM_NAME = {
    3: "metop-b",
    4: "metop-a",
    5: "metop-c",
    206: "n15",
    207: "n16",
    208: "n17",
    209: "n18",
    223: "n19",
    224: "npp",
    225: "n20",
    226: "n21",
    784: "aqua",
}

# Value columns to load. Note: for MW sounders, we only have choice for ATMS (which comes with both TMANT and TMBR). Rest are called "TMBR"
# but are actually TMANT. Only N15/16 AMSUA is actually TMBR. There is a deterministic per platform/fov/sensor to apply the TMANT->TMBR antenna correction
# Also note GSI UFS replay ran ta2tb=false, so ingested as is.
SENSOR_VALUE_PREFIX = {
    "amsua": "brightness_temperature",
    "amsub": "brightness_temperature",
    "mhs": "brightness_temperature",
    "atms": "antenna_temperature",
    "airs": "brightness_temperature",
    "iasi": "spectral_radiance",
    "cris": "spectral_radiance_code",
}

# Published as radiance, not kelvin; `transform` inverts Planck before the range check.
RADIANCE_PREFIXES = ("spectral_radiance", "spectral_radiance_code")

IR_SOUNDERS = tuple(
    name for name, config in SENSOR_CONFIGS.items() if config.sensor_type == "infrared"
)

# CrIS names the detector in its 3x3 array field_of_view and the cross-track look
# field_of_regard; the look is the scan position. Elsewhere field_of_view is the position.
SCAN_POSITION_COLUMN = {"cris": "field_of_regard"}

# Archive footprint columns → unified schema. Scan_Angle is derived from the scan identity, and
# Sat_Zenith_Angle is the archive's unsigned magnitude signed by scan side.
FOOTPRINT_COLUMNS = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "time_utc": "Absolute_Obs_Time",
    "da_window": "DA_window",
    "platform_id": "Platform_ID",
    "satellite_zenith_angle": "Sat_Zenith_Angle",
    "solar_zenith_angle": "Sol_Zenith_Angle",
    "field_of_view": "Scan_Angle",
}

# Footprint columns the metadata featurizer encodes on the satellite branch. Need
# to sanitize based on these to prevent NaN from reaching the model.
# Rare and granule-shaped NaNs
GEOMETRY_COLUMNS = (
    "Latitude",
    "Longitude",
    "Scan_Angle",
    "Sat_Zenith_Angle",
    "Sol_Zenith_Angle",
)

# Read for selection only; never emitted.
SELECTION_COLUMNS = ("hpx2048_nest",)

# Unified columns the sat path leaves null.
NULL_COLUMNS = (
    "Pressure",
    "Height",
    "Observation_Type",
    "QC_Flag",
    "Analysis_Use_Flag",
)


# Dense SAID-indexed lookups for selection keys; the vocabulary is fixed, so both come
# from an array instead of a dict get or a sort over every footprint's SAID.
SAID_TO_PLATFORM_ID = np.zeros(max(SAID_TO_PLATFORM_NAME) + 1, dtype=np.uint16)
for _said, _name in SAID_TO_PLATFORM_NAME.items():
    SAID_TO_PLATFORM_ID[_said] = PLATFORM_NAME_TO_ID[_name]

SAID_TO_PLATFORM_CODE = np.zeros(max(SAID_TO_PLATFORM_NAME) + 1, dtype=np.int64)
for _code, _said in enumerate(sorted(SAID_TO_PLATFORM_NAME)):
    SAID_TO_PLATFORM_CODE[_said] = _code


# --- arrow helpers ---------------------------------------------------------


def _all_null_columns(
    types: Sequence[pa.DataType], rows: int, values: pa.Buffer
) -> list[pa.Array]:
    """All-null fixed-width columns aliasing one values buffer.

    `pa.nulls` zeroes a values buffer per column, ~25 ms per 25M-row float32 column;
    null values are never read, so only the validity bitmap needs zeroing.
    """
    validity = pa.py_buffer(bytearray((rows + 7) // 8))
    widest = max(type.bit_width // 8 for type in types)
    if values.size < rows * widest:
        raise ValueError("shared values buffer is too small")
    return [
        pa.Array.from_buffers(type, rows, [validity, values], null_count=rows)
        for type in types
    ]


def _numpy(column: pa.ChunkedArray) -> np.ndarray:
    """Zero-copy view when the column is one non-null chunk; `to_numpy` always allocates."""
    if column.num_chunks == 1 and column.null_count == 0:
        try:
            return column.chunk(0).to_numpy(zero_copy_only=True)
        except pa.ArrowInvalid:
            pass
    return column.to_numpy(zero_copy_only=False)


# --- thinning: score every footprint, keep the winner of each cell ---------
#
# A cell is (platform, coarsened pixel). Each policy in WINNER_POLICIES supplies a
# uint32 score, lower wins, and the reduction below picks the minimum per cell.
#
# 'random' scores a row by hashing its identity, not by drawing from a stream: a stream's value
# depends on how many draws preceded it, so an upstream screen would move the winner of a cell it
# never touched, and ranks thinning one window in separate processes would have to consume in
# lockstep to agree. A counter-based RNG is this hash with more rounds.


# splitmix64's golden-ratio Weyl increment and first multiplier. Shared by the numpy and
# compiled forms of the hash so the two cannot drift.
_HASH_ADD = np.uint64(0x9E3779B97F4A7C15)
_HASH_MUL = np.uint64(0xBF58476D1CE4E5B9)
_SCORE_SHIFT = np.uint64(32)
_LOW_32 = np.uint64(0xFFFFFFFF)


def winner_hash(index: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-footprint score, in the low 32 bits, for random thinning.

    One round of splitmix64: Weyl step, multiply, xor-fold, against the two rounds and three
    folds of the full finalizer. Only 32 bits of avalanche are needed since the score is the
    low half. Rows sharing a cell are near-consecutive, so the multiply alone leaves a ramp
    biasing the winner as far as 0.234; the fold takes mean normalized rank to 0.496-0.498.
    """
    value = index.astype(np.uint64, copy=True) + np.uint64(seed)
    value = (value + _HASH_ADD) * _HASH_MUL
    return (value ^ (value >> _SCORE_SHIFT)) & _LOW_32


# Both selection paths pack a cell as `score << 32 | row`, so an integer compare takes the
# lowest score and breaks ties on the earliest row. All ones loses to every real value.
_PACKED_EMPTY = np.uint64(0xFFFFFFFFFFFFFFFF)

# Scoring modes for the selection kernel.
_MODE_FIRST = 0
_MODE_HASH = 1
_MODE_SCORE = 2

# random: row hash against a fixed seed. resample: reseeded per sel_time, so a revisited
# target draws different footprints. synoptic: nearest analysis hour. first: biased, scan order.
WINNER_POLICIES = ("random", "resample", "synoptic", "first")
SYNOPTIC_PERIOD = np.int64(6 * 3600)


def synoptic_distance(time_utc: np.ndarray) -> np.ndarray:
    """Seconds from each observation time to the nearest 00/06/12/18 analysis hour."""
    seconds = time_utc.astype("datetime64[s]").astype(np.int64)
    offset = np.mod(seconds, SYNOPTIC_PERIOD)
    return np.minimum(offset, SYNOPTIC_PERIOD - offset).astype(np.uint32)


def _winner_rows(best_per_cell: np.ndarray, row_count: int) -> np.ndarray:
    """Ascending row indices out of a packed argmin table.

    `best_per_cell` is in cell order, and the caller gathers `channels` columns by this
    index, so returning it unsorted would make each of those a random walk.

    Winners are distinct and dense in [0, row_count), so a bitmap plus `flatnonzero` beats
    sorting them: linear in `row_count` against k log k for k winners. 1.2-1.9x on the select
    stage at nside 256, break-even near 30x thinning, 2% the wrong way past 90x, coarser than
    the archive is thinned for.

    Assumes no row wins twice, which holds because a row keys to exactly one cell. A
    repeat collapses in the bitmap and comes back once, with nothing raised.
    """
    # Drop the cells nothing keyed to, then mask off the score to leave the row index.
    filled = best_per_cell[best_per_cell != _PACKED_EMPTY]
    is_winner = np.zeros(row_count, dtype=bool)
    is_winner[(filled & _LOW_32).astype(np.int64)] = True
    return np.flatnonzero(is_winner)


def dense_argmin_per_key(
    keys: np.ndarray, score_bits: np.ndarray, key_count: int
) -> np.ndarray:
    """Score-minimizing row per key, ascending; ties break to the earliest row.

    `score_bits` must be uint32 and monotone in the score. Packing score above row index
    lets one `np.minimum.at` resolve winner and tie-break, halving the two-reduction form.
    """
    packed = (score_bits.astype(np.uint64) << _SCORE_SHIFT) | np.arange(
        keys.size, dtype=np.uint64
    )
    best_per_key = np.full(key_count, _PACKED_EMPTY, dtype=np.uint64)
    np.minimum.at(best_per_key, keys, packed)
    return _winner_rows(best_per_key, keys.size)


def _compile_fused_select():
    """Screen, key, score, and argmin in one pass over footprints.

    Staged NumPy builds a mask, a compacted index, and three gathered key columns first.
    Fusing runs 1.5-2.8x faster. Runs `nogil`, so the per-sensor threads run it concurrently.
    Returns None without numba, which leaves the staged path.
    """
    try:
        import numba
    except ImportError:
        # The staged path is correct, just slower, so warn rather than fail.
        warnings.warn(
            "numba is not installed; satellite selection falls back to the staged "
            "NumPy path, which is 1.5-2.8x slower.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    @numba.njit(cache=True, nogil=True)
    def kernel(
        pixel,
        said,
        field_of_view,
        platform_codes,
        keep_low,
        keep_high,
        screen_fov,
        bit_shift,
        pixels_per_platform,
        mode,
        seed,
        scores,
        best_per_cell,
    ):
        for row in range(pixel.shape[0]):
            if screen_fov:
                position = field_of_view[row]
                if position < keep_low or position > keep_high:
                    continue
            # NESTED is a quadtree, two bits per level, so the shift is the parent cell.
            cell = np.int64(platform_codes[said[row]]) * pixels_per_platform + np.int64(
                pixel[row] >> bit_shift
            )
            if mode == _MODE_HASH:
                # The arithmetic of `winner_hash`, on one row at a time.
                hashed = (np.uint64(row) + seed + _HASH_ADD) * _HASH_MUL
                folded = (hashed ^ (hashed >> _SCORE_SHIFT)) & _LOW_32
                packed = (folded << _SCORE_SHIFT) | np.uint64(row)
            elif mode == _MODE_SCORE:
                packed = (np.uint64(scores[row]) << _SCORE_SHIFT) | np.uint64(row)
            else:
                # Score zero for every row, so the minimum is the earliest row.
                packed = np.uint64(row)
            if packed < best_per_cell[cell]:
                best_per_cell[cell] = packed
        return best_per_cell

    return kernel


_FUSED_SELECT = _compile_fused_select()


# --- read plan and per-stage results ---------------------------------------


@dataclass(frozen=True)
class ReadPlan:
    """Per-sensor column layout and positional channel metadata."""

    value_columns: tuple[str, ...]
    global_channel_id: np.ndarray
    local_channel_id: np.ndarray
    mean: np.ndarray
    stddev: np.ndarray
    min_valid: np.ndarray
    max_valid: np.ndarray
    # Set only for the radiance sensors: channel centre wavenumbers in cm^-1, for the kelvin
    # conversion.
    wavenumber: np.ndarray | None = None
    # CrIS's `reference` and `scale`, from its column metadata, which reconstruct its integer
    # radiance code. Empty for IASI, which publishes the float, so it splats into
    # `radiance_mw` either way.
    value_encoding: Mapping[str, int] = field(default_factory=dict)
    # The sensor's scan-identity column, when it is not the field_of_view every other
    # footprint column is keyed on.
    scan_columns: tuple[str, ...] = ()

    @property
    def channel_count(self) -> int:
        return len(self.value_columns)

    @property
    def columns(self) -> list[str]:
        return [
            *FOOTPRINT_COLUMNS,
            *self.scan_columns,
            *SELECTION_COLUMNS,
            *self.value_columns,
        ]


@dataclass
class Normalized:
    """Normalized values with a per-cell validity mask; None means every cell valid.

    A cell is valid when its footprint has finite geometry and its value lies inside the
    channel's min_valid..max_valid.
    """

    values: np.ndarray
    valid: np.ndarray | None
    footprint_columns: dict[str, np.ndarray]

    @property
    def footprint_count(self) -> int:
        return self.values.shape[1]


def _resolve_ir_channels(
    ir_channels: str | Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, ...]] | None:
    """A preset name or an explicit mapping as one form: sounder to channel numbers, or None.

    None, and a sounder the mapping omits, mean the full published axis.
    """
    if ir_channels is None:
        return None
    if isinstance(ir_channels, str):
        return ir_spectral.ir_channel_preset(ir_channels)
    resolved = {
        sensor: tuple(int(channel) for channel in wanted)
        for sensor, wanted in ir_channels.items()
    }
    # `wanted_channels` keys on sensor alone, so a microwave name would narrow that axis too.
    # The presets draw on this vocabulary, so a mapping can name whatever a preset can.
    other = sorted(
        name for name in resolved if name not in ir_spectral.ir_channel_sets()
    )
    if other:
        raise ValueError(f"ir_channels names non-sounder {other}")
    # An empty selection would otherwise build a plan of no channels and emit no rows.
    empty = sorted(name for name, wanted in resolved.items() if not wanted)
    if empty:
        raise ValueError(f"ir_channels names no channel for {empty}")
    return resolved


class NNJAWideLoader(NNJAArchiveLoader):
    """Read the wide NNJA satellite archive as unified long observation tables.

    The archive stores one row per footprint with a column per channel; this emits one row per
    (footprint, channel) on `output_schema`, so it concatenates with what UFSUnifiedLoader emits.
    Everything below happens in between, and the defaults are the training configuration
    rather than a neutral view of the archive.

    Rows dropped, in order:
      1. Scan positions outside `fov_keep_range[sensor]`, ahead of thinning so that winners
         are drawn from survivors.
      2. Footprints with a null `hpx2048_nest`, which carry no position either.
      3. All but one footprint per (sensor, window, platform, `thin_nside` pixel), by `winner`.
         Thinning runs per row group, so a sample spanning several windows keeps one footprint
         per cell per window.
      4. Footprints with a non-finite value in any of GEOMETRY_COLUMNS, in every channel,
         since geometry belongs to the footprint. Unconditional: these reach the featurizer
         as NaN otherwise.
      5. Individual (footprint, channel) cells outside the channel's `min_valid..max_valid`,
         which also catches nulls, NaN failing both comparisons.

    Values transformed:
      - Radiance sensors are inverted to brightness temperature under `value_encoding`, unless
        `ir_units='radiance'`. The range check is in kelvin either way.
      - `Observation` is z-scored per channel against `channel_table`, unless `normalize=False`.
      - `Longitude` moves from the archive's -180..180 to the 0..360 convention (matching UFS)
      - `Sat_Zenith_Angle` gains GSI's sign, negative on the first half of the scan; the archive
        publishes an unsigned magnitude.
      - `Scan_Angle` is derived from the scan identity, having no archive column.
      - `Platform_ID` is the unified vocabulary's, not BUFR SAID.
      - IR sounders contribute only what `ir_channels` selects, by default a preset drawn from
        `ir_spectral.ir_channel_sets` rather than the published axis.
      - NULL_COLUMNS are filled with nulls, being conventional-only fields.

    Not read here: every quality flag, the scene descriptors, the collocated imager groups,
    azimuth angles, QC, etc. Relevant fields/QC flags:
        AIRS  channel_auxiliary.acquisition_quality  ACQF, BUFR 0-33-032, per channel per footprint.
                                                    The only flag in the archive that marks values
                                                    which are present and wrong. Identifies channel popping.
        CrIS  band_calibration_quality.fov_quality   NFQF, 0-33-077, per band per footprint. Decode bits 2, 5 and 10 only:
                                                    bit 9 is a day/night indicator that fires on
                                                    half the archive. Identifies real sensor failures (ex: N21 failure on 2025-03-02).
                band_calibration_quality
                .calibration_quality                 NCQF, 0-33-076, fires on 0.0007-0.015%.
                scan_quality                         NSQF, 0-33-075, coarser: something on the scan
                                                    is degraded, without saying what.
                geolocation_quality, quality_mark, radiance_type_flags, track_qualifier
        IASI  summary_quality                        QGFQ, 0-33-060, 2 bits per footprint. Does not
                                                    identify damaged present values; flagged
                                                    footprints reconstruct no worse than clean.
                geometric_quality                      nonzero on 0.7-2.0% of footprints.
                instrument_noise_quality, radiometric_calibration_quality,
                spectral_calibration_quality, system_quality_function
                                                    pinned at 1 on every row tested; empty.
        Scene descriptors left unread: land_fraction, land_sea_qualifier, total_cloud_cover,
        cloud_top_height, surface_height and orbit_height on CrIS; total_cloud_cover and
        station_elevation on AIRS; station_elevation on IASI. Also the collocated imager groups
        (collocated_imager_scenes, collocated_avhrr_scenes, and AIRS's visible_statistics and
        companion AMSU-A/HSB records).
    """

    def __init__(
        self,
        sensors: Sequence[str],
        channel_table_path: str | None = None,
        archive_root: str = nnja_archive("parquet"),
        obs_context_hours: tuple[int, int] = (-3, 3),
        data_spacing: int = 3,
        thin_nside: int | None = None,
        winner: str = "random",
        winner_seed: int = 0,
        read_ahead: bool = True,
        fov_keep_range: Mapping[str, tuple[int, int]] | None = None,
        ir_channels: str | Mapping[str, Sequence[int]] | None = "ir32",
        ir_units: str = "kelvin",
        normalize: bool = True,
        platform_channel_dropout: Sequence[tuple[str, str, int, float]] = (),
        dropout_scope: str = "row",
    ) -> None:
        """
        Args:
            sensors: NNJA archive sensor names to load.
            channel_table_path: Per-channel stats to normalize against. None (default) derives them
                from the registry, preventing divergence of a stale table from the stats.
            archive_root: Root of the daily wide NNJA Parquet archive. A thinned store under it
                serving `ir_channels` is discovered and preferred, per `_sensor_root`.
            obs_context_hours: Hours relative to target time, as in UFSUnifiedLoader.
            data_spacing: Hours between DA windows.
            thin_nside: Retain one footprint per (sensor, platform, NESTED pixel)
                at this nside. None keeps every footprint.
            winner: Which footprint of a cell to retain, one of WINNER_POLICIES.
            winner_seed: Seed for 'random', and for the draw sequence of 'resample'.
            read_ahead: Fetch the next row group while the current one converts.
            fov_keep_range: Inclusive scan positions per sensor, the model range being
                `SCAN_GEOMETRY[sensor].keep`. Screens before thinning, so thinning picks
                winners among survivors.
            ir_channels: Which channels of the IR sounders to read, microwave always contributing
                its full axis. A preset name of `ir_spectral.IR_CHANNEL_PRESETS`, or a mapping of
                sounder to source channel numbers, sounders it omits keeping their full axis, or
                None for every published channel.
            ir_units: 'kelvin', or 'radiance' to leave the radiance sounders in
                mW/(m2 sr cm-1) as published. Screening is in kelvin either way.
            normalize: Apply per-channel z-score normalization.
            platform_channel_dropout: Training-time rules of
                ``(sensor, platform, raw_channel_id, probability)``. Dropout is
                applied to model-facing rows after footprint thinning.
        """
        for sensor in sensors:
            if sensor not in SENSOR_VALUE_PREFIX:
                raise ValueError(f"sensor {sensor!r} has no NNJA archive mapping")
            if sensor not in SENSOR_CONFIGS:
                raise ValueError(f"sensor {sensor!r} is not in the NNJA vocabulary")
        if winner not in WINNER_POLICIES:
            raise ValueError(f"winner must be one of {WINNER_POLICIES}, got {winner!r}")
        if ir_units not in ("kelvin", "radiance"):
            raise ValueError(
                f"ir_units must be 'kelvin' or 'radiance', got {ir_units!r}"
            )
        # channel_table's mean and stddev are kelvin, so they say nothing about a radiance.
        if ir_units == "radiance" and normalize:
            raise ValueError("ir_units='radiance' needs normalize=False")
        if thin_nside is not None:
            order = int(thin_nside).bit_length() - 1
            if 1 << order != int(thin_nside) or not 0 <= order <= ARCHIVE_HPX_ORDER:
                raise ValueError(
                    f"thin_nside must be a power of two up to "
                    f"{1 << ARCHIVE_HPX_ORDER}, got {thin_nside}"
                )
        self.channel_table_path = channel_table_path
        self.sensors = tuple(sensors)
        self.archive_root = archive_root
        self.obs_context_hours = obs_context_hours
        self.data_spacing = data_spacing
        self.thin_nside = thin_nside
        self.winner = winner
        self.winner_seed = winner_seed
        # Redrawn per sel_time under 'resample', fixed otherwise. One draw covers the
        # whole call, so the seed never changes between the sensors of a batch.
        self._current_seed = np.uint64(winner_seed)
        self._seed_rng = np.random.default_rng(winner_seed)
        if dropout_scope not in ("row", "sample"):
            raise ValueError(
                f'dropout_scope must be "row" or "sample", got {dropout_scope!r}'
            )
        self.dropout_scope = dropout_scope
        self.read_ahead = read_ahead
        # Restating instrument info in the NNJA archive's own scan numbering, which is what
        # screening compares. Only IASI moves: each of its 60 positions is a pair of source FOVs,
        # so a keep of (5, 56) screens on (9, 112). CrIS groups 9 detectors per look but publishes
        # the look separately as field_of_regard, which `select` screens on, so its range stands.
        self.fov_keep_range = {
            sensor: scan_geometry.keep_range_in_source_units(sensor, keep)
            for sensor, keep in (fov_keep_range or {}).items()
        }
        self.ir_channels = _resolve_ir_channels(ir_channels)
        self.ir_units = ir_units
        self.normalize = normalize
        dropout_rules: dict[str, list[tuple[int, int, float]]] = {}
        seen_dropout_rules = set()
        for sensor, platform, raw_channel_id, probability in platform_channel_dropout:
            if sensor not in self.sensors:
                raise ValueError(
                    f"dropout rule sensor {sensor!r} is not loaded by this loader"
                )
            if platform not in PLATFORM_NAME_TO_ID:
                raise ValueError(f"unknown dropout platform: {platform!r}")
            platform_id = PLATFORM_NAME_TO_ID[platform]
            if platform_id not in SENSOR_CONFIGS[sensor].platform_ids:
                raise ValueError(
                    f"{sensor} is not available on dropout platform {platform!r}"
                )
            if raw_channel_id not in SENSOR_CONFIGS[sensor].channels:
                raise ValueError(f"{sensor} has no raw channel {raw_channel_id}")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"dropout probability must be in [0, 1], got {probability}"
                )
            global_channel_id = int(get_global_channel_id(sensor, [raw_channel_id])[0])
            key = (sensor, platform_id, global_channel_id)
            if key in seen_dropout_rules:
                raise ValueError(
                    "duplicate platform/channel dropout rule for "
                    f"{(sensor, platform, raw_channel_id)}"
                )
            seen_dropout_rules.add(key)
            dropout_rules.setdefault(sensor, []).append(
                (platform_id, global_channel_id, probability)
            )
        self._platform_channel_dropout = {
            sensor: tuple(rules) for sensor, rules in dropout_rules.items()
        }
        self._read_plans: dict[str, ReadPlan] = {}
        self._sensor_roots: dict[str, str] = {}
        configure_arrow_pools()

    @functools.cached_property
    def channel_table(self) -> pa.Table:
        if self.channel_table_path is None:
            return nnja_channel_table()
        return pq.read_table(self.channel_table_path)

    def wanted_channels(self, sensor: str, published_ids: Sequence[int]) -> set[int]:
        named = None if self.ir_channels is None else self.ir_channels.get(sensor)
        return set(published_ids if named is None else named)

    def select_channels(
        self, sensor: str, day: str, published: tuple[str, ...], prefix: str
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
        """Narrow a published channel axis to the wanted channels, aligned against the stats table.

        Returns the columns to read, their source channel numbers, their global channel ids, and
        the `channel_table` row each draws its statistics from. Only `wanted_channels` distinguishes
        an IR sounder from a microwave sensor; every sensor needs the rest.
        """
        source_ids = [int(name[len(prefix) :]) for name in published]
        global_ids = get_global_channel_id(sensor, np.array(source_ids)).tolist()
        row_by_global_id = {
            int(value): row
            for row, value in enumerate(
                self.channel_table[GLOBAL_CHANNEL_ID.name].to_pylist()
            )
        }
        wanted = self.wanted_channels(sensor, source_ids)

        columns, kept_source, kept_global, kept_rows = [], [], [], []
        absent = set(wanted)
        for column, source_id, global_id in zip(published, source_ids, global_ids):
            row = row_by_global_id.get(global_id)
            if source_id not in wanted or row is None:
                continue
            columns.append(column)
            kept_source.append(source_id)
            kept_global.append(global_id)
            kept_rows.append(row)
            absent.discard(source_id)
        # Dropping a wanted channel the table has no stats for would narrow the model's
        # input width in silence.
        if absent:
            raise ValueError(
                f"{sensor} {day}: {len(absent)} of {len(wanted)} channels are absent from "
                f"the archive or the channel table, starting {sorted(absent)[:8]}"
            )
        return (
            tuple(columns),
            np.asarray(kept_source),
            np.asarray(kept_global, dtype=np.uint16),
            np.asarray(kept_rows),
        )

    def read_plan(self, sensor: str, day: str, parquet: pq.ParquetFile) -> ReadPlan:
        """Resolve the sensor's channel axis from one published file, then reuse it per sensor.

        Built from whichever day is read first, so it assumes a sensor's channel columns and value
        encoding hold across its whole archive. They do, on every day of the two radiance sensors,
        where a drifting encoding would mis-scale in silence.
        """
        cached = self._read_plans.get(sensor)
        if cached is not None:
            return cached
        # CrIS's value encoding lives in Arrow field metadata, which the parquet schema drops.
        # Read past the cache: schema_arrow rebuilds on access, 7 ms on the widest sensors.
        schema = parquet.schema_arrow
        prefix = f"{SENSOR_VALUE_PREFIX[sensor]}__ch_"  # align all channels
        published = tuple(name for name in schema.names if name.startswith(prefix))
        if not published:
            raise ValueError(f"{sensor} {day}: no {prefix} columns")

        value_columns, kept_source_ids, global_channel_ids, channel_rows = (
            self.select_channels(sensor, day, published, prefix)
        )
        local_channel_ids = (global_channel_ids - SENSOR_OFFSET[sensor]).astype(
            np.uint16
        )

        def channel_stat(name: str) -> np.ndarray:
            return np.asarray(
                self.channel_table[name].to_numpy(zero_copy_only=False)[channel_rows],
                dtype=np.float32,
            )

        wavenumber = value_encoding = None
        if SENSOR_VALUE_PREFIX[sensor].startswith(RADIANCE_PREFIXES):
            # One centre wavenumber per kept channel, ordered like `value_columns`.
            wavenumber = ir_spectral.wavenumber_cm_inverse(sensor, kept_source_ids)
            # The encoding is per-column metadata that every channel column of every day repeats
            # identically, so whichever column is read gives the scale all of them decode with.
            any_channel = value_columns[0]
            value_encoding = ir_spectral.archive_value_encoding(
                schema.field(any_channel)
            )

        plan = ReadPlan(
            value_columns=value_columns,
            global_channel_id=global_channel_ids.astype(np.uint16),
            local_channel_id=local_channel_ids,
            mean=channel_stat("mean"),
            stddev=channel_stat("stddev"),
            min_valid=channel_stat("min_valid"),
            max_valid=channel_stat("max_valid"),
            wavenumber=wavenumber,
            value_encoding=value_encoding or {},
            scan_columns=tuple(
                name for name in (SCAN_POSITION_COLUMN.get(sensor),) if name is not None
            ),
        )
        self._read_plans[sensor] = plan
        return plan

    def _sensor_root(self, sensor: str) -> str:
        """The narrowest thinned store under `archive_root` holding every channel wanted of
        `sensor`, else `archive_root`.

        Stores sit beside the sensor directories, one per `IR_CHANNEL_PRESETS` name, so an ir48
        store serves an ir32 run. Microwave sensors and full-axis reads have none.
        """
        if sensor in self._sensor_roots:
            return self._sensor_roots[sensor]
        wanted = None if self.ir_channels is None else self.ir_channels.get(sensor)
        root = self.archive_root
        if wanted:
            # IR_CHANNEL_PRESETS is declared narrowest first, so the first store that covers
            # the request is also the cheapest to read.
            for preset in ir_spectral.IR_CHANNEL_PRESETS:
                held = ir_spectral.ir_channel_preset(preset).get(sensor, ())
                store = os.path.join(self.archive_root, preset)
                if all(channel in held for channel in wanted) and os.path.isdir(
                    os.path.join(store, sensor)
                ):
                    root = store
                    break
        self._sensor_roots[sensor] = root
        return root

    def _path(self, sensor: str, day: str) -> str:
        """A thinned store's copy of a day, else the published archive's: the store lags the
        archive between thinning runs, and the caller skips a day it finds no file for.
        """
        path = os.path.join(self._sensor_root(sensor), sensor, f"{day}.parquet")
        if os.path.exists(path):
            return path
        return os.path.join(self.archive_root, sensor, f"{day}.parquet")

    def _interval_times(self, target: datetime) -> pd.DatetimeIndex:
        start, end = self.obs_context_hours
        # DA windows are end-aligned, matching UFSUnifiedLoader.
        start += self.data_spacing
        return pd.date_range(
            target + pd.Timedelta(hours=start),
            target + pd.Timedelta(hours=end),
            freq=f"{self.data_spacing}h",
        )

    @property
    def _winner_mode(self) -> int:
        if self.winner in {"random", "resample"}:
            return _MODE_HASH
        return _MODE_SCORE if self.winner == "synoptic" else _MODE_FIRST

    def _winner_score(
        self, table: pa.Table, candidate_rows: np.ndarray | None
    ) -> np.ndarray:
        """Per-row score for policies that rank on a column rather than a hash."""
        if self.winner != "synoptic":
            # The kernel takes the score array whatever the mode, and needs it typed.
            return np.empty(1, dtype=np.uint32)
        distance = synoptic_distance(_numpy(table["time_utc"]))
        return distance if candidate_rows is None else distance[candidate_rows]

    def _fused_select(
        self,
        table: pa.Table,
        keep_range: tuple[int, int] | None,
        scan_position: np.ndarray,
    ) -> np.ndarray:
        """Winner indices from one compiled pass over the key columns."""
        said = _numpy(table["platform_id"])
        pixels_per_platform = 12 * self.thin_nside * self.thin_nside
        # Cells flatten as code * pixels_per_platform + pixel, so the table must reach
        # the largest code present, and codes rise with SAID: take the largest SAID's
        codes = int(SAID_TO_PLATFORM_CODE[said.max()]) + 1
        best_per_cell = np.full(
            codes * pixels_per_platform, _PACKED_EMPTY, dtype=np.uint64
        )
        # Levels to climb; the kernel shifts by twice this, two bits per quadtree level.
        levels_up = ARCHIVE_HPX_ORDER - (int(self.thin_nside).bit_length() - 1)
        _FUSED_SELECT(
            _numpy(table["hpx2048_nest"]),
            said,
            scan_position,
            SAID_TO_PLATFORM_CODE,
            *(keep_range if keep_range is not None else (0, 0)),
            keep_range is not None,
            np.uint64(2 * levels_up),
            np.int64(pixels_per_platform),
            self._winner_mode,
            self._current_seed,
            self._winner_score(table, None),
            best_per_cell,
        )
        return _winner_rows(best_per_cell, table.num_rows)

    def select(self, table: pa.Table, sensor: str) -> np.ndarray | None:
        """Row indices to keep: screen FOVs, then one footprint per (platform, pixel) if thinning.

        None keeps every row in source order, saving an `arange` when neither is on.

        Called on one row group, which is one (3h) DA window, and each sensor has its own file, so the
        effective key is (sensor, window, platform, pixel). A sample spanning several windows
        therefore holds a footprint per cell per window rather than one per cell. Can bias selection based on
        the WINNER policy.
        """
        pixel_column = table["hpx2048_nest"]
        keep_range = self.fov_keep_range.get(sensor)
        position_column = SCAN_POSITION_COLUMN.get(sensor, "field_of_view")

        # Both paths bound their reduction by the largest platform code present, which is
        # undefined with no rows to take it from.
        if not table.num_rows:
            return np.empty(0, dtype=np.int64)

        # A null pixel widens the column to float64 in numpy, which the kernel's shift rejects at
        # compile time, so a row group holding one takes the staged path. Filling the null instead
        # would let a positionless row win a cell and then drop in transform, losing the cell.
        if (
            self.thin_nside is not None
            and _FUSED_SELECT is not None
            and not pixel_column.null_count
        ):
            with cpu_timing_range("nnja.select.fused"):
                return self._fused_select(
                    table, keep_range, _numpy(table[position_column])
                )

        # Explicit form of the four steps the kernel fuses: screen, key, score, reduce. `None`
        # means everything passes, so a clean row group never materializes a mask.
        with cpu_timing_range("nnja.select.mask"):
            keep_mask = (
                pixel_column.is_valid().to_numpy(zero_copy_only=False)
                if pixel_column.null_count
                else None
            )
            if keep_range is not None:
                scan_position = _numpy(table[position_column])
                in_keep_range = (scan_position >= keep_range[0]) & (
                    scan_position <= keep_range[1]
                )
                keep_mask = (
                    in_keep_range if keep_mask is None else (keep_mask & in_keep_range)
                )

            # `candidate_rows` maps positions in the arrays below back to source rows, so
            # the returned index addresses the original table whether or not screening ran.
            if keep_mask is None:
                if self.thin_nside is None:
                    return None
                candidate_rows = np.arange(table.num_rows, dtype=np.int64)
            else:
                candidate_rows = np.flatnonzero(keep_mask)
                if self.thin_nside is None:
                    return candidate_rows

        # The table had rows but the screen kept none of them, so there is nothing to
        # thin and nothing to take the platform bound from.
        if not candidate_rows.size:
            return candidate_rows

        # NESTED is a quadtree, so coarsening is a right shift of two bits per level
        with cpu_timing_range("nnja.select.key"):
            pixel = _numpy(pixel_column)[candidate_rows].astype(np.int64)
            levels_up = ARCHIVE_HPX_ORDER - (int(self.thin_nside).bit_length() - 1)
            parent_pixel = pixel >> (2 * levels_up)
            platform_code = SAID_TO_PLATFORM_CODE[
                _numpy(table["platform_id"])[candidate_rows]
            ]
            pixels_per_platform = 12 * self.thin_nside * self.thin_nside
            cell = platform_code * pixels_per_platform + parent_pixel

        # Lower score wins its cell; zeros everywhere means the earliest row wins, since
        # the reduction breaks ties by row index.
        with cpu_timing_range("nnja.select.score"):
            mode = self._winner_mode
            if mode == _MODE_HASH:
                # Hashing the source row index, not the position in `candidate_rows`,
                # keeps the choice independent of how many the screen removed.
                score_bits = winner_hash(
                    candidate_rows, int(self._current_seed)
                ).astype(np.uint32)
            elif mode == _MODE_SCORE:
                score_bits = self._winner_score(table, candidate_rows)
            else:
                score_bits = np.zeros(candidate_rows.size, dtype=np.uint32)

        with cpu_timing_range("nnja.select.argmin"):
            winners = dense_argmin_per_key(
                cell, score_bits, (int(platform_code.max()) + 1) * pixels_per_platform
            )
        return candidate_rows[winners]

    def transform(
        self, table: pa.Table, plan: ReadPlan, sensor: str, keep_rows: np.ndarray | None
    ) -> Normalized:
        """Gather selected footprints into a normalized channel-by-footprint matrix.

        Channel columns are gathered one at a time, so only the output matrix is resident,
        and footprint columns are cast here rather than after expand(), at `channels` times
        fewer values. The range check runs on physical values, radiance sensors having been
        inverted to kelvin first, and doubles as a finite check: NaN radiance propogates to NaN kelvin which gets caught.
        A footprint missing any of GEOMETRY_COLUMNS is dropped in every channel.
        """
        footprint_count = table.num_rows if keep_rows is None else keep_rows.size

        # One value per footprint; take() first so casts and geometry skip dropped rows.
        with cpu_timing_range("nnja.transform.footprint"):

            def kept(name: str) -> pa.ChunkedArray:
                column = table[name]
                return column if keep_rows is None else column.take(keep_rows)

            field_of_view = _numpy(kept("field_of_view"))
            field_of_regard = (
                _numpy(kept(SCAN_POSITION_COLUMN[sensor]))
                if sensor in SCAN_POSITION_COLUMN
                else None
            )
            footprint_columns = {}
            for source, unified in FOOTPRINT_COLUMNS.items():
                if source == "platform_id":
                    values = SAID_TO_PLATFORM_ID[_numpy(kept(source))]
                elif source == "field_of_view":
                    values = scan_geometry.scan_angle(
                        sensor, field_of_view, field_of_regard
                    ).astype(np.float32)
                elif source == "satellite_zenith_angle":
                    # The archive stores a magnitude; the unified column follows GSI's
                    # diagnostic, negative on the first half of the scan.
                    values = scan_geometry.signed_zenith(
                        sensor,
                        field_of_view,
                        _numpy(kept(source)),
                        field_of_regard,
                    ).astype(np.float32)
                else:
                    values = _numpy(
                        kept(source).cast(self.output_schema.field(unified).type)
                    )
                    if source == "longitude":
                        # The archive is -180..180, the UFS store 0..360. One table cannot
                        # carry both, and 0..360 is what the rest of healda emits.
                        values = np.mod(values, 360.0)
                footprint_columns[unified] = values

        # A footprint with null geometry is dropped in every channel.
        with cpu_timing_range("nnja.transform.geometry"):
            located = np.isfinite(footprint_columns[GEOMETRY_COLUMNS[0]])
            for name in GEOMETRY_COLUMNS[1:]:
                located &= np.isfinite(footprint_columns[name])
            valid: np.ndarray | None = (
                None
                if located.all()
                else np.broadcast_to(
                    located, (plan.channel_count, footprint_count)
                ).copy()
            )

        # One value per (channel, footprint): fill the matrix a channel at a time, flagging
        # out-of-range cells and zscoring
        observation = np.empty((plan.channel_count, footprint_count), dtype=np.float32)
        # Scratch for the two range comparisons, refilled per channel rather than reallocated.
        in_bound = np.empty(footprint_count, dtype=bool)
        for index, name in enumerate(plan.value_columns):
            with cpu_timing_range("nnja.transform.fetch"):
                column = _numpy(table[name])
                values = column if keep_rows is None else column[keep_rows]

            # Screened in kelvin whatever the emitted unit
            screened = values
            if plan.wavenumber is not None:
                with cpu_timing_range("nnja.transform.planck"):
                    radiance = ir_spectral.radiance_mw(
                        sensor, values, **plan.value_encoding, dtype=np.float32
                    )
                    screened = ir_spectral.brightness_temperature(
                        radiance, plan.wavenumber[index], dtype=np.float32
                    )
                    values = radiance if self.ir_units == "radiance" else screened

            # Two reductions cost less than three masked passes plus the mask. NaN propagates
            # through min and max, so a channel holding a null always takes the masked path.
            with cpu_timing_range("nnja.transform.range"):
                all_in_range = footprint_count > 0 and (
                    screened.min() >= plan.min_valid[index]
                    and screened.max() <= plan.max_valid[index]
                )
                if not all_in_range:
                    if valid is None:
                        # Geometry left nothing to mask, so this channel is the first to need
                        # the matrix.
                        valid = np.ones(
                            (plan.channel_count, footprint_count), dtype=bool
                        )
                    # A row view: the `&=` below lands in `valid`, which is what gets returned.
                    channel_valid = valid[index]
                    np.greater_equal(screened, plan.min_valid[index], out=in_bound)
                    channel_valid &= in_bound
                    np.less_equal(screened, plan.max_valid[index], out=in_bound)
                    channel_valid &= in_bound

            with cpu_timing_range("nnja.transform.normalize"):
                channel_row = observation[index]
                if self.normalize:
                    np.subtract(values, plan.mean[index], out=channel_row)
                    # mul with 1/std runs faster than div
                    np.multiply(channel_row, 1.0 / plan.stddev[index], out=channel_row)
                else:
                    channel_row[:] = values
        return Normalized(observation, valid, footprint_columns)

    def expand(self, normalized: Normalized, plan: ReadPlan, sensor: str) -> pa.Table:
        """Expand selected footprints to unified long rows.

        Row order within a window carries no meaning, so the matrix is flattened in its
        own channel-major order. With every cell valid that costs a free `reshape`, one
        `np.tile` per footprint column and one `np.repeat` run per channel identity; a
        validity mask turns the tile into a gather and the run into a per-channel count.
        """
        footprint_count = normalized.footprint_count
        arrays: dict[str, pa.Array] = {}

        # Flatten the matrix to one row per surviving cell, dropping invalid ones.
        with cpu_timing_range("nnja.expand.observation"):
            if normalized.valid is None:
                observation = normalized.values.reshape(-1)
                footprint_rows = None
                counts = None
            else:
                # Per-channel `flatnonzero` gives the footprint index directly; a flat
                # index would need a division and modulo per surviving cell.
                kept_per_channel = [np.flatnonzero(row) for row in normalized.valid]
                counts = np.fromiter(
                    (kept.size for kept in kept_per_channel),
                    dtype=np.int64,
                    count=len(kept_per_channel),
                )
                footprint_rows = np.concatenate(kept_per_channel)
                observation = np.empty(footprint_rows.size, dtype=np.float32)
                offset = 0
                for values, kept in zip(normalized.values, kept_per_channel):
                    np.take(values, kept, out=observation[offset : offset + kept.size])
                    offset += kept.size

        # Repeat each footprint's metadata once per channel that kept it.
        with cpu_timing_range("nnja.expand.footprint"):
            for unified, column in normalized.footprint_columns.items():
                expanded = (
                    np.tile(column, plan.channel_count)
                    if footprint_rows is None
                    else column[footprint_rows]
                )
                arrays[unified] = pa.array(
                    expanded, type=self.output_schema.field(unified).type
                )

        # Repeat each channel's ids once per footprint that kept it.
        with cpu_timing_range("nnja.expand.channel"):
            repeats = footprint_count if counts is None else counts
            global_channel_ids = np.repeat(plan.global_channel_id, repeats)
            local_channel_ids = np.repeat(plan.local_channel_id, repeats)

        # Assemble the unified table: values, ids, and the columns the sat path nulls.
        with cpu_timing_range("nnja.expand.table"):
            long_rows = observation.size
            observation_column = pa.array(observation, type=pa.float32())
            arrays["Observation"] = observation_column
            arrays[GLOBAL_CHANNEL_ID.name] = pa.array(
                global_channel_ids, type=GLOBAL_CHANNEL_ID.type
            )
            arrays[LOCAL_CHANNEL_ID.name] = pa.array(
                local_channel_ids, type=LOCAL_CHANNEL_ID.type
            )
            arrays[SENSOR_ID.name] = pa.array(
                np.full(long_rows, SENSOR_NAME_TO_ID[sensor], dtype=np.uint16),
                type=SENSOR_ID.type,
            )
            null_types = [self.output_schema.field(name).type for name in NULL_COLUMNS]
            for name, column in zip(
                NULL_COLUMNS,
                _all_null_columns(
                    null_types, long_rows, observation_column.buffers()[1]
                ),
            ):
                arrays[name] = column
            table = pa.table(
                [arrays[field.name] for field in self.output_schema],
                schema=self.output_schema,
            )
        return table

    def _scoped_rules(
        self, sensor: str, dropout: SampleDropout
    ) -> tuple[tuple[int, int, float], ...]:
        # (platform_id, global_channel_id, rate), from nnja_platform_channel_dropout.
        return tuple(
            (platform, channel, dropout.rate(probability))
            for platform, channel, probability in self._platform_channel_dropout.get(
                sensor, ()
            )
        )

    def _apply_platform_channel_dropout(
        self,
        table: pa.Table,
        rng: np.random.Generator | None,
        rules: tuple,
    ) -> pa.Table:
        if not rules or not table.num_rows:
            return table

        platform = _numpy(table["Platform_ID"])
        channel = _numpy(table[GLOBAL_CHANNEL_ID.name])
        keep = np.ones(table.num_rows, dtype=bool)
        rng = rng or np.random.default_rng()
        for platform_id, global_channel_id, probability in rules:
            targeted = (platform == platform_id) & (channel == global_channel_id)
            # Serves both scopes: `probability` is the configured p under "row", and 0
            # or 1 under "sample", where this drops the whole channel or keeps it.
            if probability > 0.0 and targeted.any():
                keep[targeted] = rng.random(int(targeted.sum())) >= probability
        return table if keep.all() else table.filter(pa.array(keep))

    def to_long(
        self,
        table: pa.Table,
        plan: ReadPlan,
        sensor: str,
        rng: np.random.Generator | None,
        rules: tuple,
    ) -> pa.Table:
        """Convert one wide row group to long rows: select, then transform, then expand.

        The retained rows are passed into transform so only those are gathered and
        normalized.
        """
        keep_rows = self.select(table, sensor)
        long = self.expand(self.transform(table, plan, sensor, keep_rows), plan, sensor)
        return self._apply_platform_channel_dropout(long, rng, rules)

    def _row_group_jobs(
        self, sensor: str, windows: pd.DatetimeIndex
    ) -> list[tuple[pq.ParquetFile, ReadPlan, pd.Timestamp, int]]:
        jobs = []
        for day in sorted({timestamp.strftime("%Y%m%d") for timestamp in windows}):
            path = self._path(sensor, day)
            if not os.path.exists(path):
                continue
            with cpu_timing_range("nnja.open"):
                # pre_buffer coalesces projected column ranges, ~1.3x on a cold Lustre read.
                parquet = pq.ParquetFile(path, pre_buffer=True)
            with cpu_timing_range("nnja.plan"):
                plan = self.read_plan(sensor, day, parquet)
                window_groups = list(window_row_groups(parquet, windows, path=path))
            for window, group in window_groups:
                jobs.append((parquet, plan, window, group))
        return jobs

    def _load_sensor(
        self,
        sensor: str,
        windows: pd.DatetimeIndex,
        dropout_seed: int | None,
        rules: tuple,
    ) -> list[tuple[pd.Timestamp, pa.Table]]:
        """Read each window of this sensor and convert to long rows.

        `read_ahead` fetches the next row group while the current one converts, overlapping
        I/O and compute. Row groups that survive nothing are
        dropped: a zero-row table still adds a chunk to every column downstream.
        """
        jobs = self._row_group_jobs(sensor, windows)
        rng = row_generator(dropout_seed)

        def fetch(parquet: pq.ParquetFile, plan: ReadPlan, group: int) -> pa.Table:
            with cpu_timing_range("nnja.read"):
                return parquet.read_row_group(group, columns=plan.columns)

        parts: list[tuple[pd.Timestamp, pa.Table]] = []
        if not self.read_ahead or len(jobs) < 2:
            for parquet, plan, window, group in jobs:
                long_table = self.to_long(
                    fetch(parquet, plan, group), plan, sensor, rng, rules
                )
                if long_table.num_rows:
                    parts.append((window, long_table))
            return parts

        # Reads run here, conversion on the calling thread, so the two overlap. Private pool
        # because this call holds a thread_pool() worker and resubmitting there would deadlock.
        with ThreadPoolExecutor(max_workers=1) as pool:
            queued = deque()
            issued = 0

            def submit(job):
                parquet, plan, _window, group = job
                return pool.submit(fetch, parquet, plan, group)

            # One job to convert plus one in flight. A flag rather than a count because the
            # refill keeps the single reader saturated: only depth 0 -> 1 helps
            while issued < 2:
                queued.append((jobs[issued], submit(jobs[issued])))
                issued += 1
            while queued:
                (_parquet, plan, window, _group), pending = queued.popleft()
                wide_table = pending.result()  # blocks only if the reader is behind
                # Refill before converting, so the reader has work during to_long.
                if issued < len(jobs):
                    queued.append((jobs[issued], submit(jobs[issued])))
                    issued += 1
                long_table = self.to_long(wide_table, plan, sensor, rng, rules)
                if long_table.num_rows:
                    parts.append((window, long_table))
        return parts

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]:
        """Load unified long observations for each target time.

        A coroutine so call sites keep UFSUnifiedLoader's `asyncio.run(sel_time(times))`
        form. Sensors read concurrently: one cold Lustre stream saturates at 0.33 GiB/s
        while four together hold 0.21 GiB/s each, 2.5x the aggregate.
        """
        # Windows are matched against the archive's, which `window_row_groups` strips to naive
        # UTC. An aware index would match no row group and come back empty rather than fail.
        if times.tz is not None:
            raise ValueError(f"times must be tz-naive UTC, got {times.tz}")

        needed_windows: set[pd.Timestamp] = set()
        for target in times:
            needed_windows.update(self._interval_times(target))
        windows = pd.DatetimeIndex(sorted(needed_windows))

        if self.winner == "resample":
            self._current_seed = np.uint64(
                self._seed_rng.integers(1 << 64, dtype=np.uint64)
            )

        # Outer of three concurrency levels: a thread per sensor, Arrow's global I/O
        # pool for the reads those threads issue, its global CPU pool for the decode.
        # An `io_thread_count` below the sensor count serializes the reads back down
        # however many sensor threads are waiting here.
        pool = thread_pool()
        loop = asyncio.get_running_loop()
        dropout = self._dropout()
        # Per sensor: a row-level seed, and rules whose rates the scope has resolved.
        seeds = dropout.seeds(self.sensors)
        rules = {s: self._scoped_rules(s, dropout) for s in self.sensors}
        by_sensor = await asyncio.gather(
            *(
                loop.run_in_executor(
                    pool,
                    self._load_sensor,
                    sensor,
                    windows,
                    seeds[sensor],
                    rules[sensor],
                )
                for sensor in self.sensors
            )
        )
        # Adjacent targets' context intervals could overlap, so regroup by window and let each target
        # reference the ones it spans. `concat_tables` appends chunk lists rather than copying.
        by_window: dict[pd.Timestamp, list[pa.Table]] = {}
        for parts in by_sensor:
            for window, long_table in parts:
                by_window.setdefault(window, []).append(long_table)

        # Shared by every target whose windows all came back empty.
        empty_table = pa.table(
            [pa.array([], type=field.type) for field in self.output_schema],
            schema=self.output_schema,
        )

        def combine(target: datetime) -> pa.Table:
            parts = [
                part
                for window in self._interval_times(target)
                for part in by_window.get(window, ())
            ]
            return pa.concat_tables(parts) if parts else empty_table

        return {"obs_v2": [combine(target) for target in times]}


if __name__ == "__main__":
    # Run as `python -m healda.observations.loaders.nnja_wide`. As a path, this file's
    # directory leads sys.path and its `types.py` shadows the stdlib module.
    import time

    import pyarrow.compute as pc

    sensors = ["atms", "amsua", "mhs"]
    sample_time = pd.Timestamp("2022-01-18T21:00:00")

    loader = NNJAWideLoader(
        sensors=sensors,
        obs_context_hours=(-3, +3),
        thin_nside=64,
    )

    windows = loader._interval_times(sample_time)
    days = sorted({window.strftime("%Y%m%d") for window in windows})
    files = [
        (sensor, loader._path(sensor, day))
        for sensor in sensors
        for day in days
        if os.path.exists(loader._path(sensor, day))
    ]
    disk_bytes = sum(os.path.getsize(path) for _, path in files)
    print(f"Read files:   {len(files)} parquet files")
    for sensor, path in files:
        print(f"  {sensor:20s}  {os.path.getsize(path) / 1e6:6.1f} MB  {path}")

    start = time.perf_counter()
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([sample_time])))["obs_v2"][0]
    elapsed = time.perf_counter() - start

    da_col = table.column("DA_window")
    print(f"Rows:    {table.num_rows:,}")
    print(f"DA range: {pc.min(da_col).as_py()} -> {pc.max(da_col).as_py()}")
    print(f"On disk: {disk_bytes / 1e6:.1f} MB")
    print(f"In mem:  {table.nbytes / 1e6:.1f} MB")
    print(f"Load:    {elapsed * 1e3:.0f} ms")

    id_to_name = {SENSOR_NAME_TO_ID[sensor]: sensor for sensor in sensors}
    ids, counts = np.unique(table["sensor_id"].to_numpy(), return_counts=True)
    for sensor_id, count in zip(ids, counts):
        print(f"Sensor {id_to_name[int(sensor_id)]}: {count:,} obs")
