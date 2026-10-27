# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Earth2Studio NNJA frames as the canonical NNJA archive tables.

``earth2studio.data.NNJAObsConv`` (PrepBUFR ``u``, ``v``, ``q``, ``t``, ``pres``;
GPS-RO ``gps``, ``gps_refractivity``), ``NNJAObsSatwnd`` (``u``, ``v``) and
``NNJAObsSat`` return raw decoded observations in one of two layouts
(``layout``):

- ``long``: one row per observation, its variable in ``variable`` and its value in
  ``observation``;
- ``wide``: one row per report level, profile level, wind or footprint, each variable's
  value in ``observation_<variable>`` and, for conventional data, its quality mark in
  ``quality_<variable>``. A satellite frame holds one sensor, each footprint's channels
  as fixed-size lists ``quality`` and ``observation_<variable>`` over the sensor's
  channel axis, which the frame carries in ``attrs["channels"]``.

``analysis_tables`` takes wide frames and ``analysis_tables_long`` long ones. Both
restate them in the column contract of healda's NNJA PrepBUFR, GPS-RO, SATWND and wide
satellite archive tables, deriving what the ETL derives with the same code, so the
loaders screen, thin and emit them exactly as they do archive rows, and both give the
same tables. They return ``DAModel.run_analysis``'s table arguments: each a
``{cycle: table}`` ``CycleTables``, ``satellite_tables`` one per sensor.

The module holds the wide layout and the archive builders both layouts share; the
long layout's regrouping is its last section.

Satellite values stay in kelvin as earth2studio returns them. The IR sounders' columns
are ``brightness_temperature__ch_NNNNN`` with field metadata ``nnja.value_unit = K``,
which ``NNJAWideLoader`` reads without a Planck inversion.

Rows are keyed by ``cycle_time``, the NCEP file each was decoded from, which the
Earth2Studio NNJA sources fill: an observation at the shared edge of two files' windows
is in both files, and stays in both, as in the archive.

Usage, every stream at once from wide frames::

    tables = analysis_tables(
        prepbufr=prepbufr, gpsro=gpsro, satwnd=satwnd, sat={"atms": atms},
        ir_channels=...,
    )
    model.run_analysis(times, **tables)
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from healda.observations.loaders.nnja_base import archive_window
from healda.observations.loaders.nnja_satwnd import ARCHIVE_HPX_LEVEL
from healda.observations.loaders.nnja_wide import (
    ARCHIVE_HPX_ORDER,
    CRIS_FOV_QUALITY_BAD,
    KELVIN_PREFIX,
    RADIANCE_PREFIXES,
    SAID_TO_PLATFORM_NAME,
    SCAN_POSITION_COLUMN,
    SENSOR_VALUE_PREFIX,
    VALUE_UNIT_KEY,
    _resolve_ir_channels,
)
from healda.observations.preprocessing import satwnd_gsi_types
from healda.observations.preprocessing.gpsro_pressure import (
    derive_level_coordinates_batch,
)
from healda.observations.preprocessing.sat_helpers import hpx_nest_pixels
from healda.observations.sensors_nnja import SENSOR_CONFIGS

# Wide frames: a variable's value and quality columns are these prefixes + its name.
WIDE_VALUE = "observation_"
WIDE_QUALITY = "quality_"

GPS_VARIABLE = "gps"
REFRACTIVITY_VARIABLE = "gps_refractivity"
# e2s variable -> PrepBUFR archive value and quality columns.
PREPBUFR_VARIABLES = {
    "q": ("QOB", "QQM"),
    "t": ("TOB", "TQM"),
    "u": ("UOB", "WQM"),
    "v": ("VOB", "WQM"),
}
PREPBUFR_LEVEL_VARIABLES = (*PREPBUFR_VARIABLES, "pres")
# e2s columns that every observation of one report level shares.
PREPBUFR_LEVEL_COLUMNS = [
    "cycle_time",
    "report_time",
    "station",
    "type",
    "report_lat",
    "report_lon",
    "station_elev",
    "level_cat",
    "pres",
    "elev",
    "pressure_quality",
]

# The NNJAObsSat variable holding each sensor's archive value column
# (SENSOR_VALUE_PREFIX); every other sensor's variable is its own name. The ATMS
# archive column is antenna temperature, which e2s `atms` (brightness temperature) is not.
SAT_VARIABLE = {"atms": "atms_antenna_temperature"}

PLATFORM_NAME_TO_SAID = {name: said for said, name in SAID_TO_PLATFORM_NAME.items()}

# NNJAObsSatwnd wind columns; the float ones also identify a wind.
SATWND_FLOAT_KEYS = (
    "lat",
    "lon",
    "pres",
    "satellite_id",
    "wind_method",
    "wind_method_local",
    "height_method",
    "satellite_za",
    "quality",
)
SATWND_WIND_COLUMNS = ("cycle_time", "time", "subset", *SATWND_FLOAT_KEYS)
WIDE_SATWND_COLUMNS = (*SATWND_WIND_COLUMNS, f"{WIDE_VALUE}u", f"{WIDE_VALUE}v")
# Columns constant over one footprint's channel rows.
FOOTPRINT_KEYS = (
    "cycle_time",
    "time",
    "lat",
    "lon",
    "satellite",
    "scan_position",
    "scan_line",
    "detector",
    "satellite_za",
    "solza",
)
# NNJAObsSat footprint flag columns -> the archive's; null for sensors that lack them.
FOOTPRINT_FLAGS = {
    "scan_quality": "scan_quality",
    "granule_quality": "granule_quality",
    "footprint_quality": "summary_quality",
}
# Footprint columns read when a sensor's frame has them.
OPTIONAL_FOOTPRINT_COLUMNS = ("scan_line", "detector", *FOOTPRINT_FLAGS)


def _integer(values: np.ndarray, pa_type: pa.DataType) -> pa.Array:
    """Float values with NaN as an integer column with nulls."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    integers = np.where(valid, values, 0).astype(np.int64)
    return pa.array(integers, mask=~valid).cast(pa_type)


def _strings(codes: np.ndarray, dictionary: pa.Array) -> pa.Array:
    """``dictionary[codes]`` as a string column, null where the code is -1."""
    indices = pa.array(np.maximum(codes, 0).astype(np.int32), mask=codes < 0)
    return pa.DictionaryArray.from_arrays(indices, dictionary).cast(pa.string())


def _gpsro_wide(table: pa.Table) -> pa.Table:
    return _gpsro_archive(
        table,
        is_gps=_wide_present(table, GPS_VARIABLE),
        is_profile=_wide_present(table, REFRACTIVITY_VARIABLE),
        bending=_wide_floats(table, WIDE_VALUE + GPS_VARIABLE),
        refractivity=_wide_floats(table, WIDE_VALUE + REFRACTIVITY_VARIABLE),
        qfro=_wide_floats(table, WIDE_QUALITY + GPS_VARIABLE),
    )


def _gpsro_archive(
    table: pa.Table,
    *,
    is_gps: np.ndarray,
    is_profile: np.ndarray,
    bending: np.ndarray,
    refractivity: np.ndarray,
    qfro: np.ndarray,
) -> pa.Table:
    radius = _floats(table["radius_curvature"])
    geoid = _floats(table["geoid_undulation"])
    elev = _floats(table["elev"])
    lat = _floats(table["lat"])
    lon = _floats(table["lon"])
    time = _native(table["time"])
    cycle = _native(table["cycle_time"])
    kind = _floats(table["type"])
    station, stations = _codes(table["station"])

    derived = {
        name: np.full(table.num_rows, np.nan)
        for name in (
            "impact_height_m",
            "heit_at_impact_m",
            "dry_pressure_hpa",
            "standard_atmosphere_pressure_hpa",
            "blended_pressure_5km_hpa",
        )
    }
    occ_lat = np.full(table.num_rows, np.nan)
    occ_lon = np.full(table.num_rows, np.nan)
    # An occultation is one message: its file, header time, receiver/transmitter and
    # radius of curvature.
    members = np.flatnonzero(is_gps | is_profile)
    if not members.size:
        return pa.table({})
    keys = [cycle, time, _group_key(kind), station, _group_key(radius)]
    group = _group_ids([key[members] for key in keys])[0]
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.diff(group[order])) + 1
    level_rows, profile_rows, occ_header = [], [], []
    for positions in np.split(order, bounds):
        rows = members[positions]
        levels = rows[is_gps[rows]]
        if not levels.size:
            continue
        profile = rows[is_profile[rows]]
        # Refractivity rows sit at the occultation's reference point.
        header_lat = lat[profile[0]] if profile.size else np.nan
        header_lon = lon[profile[0]] if profile.size else np.nan
        occ_lat[levels], occ_lon[levels] = header_lat, header_lon
        level_rows.append(levels)
        profile_rows.append(profile)
        occ_header.append((header_lat, radius[levels[0]], geoid[levels[0]]))
    if level_rows:
        all_levels = np.concatenate(level_rows)
        all_profiles = np.concatenate(profile_rows)
        header_lat, elrc, geodu = (
            np.asarray(v, dtype=np.float64) for v in zip(*occ_header)
        )
        columns = derive_level_coordinates_batch(
            elev[all_levels] + np.repeat(elrc, [r.size for r in level_rows]),
            np.cumsum([0, *(r.size for r in level_rows)]),
            elev[all_profiles],
            refractivity[all_profiles],
            np.cumsum([0, *(r.size for r in profile_rows)]),
            latitude_degrees=header_lat,
            earth_radius_of_curvature_m=np.where(np.isfinite(elrc), elrc, np.nan),
            geoid_undulation_m=np.where(np.isfinite(geodu), geodu, 0.0),
        )
        for name, values in columns.items():
            derived[name][all_levels] = values

    # The transmitter is the station id after its first four characters.
    names = stations.to_pylist()
    transmitter_by_station = np.array(
        [
            *pd.to_numeric(pd.Series(names, dtype=object).str[4:], errors="coerce"),
            np.nan,
        ],
        dtype=np.float64,
    )
    gps = np.flatnonzero(is_gps)
    columns = {
        "satellite_id": _integer(kind[gps], pa.int64()),
        "transmitter_id": _integer(transmitter_by_station[station[gps]], pa.int64()),
        "occ_latitude": occ_lat[gps],
        "occ_longitude": occ_lon[gps],
        "tangent_latitude": lat[gps],
        "tangent_longitude": lon[gps],
        "qfro": _integer(qfro[gps], pa.int64()),
        "earth_radius_curvature": radius[gps],
        "geoid_undulation": geoid[gps],
        "obs_time": pa.array(time[gps]).cast(pa.timestamp("ns")),
        "impact_parameter": elev[gps] + radius[gps],
        "bending_angle": bending[gps].astype(np.float32),
        **{name: values[gps] for name, values in derived.items()},
    }
    out = pa.table(
        {
            name: values
            if isinstance(values, pa.Array)
            else pa.array(values, from_pandas=True)
            for name, values in columns.items()
        }
    )
    return out


def _satwnd_winds(table: pa.Table) -> tuple[dict[str, np.ndarray], pa.Array]:
    subset, subset_names = _codes(table["subset"])
    winds = {
        "cycle_time": _native(table["cycle_time"]),
        "time": _native(table["time"]),
        "subset": subset,
    }
    winds.update((name, _floats(table[name])) for name in SATWND_FLOAT_KEYS)
    return winds, subset_names


def _satwnd_wide(table: pa.Table) -> pa.Table:
    winds, subset_names = _satwnd_winds(table)
    u = _floats(table[f"{WIDE_VALUE}u"])
    v = _floats(table[f"{WIDE_VALUE}v"])
    return _satwnd_archive(winds, subset_names, u, v)


def _satwnd_archive(
    winds: dict[str, np.ndarray], subset_names: pa.Array, u: np.ndarray, v: np.ndarray
) -> pa.Table:
    cycle = winds["cycle_time"].astype("datetime64[ns]")
    window = archive_window(winds["time"], cycle)
    pixels, valid_position = _pixels(winds["lat"], winds["lon"], ARCHIVE_HPX_LEVEL)
    subset = winds["subset"]
    swcm = winds["wind_method"]
    types = satwnd_gsi_types.resolve_codes(
        subset,
        subset_names.to_pylist(),
        winds["satellite_id"],
        swcm,
        winds["wind_method_local"],
    )
    # The archive keeps SWCM as read, with CMCM only for templates lacking it.
    known = subset >= 0
    has_swcm = np.bincount(
        subset[known], weights=np.isfinite(swcm[known]), minlength=len(subset_names)
    )
    use_swcm = np.ones(swcm.size, dtype=bool)
    use_swcm[known] = has_swcm[subset[known]] > 0
    method = np.where(use_swcm, swcm, winds["wind_method_local"])
    tier_names = pa.array(satwnd_gsi_types.TIER_NAMES, pa.string())

    table = pa.table(
        {
            "time_utc": pa.array(winds["time"]).cast(pa.timestamp("ns")),
            "da_window": pa.array(window),
            "cycle": pa.array(cycle),
            "latitude": winds["lat"],
            "longitude": winds["lon"],
            "hpx4096_nest": pa.array(pixels, mask=~valid_position),
            "wind_u_derived": u,
            "wind_v_derived": v,
            "assigned_pressure": pa.array(winds["pres"], from_pandas=True),
            "satellite_id": _integer(winds["satellite_id"], pa.uint32()),
            "wind_computation_method": _integer(method, pa.uint32()),
            "ncep_dump_subtype": _strings(subset, subset_names),
            "sdmedit_wind_quality_mark": _integer(winds["quality"], pa.uint32()),
            "satellite_zenith_angle": pa.array(winds["satellite_za"], from_pandas=True),
            "gsi_observation_type": _integer(
                np.where(types.report_type >= 0, types.report_type, np.nan), pa.uint16()
            ),
            "gsi_internal_subtype": _integer(
                np.where(types.internal_subtype >= 0, types.internal_subtype, np.nan),
                pa.uint8(),
            ),
            "gsi_type_mapping_tier": _strings(types.tier_index, tier_names),
        }
    )
    return table


def _prepbufr_wide(table: pa.Table) -> pa.Table:
    present = {name: _wide_present(table, name) for name in PREPBUFR_LEVEL_VARIABLES}
    rows = np.flatnonzero(np.logical_or.reduce(list(present.values())))
    if not rows.size:
        return pa.table({})
    values = {
        name: _wide_floats(table, WIDE_VALUE + name)[rows]
        for name in PREPBUFR_VARIABLES
    }
    qualities = {
        name: _wide_floats(table, WIDE_QUALITY + name)[rows]
        for name in PREPBUFR_VARIABLES
    }
    present = {name: present[name][rows] for name in PREPBUFR_VARIABLES}
    stations = _codes(table["station"])[1]
    header = _prepbufr_levels(table, rows)
    return _prepbufr_archive(header, stations, values, qualities, present)


def _prepbufr_levels(table: pa.Table, rows: np.ndarray) -> dict[str, np.ndarray]:
    """``rows``' level columns, ``station`` as codes."""
    levels = {"station": _codes(table["station"])[0][rows]}
    for name in ("cycle_time", "report_time", "time"):
        levels[name] = _native(table[name])[rows]
    for name in (*PREPBUFR_LEVEL_COLUMNS, "lat", "lon"):
        if name not in levels:
            levels[name] = _floats(table[name])[rows]
    return levels


def _prepbufr_archive(
    header: dict, stations: pa.Array, values: dict, qualities: dict, present: dict
) -> pa.Table:
    """One archive row per level: its ``_prepbufr_levels`` columns, station codes
    indexing ``stations``, and per variable the value in e2s units (K, kg/kg, m/s), the
    quality mark, and whether the level holds it.

    The rows come from ``NNJAObsConv(original_event=True)``: the archive holds each
    observation's original event (program code 1), while the latest event is for
    temperature often the virtual temperature, with later quality marks, which the
    loader reads as a different observation. ``XOB``, ``YOB`` and ``DHR`` are the
    report header's position and time; ``XDR``, ``YDR`` and ``HRDR`` the level's, the
    header's except for drifting radiosonde levels.
    """
    cycle = header["cycle_time"].astype("datetime64[ns]")

    def hours_from_cycle(time_ns):
        return (time_ns - cycle.view(np.int64)) / 3_600_000_000_000

    columns = {
        "XOB": header["report_lon"],
        "YOB": header["report_lat"],
        "DHR": hours_from_cycle(header["report_time"]),
        "XDR": header["lon"],
        "YDR": header["lat"],
        "HRDR": hours_from_cycle(header["time"]),
        "ELV": header["station_elev"],
        "TYP": header["type"],
        "CAT": header["level_cat"],
        # e2s returns Pa; PrepBUFR stores hPa.
        "POB": header["pres"] / 100.0,
        "PQM": header["pressure_quality"],
        "ZOB": header["elev"],
        # e2s returns kg/kg and K; PrepBUFR stores QOB in whole mg/kg and TOB in 0.1 degC.
        "QOB": np.rint(values["q"] * 1e6),
        "QQM": qualities["q"],
        "TOB": np.round(values["t"] - 273.15, 1),
        "TQM": qualities["t"],
        "UOB": values["u"],
        "VOB": values["v"],
        # One wind quality mark per level: v's where the level has v.
        "WQM": np.where(present["v"], qualities["v"], qualities["u"]),
    }
    table = pa.table(
        {
            "SID": _strings(header["station"], stations),
            **{
                name: pa.array(values.astype(np.float32), from_pandas=True)
                for name, values in columns.items()
            },
        }
    )
    return table


def _sat_channels(sensor: str, ir_channels) -> tuple[np.ndarray, np.ndarray]:
    """The archive channels of ``sensor``, ascending, and a channel -> slot lookup."""
    wanted = (_resolve_ir_channels(ir_channels) or {}).get(sensor)
    channels = np.array(
        sorted(SENSOR_CONFIGS[sensor].channels if wanted is None else wanted),
        dtype=np.int64,
    )
    channel_slot = np.full(channels.max() + 1, -1, dtype=np.int64)
    channel_slot[channels] = np.arange(channels.size)
    return channels, channel_slot


def _slots(sensor_index: np.ndarray, channel_slot: np.ndarray) -> np.ndarray:
    """Each channel's slot in the archive's channel axis, -1 for channels it lacks."""
    known = (sensor_index >= 0) & (sensor_index < channel_slot.size)
    return np.where(known, channel_slot[np.where(known, sensor_index, 0)], -1)


def _sat_wide(table: pa.Table, sensor: str, axis: np.ndarray, ir_channels) -> pa.Table:
    """Archive rows of one sensor's wide frame, whose per-channel columns hold one slot
    per channel of ``axis``."""
    channels, channel_slot = _sat_channels(sensor, ir_channels)
    slot = _slots(axis, channel_slot)
    if not (slot >= 0).any():
        return pa.table({})
    observation = _list_channel_rows(
        table[WIDE_VALUE + SAT_VARIABLE.get(sensor, sensor)], slot, channels.size
    )
    flags = _list_or_flags(table["quality"], slot)
    footprints = {
        name: _native(table[name])
        for name in (*FOOTPRINT_KEYS, *FOOTPRINT_FLAGS)
        if name in table.column_names
    }
    platforms = _codes(table["satellite"])[1]
    return _sat_archive(sensor, footprints, platforms, channels, observation, flags)


def _list_width(column: pa.ChunkedArray) -> int:
    if not pa.types.is_fixed_size_list(column.type):
        raise ValueError(
            "wide satellite columns are fixed-size lists over the channel axis, as "
            f"Arrow-backed earth2studio frames hold them; got {column.type}"
        )
    return column.type.list_size


def _list_blocks(column: pa.ChunkedArray) -> Iterator[tuple[int, pa.Array]]:
    """(first footprint, flattened values) of each chunk of a fixed-size list column."""
    if column.null_count:
        raise ValueError("satellite channel lists must not be null")
    start = 0
    for chunk in column.chunks:
        yield start, chunk.flatten()
        start += len(chunk)


# Transposes and reductions run in row ranges that stay in cache.
_ROWS_PER_PASS = 1 << 16


def _transpose_into(
    out: np.ndarray, block: np.ndarray, slot: np.ndarray, start: int
) -> None:
    """Write (footprint, slot) ``block`` into ``out``'s channel rows from column ``start``."""
    columns = np.flatnonzero(slot >= 0)
    for lo in range(0, len(block), _ROWS_PER_PASS):
        hi = min(lo + _ROWS_PER_PASS, len(block))
        out[slot[columns], start + lo : start + hi] = block[lo:hi, columns].T


def _or_into(out: np.ndarray, block: np.ndarray, slot: np.ndarray, start: int) -> None:
    """Write the bitwise OR of each footprint's archive-channel flags from ``start``."""
    columns = np.flatnonzero(slot >= 0)
    for lo in range(0, len(block), _ROWS_PER_PASS):
        hi = min(lo + _ROWS_PER_PASS, len(block))
        chunk = np.ascontiguousarray(block[lo:hi, columns].T)
        out[start + lo : start + hi] = np.bitwise_or.reduce(chunk, axis=0)


def _channel_rows(
    values: np.ndarray, width: int, slot: np.ndarray, channel_count: int
) -> np.ndarray:
    """(channel, footprint) values from footprints of ``width`` consecutive entries
    whose channels sit at ``slot``; NaN where a footprint lacks a channel."""
    block = values.reshape(-1, width)
    out = np.full((channel_count, block.shape[0]), np.nan, dtype=values.dtype)
    _transpose_into(out, block, slot, 0)
    return out


def _or_flags(quality: np.ndarray, width: int, slot: np.ndarray) -> np.ndarray:
    """Per footprint of ``width`` entries, the bitwise OR of its archive channels' flags."""
    block = quality.reshape(-1, width)
    out = np.empty(block.shape[0], dtype=np.int64)
    _or_into(out, block, slot, 0)
    return out


def _list_channel_rows(
    column: pa.ChunkedArray, slot: np.ndarray, channel_count: int
) -> np.ndarray:
    """``_channel_rows`` of a fixed-size list column, chunk by chunk, as float32."""
    width = _list_width(column)
    out = np.full((channel_count, len(column)), np.nan, dtype=np.float32)
    for start, values in _list_blocks(column):
        block = values.to_numpy(zero_copy_only=False).reshape(-1, width)
        _transpose_into(out, block, slot, start)
    return out


def _list_or_flags(column: pa.ChunkedArray, slot: np.ndarray) -> np.ndarray:
    """``_or_flags`` of a fixed-size list column, chunk by chunk; null flags are 0."""
    width = _list_width(column)
    out = np.empty(len(column), dtype=np.int64)
    for start, values in _list_blocks(column):
        _or_into(
            out, pc.fill_null(values, 0).to_numpy().reshape(-1, width), slot, start
        )
    return out


def _unsigned(column: pa.ChunkedArray) -> np.ndarray:
    """An unsigned flag column in its own dtype, 0 for null."""
    return pc.fill_null(column, 0).to_numpy()


def _sat_archive(
    sensor: str,
    footprints: dict[str, np.ndarray],
    platforms: pa.Array,
    channels: np.ndarray,
    observation: np.ndarray,
    flags: np.ndarray | None,
) -> pa.Table:
    """One archive row per footprint from its ``footprints`` columns (``satellite`` as
    codes into ``platforms``), the (channel, footprint) values in kelvin and, when the
    frame carries per-channel quality, each footprint's bitwise OR of it over the
    archive channels."""
    time_ns = footprints["time"]
    cycle = footprints["cycle_time"].astype("datetime64[ns]")
    window = archive_window(time_ns, cycle)
    lat = footprints["lat"].astype(np.float64)
    lon = footprints["lon"].astype(np.float64)
    pixels, valid_position = _pixels(lat, lon, ARCHIVE_HPX_ORDER)
    said_by_code = np.array(
        [PLATFORM_NAME_TO_SAID.get(name, -1) for name in platforms.to_pylist()] + [-1]
    )
    said = said_by_code[footprints["satellite"]]
    if (said < 0).any():
        unknown = sorted(
            {str(platforms[c]) for c in np.unique(footprints["satellite"][said < 0])}
        )
        raise ValueError(f"{sensor}: no SAID for satellites {unknown}")

    def integer(name, pa_type):
        values = footprints[name]
        return pa.array(np.maximum(values, 0), mask=values < 0).cast(pa_type)

    scan_columns = {"field_of_view": integer("scan_position", pa.uint16())}
    if sensor in SCAN_POSITION_COLUMN:
        # The detector within the look, which the archive names field_of_view.
        if "detector" not in footprints:
            raise ValueError(f"{sensor} needs the frame's detector column")
        scan_columns = {
            "field_of_view": integer("detector", pa.uint16()),
            SCAN_POSITION_COLUMN[sensor]: integer("scan_position", pa.uint16()),
        }

    utc = pa.timestamp("ns", tz="UTC")
    columns = {
        "latitude": pa.array(lat),
        "longitude": pa.array(lon),
        "time_utc": pa.array(time_ns).cast(pa.timestamp("ns")).cast(utc),
        "da_window": pa.array(window).cast(utc),
        "platform_id": pa.array(said, pa.uint16()),
        "satellite_zenith_angle": pa.array(
            footprints["satellite_za"].astype(np.float32), from_pandas=True
        ),
        "solar_zenith_angle": pa.array(
            footprints["solza"].astype(np.float32), from_pandas=True
        ),
        **scan_columns,
        "hpx2048_nest": pa.array(pixels, mask=~valid_position),
        **_quality_columns(sensor, flags),
        **_footprint_flags(footprints),
    }
    fields = [pa.field(name, values.type) for name, values in columns.items()]
    arrays = list(columns.values())
    prefix, metadata = SENSOR_VALUE_PREFIX[sensor], None
    if prefix in RADIANCE_PREFIXES:
        prefix, metadata = KELVIN_PREFIX, {VALUE_UNIT_KEY: b"K"}
    for channel, values in zip(channels, observation):
        arrays.append(pa.array(values, from_pandas=True))
        fields.append(
            pa.field(f"{prefix}__ch_{channel:05d}", pa.float32(), metadata=metadata)
        )
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    return table


def _pixels(
    lat: np.ndarray, lon: np.ndarray, level: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return hpx_nest_pixels(lat, lon, level, device=device)


def _footprint_flags(footprints: dict[str, np.ndarray]) -> dict:
    """The archive's footprint flag columns from those NNJAObsSat carries."""
    columns = {}
    for name, archive_name in FOOTPRINT_FLAGS.items():
        if name not in footprints:
            continue
        values = footprints[name].astype(np.float64)
        values[values < 0] = np.nan
        if np.isfinite(values).any():
            columns[archive_name] = _integer(values, pa.uint32())
    return columns


def _quality_columns(sensor: str, flags: np.ndarray | None) -> dict:
    """The provider flags NNJAObsSat's per-channel quality carries, as the loader's
    ``SOURCE_QUALITY_FLAGS`` read them, from the OR of a footprint's channel flags:
    CrIS ``fov_quality_bad`` (NFQF bits in any channel) and ATMS ``channel_auxiliary``,
    whose flag bits are set exactly when they are in one of its channels."""
    if flags is None or sensor not in ("atms", "cris"):
        return {}
    if sensor == "cris":
        bad = (flags & CRIS_FOV_QUALITY_BAD) != 0
        return {"fov_quality_bad": pa.array(bad.astype(np.uint8))}
    return {"channel_auxiliary": pa.array(flags.astype(np.uint16))}


def analysis_tables(
    *,
    prepbufr: pd.DataFrame | None = None,
    gpsro: pd.DataFrame | None = None,
    satwnd: pd.DataFrame | None = None,
    sat: Mapping[str, pd.DataFrame] | None = None,
    ir_channels: str | Mapping[str, Sequence[int]] | None,
    workers: int | None = None,
) -> dict:
    """``DAModel.run_analysis`` table arguments from wide Earth2Studio NNJA frames.

    Each stream is its own frame: ``prepbufr`` and ``gpsro`` from ``NNJAObsConv``,
    ``satwnd`` from ``NNJAObsSatwnd``, and ``sat`` a mapping from sensor to that
    sensor's ``NNJAObsSat`` frame. A stream or sensor not passed gives no tables, which
    the loaders read as no observations. Long frames go to ``analysis_tables_long``.

    Every footprint, report, occultation and wind comes from one file, so each stream
    is cut by ``cycle_time`` and the pieces adapted in parallel on ``workers`` threads (default
    the smaller of 32 and the CPUs this process may run on); the result equals adapting each
    whole stream.
    The function returns once the streams are cut. Each ``{cycle: table}`` is a
    ``CycleTables`` whose lookups wait for that cycle's adaptation.
    """
    if isinstance(sat, pd.DataFrame):
        raise ValueError(
            "sat is {sensor: frame}; one long frame goes to analysis_tables_long"
        )

    sat = sat or {}
    # (frame's Arrow table, its stream name)
    streams = []
    for name, frame in (("prepbufr", prepbufr), ("gps", gpsro)):
        if frame is not None:
            _check_layout(frame, True, name)
            streams.append((_arrow(frame, frame.columns, name), name))
    if satwnd is not None:
        _check_layout(satwnd, True, "satwnd")
        streams.append((_arrow(satwnd, WIDE_SATWND_COLUMNS, "satwnd"), "satwnd"))
    # Each sensor frame carries its channel axis in attrs["channels"]
    axes = {}
    unknown = sorted(set(sat) - set(SENSOR_CONFIGS))
    if unknown:
        raise ValueError(f"sat has sensors without an NNJA configuration: {unknown}")
    for sensor, frame in sat.items():
        _check_layout(frame, True, f"sat[{sensor!r}]")
        axes[sensor] = np.array(frame.attrs["channels"][sensor])
        variable = SAT_VARIABLE.get(sensor, sensor)
        names = [*FOOTPRINT_FLAGS, *FOOTPRINT_KEYS, "quality", WIDE_VALUE + variable]
        table = _arrow(frame, names, f"sat[{sensor!r}]", OPTIONAL_FOOTPRINT_COLUMNS)
        streams.append((table, variable))
    adapters = {
        "prepbufr": ("prepbufr_tables", None, _prepbufr_wide),
        "gps": ("gpsro_tables", None, _gpsro_wide),
        "satwnd": ("satwnd_tables", None, _satwnd_wide),
        **{
            SAT_VARIABLE.get(sensor, sensor): (
                "satellite_tables",
                sensor,
                functools.partial(
                    _sat_wide,
                    sensor=sensor,
                    axis=axes[sensor],
                    ir_channels=ir_channels,
                ),
            )
            for sensor in sat
        },
    }
    jobs = {}
    if prepbufr is not None:
        jobs["prepbufr_tables"] = {}
    if gpsro is not None:
        jobs["gpsro_tables"] = {}
    if satwnd is not None:
        jobs["satwnd_tables"] = {}
    if sat:
        jobs["satellite_tables"] = {sensor: {} for sensor in sat}
    return _adapt(streams, adapters, jobs, workers)


def _adapt(streams: list, adapters: dict, jobs: dict, workers: int | None) -> dict:
    """Cut each ``(table, stream)`` of ``streams`` into one piece per stream and cycle,
    adapt every piece on a thread pool and return ``jobs``, the ``run_analysis`` table
    arguments to fill, as ``CycleTables``.

    ``adapters`` maps a stream name to ``(argument, sensor or None, adapt)``; pieces of
    a stream it lacks are dropped.
    """
    pool = ThreadPoolExecutor(
        max_workers=workers or min(32, len(os.sched_getaffinity(0)))
    )
    try:
        pieces = [
            (name, cycle, piece)
            for table, stream in streams
            for (name, cycle), piece in _pieces(table, stream).items()
            if name in adapters
        ]
        for name, cycle, piece in pieces:
            argument, sensor, adapt = adapters[name]
            target = jobs[argument] if sensor is None else jobs[argument][sensor]
            target[pd.Timestamp(cycle)] = pool.submit(adapt, piece)
    finally:
        # Submitted jobs still run; the pool's threads exit once they finish.
        pool.shutdown(wait=False)

    return {
        argument: (
            {sensor: CycleTables(futures) for sensor, futures in value.items()}
            if argument == "satellite_tables"
            else CycleTables(value)
        )
        for argument, value in jobs.items()
    }


class CycleTables(Mapping):
    """``{cycle: table}`` whose tables are still being adapted on a thread pool.

    A lookup waits only for its own cycle's job, so a loader reads the first cycles
    while later ones are built. Iterating, ``len`` and pickling wait for every job. A
    cycle whose adaptation leaves no rows is absent. A failed job raises
    ``RuntimeError`` from the access that waits on it, never a ``KeyError`` that ``get``
    would read as an absent cycle.
    """

    def __init__(self, jobs: dict[pd.Timestamp, Future]):
        self._jobs = jobs

    def __getitem__(self, cycle: pd.Timestamp) -> pa.Table:
        table = _adapted(self._jobs[cycle])
        if not table.num_rows:
            raise KeyError(cycle)
        return table

    def __iter__(self) -> Iterator[pd.Timestamp]:
        for cycle, job in self._jobs.items():
            if _adapted(job).num_rows:
                yield cycle

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __reduce__(self):
        return dict, (dict(self.items()),)


def _adapted(job: Future) -> pa.Table:
    error = job.exception()
    if error is not None:
        raise RuntimeError("adapting a cycle's rows failed") from error
    return job.result()


def _check_layout(frame: pd.DataFrame, wide: bool, stream: str) -> None:
    """A long frame has ``observation``; a wide one ``observation_<variable>`` instead."""
    columns = [str(name) for name in frame.columns]
    is_wide = any(name.startswith(WIDE_VALUE) for name in columns)
    is_long = "observation" in columns
    if (is_wide, is_long) != (wide, not wide):
        found, target = (
            ("long", "analysis_tables_long") if wide else ("wide", "analysis_tables")
        )
        raise ValueError(f"{stream}: a {found} frame; pass it to {target}")


def _group_key(values: np.ndarray) -> np.ndarray:
    """Integers equal exactly where pandas groups values together: floats by bit
    pattern with every NaN one value and -0.0 as 0.0."""
    if values.dtype.kind != "f":
        return values
    bits = (values.astype(np.float64) + 0.0).view(np.int64)
    return np.where(np.isnan(values), np.int64(-1), bits)


def _group_ids(keys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rows by equal ``keys`` tuples, as pandas ``groupby(sort=False)`` does.

    Returns each row's group number (by first appearance, ``ngroup``), the first row
    of each group in that order, and each row's position within its group
    (``cumcount``). One stable sort; no Python per group. Several keys are sorted
    as one 64-bit hash of them, unless two different key tuples share a hash.
    """
    if len(keys) > 1:
        hashed = np.zeros(keys[0].size, dtype=np.uint64)
        for key in keys:
            hashed = hashed * np.uint64(0x9E3779B97F4A7C15) + key.astype(np.uint64)
            hashed ^= hashed >> np.uint64(29)
        ids, first, position = _group_ids([hashed])
        if all((key == key[first][ids]).all() for key in keys):
            return ids, first, position
    order = np.lexsort(keys[::-1])
    change = np.zeros(order.size, dtype=bool)
    change[0] = True
    for key in keys:
        ordered = key[order]
        change[1:] |= ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(change)
    run = np.cumsum(change) - 1
    first_rows = order[starts]
    rank = np.empty(starts.size, dtype=np.int64)
    rank[np.argsort(first_rows, kind="stable")] = np.arange(starts.size)
    ids = np.empty(order.size, dtype=np.int64)
    ids[order] = rank[run]
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size) - starts[run]
    return ids, np.sort(first_rows), position


def _floats(column: pa.ChunkedArray) -> np.ndarray:
    return column.cast(pa.float64()).to_numpy()


def _wide_floats(table: pa.Table, name: str) -> np.ndarray:
    """A wide value or quality column as float64, all NaN where the frame lacks it."""
    if name not in table.column_names:
        return np.full(table.num_rows, np.nan)
    return _floats(table[name])


def _wide_present(table: pa.Table, variable: str) -> np.ndarray:
    """Rows of a wide conventional frame holding ``variable``: a value or a quality mark."""
    present = np.zeros(table.num_rows, dtype=bool)
    for name in (WIDE_VALUE + variable, WIDE_QUALITY + variable):
        if name in table.column_names:
            present |= pc.is_valid(table[name]).to_numpy(zero_copy_only=False)
    return present


def _pieces(
    table: pa.Table,
    stream: str | Callable[[pa.Table], tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[str, int], pa.Table]:
    """``table`` cut into one table per (stream, archive cycle), rows in their original
    order. ``stream`` is the table's stream name, or, for a table that mixes streams,
    ``stream(part)`` gives each row's code into the stream names.

    Earth2Studio frames hold one source file per Arrow chunk, so a chunk of a
    single-stream table with one ``cycle_time`` is taken whole; other chunks are cut
    where the stream or the cycle changes.
    """
    merged: dict[tuple[str, int], list[pa.Table]] = {}
    for batch in table.to_batches():
        part = pa.Table.from_batches([batch])
        if not part.num_rows:
            continue
        if part["cycle_time"].null_count:
            raise ValueError("rows without cycle_time: every row names its source file")
        cycle_range = pc.min_max(part["cycle_time"])
        if isinstance(stream, str) and cycle_range["min"] == cycle_range["max"]:
            cycle = int(_native(part["cycle_time"].slice(0, 1))[0])
            runs = [((stream, cycle), part)]
        else:
            runs = _runs(part, stream)
        for key, piece in runs:
            merged.setdefault(key, []).append(piece)
    return {key: pa.concat_tables(pieces) for key, pieces in merged.items()}


def _runs(
    part: pa.Table, stream: str | Callable[[pa.Table], tuple[np.ndarray, np.ndarray]]
) -> list[tuple[tuple[str, int], pa.Table]]:
    if isinstance(stream, str):
        codes, names = np.zeros(part.num_rows, dtype=np.int64), np.array([stream])
    else:
        codes, names = stream(part)
    cycle = _native(part["cycle_time"])
    change = (codes[1:] != codes[:-1]) | (cycle[1:] != cycle[:-1])
    bounds = [0, *(np.flatnonzero(change) + 1), cycle.size]
    return [
        ((str(names[codes[lo]]), int(cycle[lo])), part.slice(lo, hi - lo))
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ]


def _arrow(frame: pd.DataFrame, names, stream: str, optional=()) -> pa.Table:
    """The frame's ``names`` as Arrow, zero-copy for Arrow-backed frames; every name
    not in ``optional`` must be a column of the frame."""
    missing = [
        name for name in names if name not in frame.columns and name not in optional
    ]
    if missing:
        raise ValueError(f"{stream}: the frame lacks the required columns {missing}")
    present = [name for name in names if name in frame.columns]
    # A view without attrs: Arrow cannot serialize e2s's ndarray request_time and warns.
    view = pd.DataFrame(frame, copy=False)
    return pa.Table.from_pandas(view, columns=present, preserve_index=False)


def _codes(column: pa.ChunkedArray) -> tuple[np.ndarray, pa.Array]:
    """Per-row int32 codes of a string column, -1 for null, and the code -> value table."""
    if not pa.types.is_dictionary(column.type):
        column = pc.dictionary_encode(column)
    array = column.unify_dictionaries().combine_chunks()
    return pc.fill_null(array.indices.cast(pa.int32()), -1).to_numpy(), array.dictionary


def _native(column: pa.ChunkedArray) -> np.ndarray:
    """One numpy array in the column's own dtype: floats keep NaN for null, integers
    widen to int64 with -1 for null, strings become codes, timestamps int64 ns."""
    if (
        pa.types.is_dictionary(column.type)
        or pa.types.is_string(column.type)
        or pa.types.is_large_string(column.type)
    ):
        return _codes(column)[0]
    if pa.types.is_timestamp(column.type):
        # UTC instants either way; dropping the timezone keeps the stored integers.
        return column.cast(pa.timestamp("ns")).cast(pa.int64()).to_numpy()
    if pa.types.is_integer(column.type):
        return pc.fill_null(column.cast(pa.int64()), -1).to_numpy()
    return column.to_numpy()


# ---------------------------------------------------------------------------------------
# Long layout: one row per observation, its variable in ``variable`` and its value in
# ``observation``. These functions regroup the rows into report levels, occultations,
# winds and footprints and hand them to the archive builders above, so both layouts
# give the same tables.
# ---------------------------------------------------------------------------------------

# NNJAObsSatwnd columns the SATWND adapter reads.
SATWND_COLUMNS = (*SATWND_WIND_COLUMNS, "variable", "observation")
# NNJAObsSat columns the satellite adapter reads, and those it reads when the frame has them.
SAT_CHANNEL_COLUMNS = ("sensor_index", "quality")
SAT_COLUMNS = (
    *FOOTPRINT_FLAGS,
    *FOOTPRINT_KEYS,
    *SAT_CHANNEL_COLUMNS,
    "variable",
    "observation",
)
SAT_OPTIONAL = (*OPTIONAL_FOOTPRINT_COLUMNS, "quality")


def analysis_tables_long(
    *,
    conv: pd.DataFrame | None = None,
    satwnd: pd.DataFrame | None = None,
    sat: pd.DataFrame | None = None,
    sensors: Sequence[str] = (),
    ir_channels: str | Mapping[str, Sequence[int]] | None,
    workers: int | None = None,
) -> dict:
    """``DAModel.run_analysis`` table arguments from long frames: ``conv`` an
    ``NNJAObsConv(original_event=True)`` frame of any PrepBUFR and GPS-RO variables,
    ``satwnd`` an ``NNJAObsSatwnd`` frame and ``sat`` an ``NNJAObsSat`` frame of any of
    ``sensors``."""
    if isinstance(sat, Mapping):
        raise ValueError(
            "sat is one long frame; {sensor: frame} goes to analysis_tables"
        )
    streams = []
    if conv is not None:
        _check_layout(conv, False, "conv")
        streams.append((_arrow(conv, conv.columns, "conv"), _conv_stream))
    if satwnd is not None:
        _check_layout(satwnd, False, "satwnd")
        streams.append((_arrow(satwnd, SATWND_COLUMNS, "satwnd"), "satwnd"))
    if sat is not None:
        _check_layout(sat, False, "sat")
        table = _arrow(sat, SAT_COLUMNS, "sat", SAT_OPTIONAL)
        streams.append((table, _variable_stream))
    adapters = {
        "prepbufr": ("prepbufr_tables", None, _prepbufr_long),
        "gps": ("gpsro_tables", None, _gpsro_long),
        "satwnd": ("satwnd_tables", None, _satwnd_long),
        **{
            SAT_VARIABLE.get(sensor, sensor): (
                "satellite_tables",
                sensor,
                functools.partial(_sat_long, sensor=sensor, ir_channels=ir_channels),
            )
            for sensor in sensors
        },
    }
    jobs = {}
    if conv is not None:
        jobs["prepbufr_tables"] = {}
        jobs["gpsro_tables"] = {}
    if satwnd is not None:
        jobs["satwnd_tables"] = {}
    if sat is not None:
        jobs["satellite_tables"] = {sensor: {} for sensor in sensors}
    return _adapt(streams, adapters, jobs, workers)


def _gpsro_long(table: pa.Table) -> pa.Table:
    """GPS-RO archive rows, one per bending-angle level, from ``gps`` rows; the same
    occultations' ``gps_refractivity`` rows supply the derived vertical coordinates."""
    variable, variables = _codes(table["variable"])
    value = _floats(table["observation"])
    return _gpsro_archive(
        table,
        is_gps=variable == variables.index(GPS_VARIABLE).as_py(),
        is_profile=variable == variables.index(REFRACTIVITY_VARIABLE).as_py(),
        bending=value,
        refractivity=value,
        qfro=_floats(table["quality"]),
    )


def _satwnd_long(table: pa.Table) -> pa.Table:
    """SATWND archive rows, one per wind, rejoining each wind's ``u`` and ``v`` rows on
    the wind's own columns."""
    variable, names = _codes(table["variable"])
    is_u = variable == names.index("u").as_py()
    u_rows, v_rows = np.flatnonzero(is_u), np.flatnonzero(~is_u)
    keys, subset_names = _satwnd_winds(table)
    # NNJAObsSatwnd builds a file's v rows from the same winds, in the same order, as
    # its u rows.
    if not _same_rows(keys, u_rows, v_rows):
        raise ValueError("NNJAObsSatwnd v rows are not in the u rows' wind order")
    observation = _floats(table["observation"])
    winds = {name: values[u_rows] for name, values in keys.items()}
    return _satwnd_archive(
        winds, subset_names, observation[u_rows], observation[v_rows]
    )


def _prepbufr_long(table: pa.Table) -> pa.Table:
    variable, variables = _codes(table["variable"])
    wanted = [variables.index(name).as_py() for name in PREPBUFR_LEVEL_VARIABLES]
    rows = np.flatnonzero(np.isin(variable, [code for code in wanted if code >= 0]))
    if not rows.size:
        return pa.table({})
    variable = variable[rows]
    levels = _prepbufr_levels(table, rows)
    keys = [_group_key(levels[name]) for name in PREPBUFR_LEVEL_COLUMNS]
    # A report repeated in the file gives each variable once per copy.
    occurrence = _group_ids([*keys, variable])[2]
    level, first, _ = _group_ids([*keys, occurrence])

    observation = _floats(table["observation"])[rows]
    quality = _floats(table["quality"])[rows]
    values, qualities, present = {}, {}, {}
    for name in PREPBUFR_VARIABLES:
        selected = variable == variables.index(name).as_py()
        values[name] = np.full(first.size, np.nan)
        qualities[name] = np.full(first.size, np.nan)
        present[name] = np.zeros(first.size, dtype=bool)
        values[name][level[selected]] = observation[selected]
        qualities[name][level[selected]] = quality[selected]
        present[name][level[selected]] = True
    header = {name: column[first] for name, column in levels.items()}
    stations = _codes(table["station"])[1]
    return _prepbufr_archive(header, stations, values, qualities, present)


def _sat_long(table: pa.Table, sensor: str, ir_channels) -> pa.Table:
    """Wide satellite archive rows, one per footprint, from ``sensor``'s rows in source
    order: a footprint's channel rows are consecutive and ascending in
    ``sensor_index``. Channels ``NNJAWideLoader(ir_channels=ir_channels)`` reads, null
    where the frame has no row."""
    # Arrow and numpy throughout: earth2studio frames are Arrow-backed, and pandas
    # operations on them are slower for identical output.
    channels, channel_slot = _sat_channels(sensor, ir_channels)
    footprint_columns = [
        name
        for name in (*FOOTPRINT_KEYS, *FOOTPRINT_FLAGS)
        if name in table.column_names
    ]
    has_quality = "quality" in table.column_names
    width = _footprint_width(table)
    if width:
        # Every footprint is the same `width` consecutive rows: reshape, no pivot.
        firsts = table.select(footprint_columns).take(
            pa.array(np.arange(0, table.num_rows, width))
        )
        slot = _slots(_native(table["sensor_index"].slice(0, width)), channel_slot)
        if not (slot >= 0).any():
            return pa.table({})
        footprints = {name: _native(firsts[name]) for name in footprint_columns}
        observation = _channel_rows(
            _native(table["observation"]).astype(np.float32, copy=False),
            width,
            slot,
            channels.size,
        )
        flags = None
        if has_quality:
            flags = _or_flags(_unsigned(table["quality"]), width, slot)
        platforms = _codes(firsts["satellite"])[1]
        return _sat_archive(sensor, footprints, platforms, channels, observation, flags)

    slot = _slots(_native(table["sensor_index"]), channel_slot)
    kept = np.flatnonzero(slot >= 0)
    if not kept.size:
        return pa.table({})
    slot = slot[kept]

    def at_kept(name: str) -> np.ndarray:
        values = _native(table[name])
        return values if values.size == kept.size else values[kept]

    # A footprint starts where the channel sequence restarts or a footprint column changes.
    starts = np.ones(kept.size, dtype=bool)
    starts[1:] = slot[1:] <= slot[:-1]
    columns = {name: at_kept(name) for name in footprint_columns}
    for name in FOOTPRINT_KEYS:
        if name in columns:
            values = columns[name]
            same = values[1:] == values[:-1]
            if values.dtype.kind == "f":
                same |= np.isnan(values[1:]) & np.isnan(values[:-1])
            starts[1:] |= ~same
    first = np.flatnonzero(starts)
    footprint = np.cumsum(starts) - 1

    observation = np.full((channels.size, first.size), np.nan, dtype=np.float32)
    observation[slot, footprint] = at_kept("observation")
    flags = None
    if has_quality:
        quality = np.zeros((channels.size, first.size), dtype=np.int64)
        quality[slot, footprint] = np.maximum(at_kept("quality"), 0)
        flags = np.bitwise_or.reduce(quality, axis=0)
    footprints = {name: values[first] for name, values in columns.items()}
    platforms = _codes(table["satellite"])[1]
    return _sat_archive(sensor, footprints, platforms, channels, observation, flags)


def _footprint_width(table: pa.Table) -> int:
    """Rows per footprint when every footprint is the same channel sequence in
    consecutive rows and the footprint columns hold within it, else 0."""
    sensor_index = _raw(table["sensor_index"])
    restarts = np.flatnonzero(sensor_index[1:] <= sensor_index[:-1])
    width = int(restarts[0]) + 1 if restarts.size else sensor_index.size
    if not width or sensor_index.size % width:
        return 0
    blocks = sensor_index.reshape(-1, width)
    if not (blocks == blocks[0]).all():
        return 0
    for name in FOOTPRINT_KEYS:
        if name not in table.column_names:
            continue
        values = _raw(table[name]).reshape(-1, width)
        same = values == values[:, :1]
        if values.dtype.kind == "f":
            same |= np.isnan(values) & np.isnan(values[:, :1])
        if not same.all():
            return 0
    return width


def _same_rows(keys: dict, a: np.ndarray, b: np.ndarray) -> bool:
    """True when rows ``a[i]`` and ``b[i]`` agree on every key, NaN equal to NaN."""
    if a.size != b.size:
        return False
    for values in keys.values():
        x, y = values[a], values[b]
        same = x == y
        if x.dtype.kind == "f":
            same |= np.isnan(x) & np.isnan(y)
        if not same.all():
            return False
    return True


def _raw(column: pa.ChunkedArray) -> np.ndarray:
    """Values to compare row with row, without widening: strings as codes,
    timestamps as int64, null floats NaN."""
    if pa.types.is_dictionary(column.type):
        return (
            column.unify_dictionaries()
            .combine_chunks()
            .indices.to_numpy(zero_copy_only=False)
        )
    if pa.types.is_timestamp(column.type):
        return column.to_numpy().view(np.int64)
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return _codes(column)[0]
    return column.to_numpy()


def _variable_stream(part: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    codes, names = _codes(part["variable"])
    return codes, np.array([*names.to_pylist(), None], dtype=object)


def _conv_stream(part: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    variable, names = _codes(part["variable"])
    gps = np.isin(names.to_pylist(), [GPS_VARIABLE, REFRACTIVITY_VARIABLE])
    is_gps = np.append(gps, False)[variable]
    return is_gps.astype(np.int64), np.array(["prepbufr", "gps"])
