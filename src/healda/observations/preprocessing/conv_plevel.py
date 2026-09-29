# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pressure-level channel vocabulary for conventional observations.

The processed parquet still stores the original 8-channel ``conv`` sensor. This
module defines the optional 92-channel ``conv-plevel`` view used at training time:

- gps_angle/gps_t/gps_q/q/t/u/v each get one channel per ERA5 pressure level
- ps stays a single surface channel

An expanded id is ``group * CONV_PLEVEL_N_LEVELS + level index``, where the group
is a channel's position among the non-ps channels and the level index comes from
``nearest_pressure_level_index`` (0 is 1000 hPa, 12 is 50 hPa). Because ps sits
mid-order in ``CONV_CHANNELS``, the group is not the base id -- every channel
after ps shifts down by one::

    base id  name       group  expanded ids
    0        gps_angle  0      0..12
    1        gps_t      1      13..25
    2        gps_q      2      26..38
    3        ps         -1     91
    4        q          3      39..51
    5        t          4      52..64
    6        u          5      65..77
    7        v          6      78..90

So q at 1000 hPa is 39 and q at 50 hPa is 51; reusing the base id 4 in place of
the group 3 would land in t. ``base_local_channel_to_expanded_group`` returns
that group column, ``[0, 1, 2, -1, 3, 4, 5, 6]``, for vectorised lookup, and ps
carries -1 there because it has no group.

The base conv channel order comes from ``sensors.py``. This module only adds the
pressure-level expansion rules and builds a deterministic channel map from those
rules, so the training view can be used without rewriting the parquet dataset.
"""

import functools

import numba
import numpy as np
import pandas as pd
import pyarrow as pa

from healda.observations.sensors import (
    CONV_CHANNELS,
    CONV_GPS_CHANNELS,
    CONV_GPS_LEVEL2_CHANNELS,
    CONV_UV_CHANNELS,
    NORMALIZATION_DIR,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
    SENSOR_OFFSET,
)
from healda.observations.schema import (
    GLOBAL_CHANNEL_ID,
    SENSOR_ID,
    get_channel_table_schema,
)

PACKAGED_LEVEL_STATS = NORMALIZATION_DIR / "conv_normalizations_by_level.csv"

PRESSURE_LEVELS_HPA = np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    dtype=np.int16,
)
CONV_PLEVEL_SURFACE_LOCAL_CHANNEL = next(
    i for i, channel in enumerate(CONV_CHANNELS) if channel.name == "ps"
)
# Position in this array is the group index.
CONV_PLEVEL_BASE_LOCAL_CHANNELS = np.array(
    [
        local_channel
        for local_channel in range(len(CONV_CHANNELS))
        if local_channel != CONV_PLEVEL_SURFACE_LOCAL_CHANNEL
    ],
    dtype=np.int16,
)
CONV_PLEVEL_N_LEVELS = PRESSURE_LEVELS_HPA.size
CONV_PLEVEL_N_VERTICAL_CHANNELS = CONV_PLEVEL_BASE_LOCAL_CHANNELS.size
# ps is not level-expanded, so it takes the slot after every vertical one.
CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL = (
    CONV_PLEVEL_N_VERTICAL_CHANNELS * CONV_PLEVEL_N_LEVELS
)


def base_local_channel_to_expanded_group() -> np.ndarray:
    """Expanded group index per base conv local channel, -1 for the surface channel."""
    groups = np.full(len(CONV_CHANNELS), -1, dtype=np.int16)
    for group, local_channel in enumerate(CONV_PLEVEL_BASE_LOCAL_CHANNELS):
        groups[local_channel] = group
    return groups


def _expanded_range(base_local_channels: list[int]) -> tuple[int, int]:
    # Half-open expanded span covering a family of base channels. Their groups must
    # be adjacent so the family stays a single slice.
    base_to_group = base_local_channel_to_expanded_group()
    groups = sorted(
        int(base_to_group[int(local_channel)]) for local_channel in base_local_channels
    )
    expected = list(range(groups[0], groups[-1] + 1))
    if groups != expected:
        raise ValueError(
            f"conv-plevel channels are not contiguous: {base_local_channels}"
        )
    return groups[0] * CONV_PLEVEL_N_LEVELS, (groups[-1] + 1) * CONV_PLEVEL_N_LEVELS


CONV_PLEVEL_GPS_EXPANDED_RANGE = _expanded_range(CONV_GPS_CHANNELS)
CONV_PLEVEL_GPS_LEVEL2_EXPANDED_RANGE = _expanded_range(CONV_GPS_LEVEL2_CHANNELS)
CONV_PLEVEL_UV_EXPANDED_RANGE = _expanded_range(CONV_UV_CHANNELS)

_PRESSURE_LEVELS_ASC = np.sort(PRESSURE_LEVELS_HPA)
_PRESSURE_LEVEL_EDGES = (
    _PRESSURE_LEVELS_ASC[:-1].astype(np.float32)
    + _PRESSURE_LEVELS_ASC[1:].astype(np.float32)
) / 2.0


def nearest_pressure_level_index(pressure: np.ndarray) -> np.ndarray:
    """Return indices into ``PRESSURE_LEVELS_HPA`` for nearest-level binning.

    Values outside the 50-1000 hPa span are clipped to the nearest endpoint by
    the fixed midpoint thresholds. The loop over 12 thresholds is faster than
    ``np.searchsorted`` for this fixed tiny vocabulary and avoids an Arrow join
    in the loader hot path.
    """
    pressure = np.asarray(pressure, dtype=np.float32)
    idx = np.zeros(pressure.size, dtype=np.int8)
    for edge in _PRESSURE_LEVEL_EDGES:
        idx += pressure > edge
    return (PRESSURE_LEVELS_HPA.size - 1 - idx).astype(np.int8, copy=False)


@numba.njit(cache=True, nogil=True)
def nearest_pressure_level_index_scalar(pressure_f32):
    idx = 0
    for edge in _PRESSURE_LEVEL_EDGES:
        if pressure_f32 > edge:
            idx += 1
    return PRESSURE_LEVELS_HPA.size - 1 - idx


def expanded_local_channel(
    base_local_channel: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Map base conv local ids to conv-plevel local ids.

    The pressure-expanded order is inherited from ``CONV_CHANNELS`` with the
    surface-pressure channel removed. ``ps`` always maps to the final expanded
    channel. This is the reference mapping used by tests and by any future ETL
    precompute path; the loader uses an equivalent cached LUT.
    """
    base_to_group = base_local_channel_to_expanded_group()

    expanded = np.empty_like(base_local_channel, dtype=np.int16)
    is_surface = base_local_channel == CONV_PLEVEL_SURFACE_LOCAL_CHANNEL
    expanded[is_surface] = CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL

    is_vertical = ~is_surface
    groups = base_to_group[base_local_channel[is_vertical]]
    if (groups < 0).any():
        bad = np.unique(base_local_channel[is_vertical][groups < 0]).tolist()
        raise ValueError(f"cannot map conv local channels to conv-plevel: {bad}")
    expanded[is_vertical] = (
        groups * CONV_PLEVEL_N_LEVELS
        + nearest_pressure_level_index(pressure[is_vertical])
    ).astype(np.int16, copy=False)
    return expanded


@functools.cache
def conv_plevel_channel_map() -> pa.Table:
    """Build the deterministic conv-plevel identity map.

    Each row maps one expanded ``conv-plevel`` channel back to its source base
    conv channel and pressure level. For example, base ``q`` at 1000 hPa becomes
    ``q_1000`` with a new conv-plevel Global_Channel_ID; ``ps`` has
    ``Level_hPa=-1`` because it is not pressure-expanded.
    """
    conv_offset = SENSOR_OFFSET["conv"]
    conv_plevel_offset = SENSOR_OFFSET["conv-plevel"]

    rows = []
    local_channel_id = 0
    for base_local_channel in CONV_PLEVEL_BASE_LOCAL_CHANNELS:
        base_channel = CONV_CHANNELS[int(base_local_channel)].name
        base_global_id = conv_offset + int(base_local_channel)
        for level_hpa in PRESSURE_LEVELS_HPA:
            rows.append(
                {
                    "Global_Channel_ID": conv_plevel_offset + local_channel_id,
                    "local_channel_id": local_channel_id,
                    "name": f"{base_channel}_{int(level_hpa)}",
                    "base_channel": base_channel,
                    "base_Global_Channel_ID": base_global_id,
                    "Level_hPa": int(level_hpa),
                }
            )
            local_channel_id += 1

    surface_channel = CONV_CHANNELS[CONV_PLEVEL_SURFACE_LOCAL_CHANNEL].name
    rows.append(
        {
            "Global_Channel_ID": conv_plevel_offset + local_channel_id,
            "local_channel_id": local_channel_id,
            "name": surface_channel,
            "base_channel": surface_channel,
            "base_Global_Channel_ID": conv_offset + CONV_PLEVEL_SURFACE_LOCAL_CHANNEL,
            "Level_hPa": -1,
        }
    )

    return pa.table(
        {
            "Global_Channel_ID": pa.array(
                [row["Global_Channel_ID"] for row in rows], type=GLOBAL_CHANNEL_ID.type
            ),
            "local_channel_id": pa.array(
                [row["local_channel_id"] for row in rows], type=pa.uint16()
            ),
            "name": pa.array([row["name"] for row in rows], type=pa.string()),
            "base_channel": pa.array(
                [row["base_channel"] for row in rows], type=pa.string()
            ),
            "base_Global_Channel_ID": pa.array(
                [row["base_Global_Channel_ID"] for row in rows],
                type=GLOBAL_CHANNEL_ID.type,
            ),
            "Level_hPa": pa.array([row["Level_hPa"] for row in rows], type=pa.int16()),
        }
    )


def _base_norm_lookup(
    base_channel_table: pa.Table | None,
) -> dict[int, tuple[float, float]]:
    """Return base conv normalizations keyed by original Global_Channel_ID.

    ``None`` means the packaged conv stats, not "no normalization".
    """
    if base_channel_table is None:
        conv = SENSOR_CONFIGS["conv"]
        offset = SENSOR_OFFSET["conv"]
        return {
            offset + i: (float(mean), float(std))
            for i, (mean, std) in enumerate(zip(conv.means, conv.stds))
        }
    gids = np.asarray(base_channel_table["Global_Channel_ID"], dtype=np.int64)
    means = np.asarray(base_channel_table["mean"], dtype=np.float32)
    stddevs = np.asarray(base_channel_table["stddev"], dtype=np.float32)
    return {
        int(gid): (float(mean), float(stddev))
        for gid, mean, stddev in zip(gids, means, stddevs)
    }


def _level_norm_lookup(
    level_stats: pa.Table | None,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Return pressure-level normalizations keyed by (base Global_Channel_ID, hPa).

    ``None`` means the packaged by-level stats, not "no normalization".
    """
    if level_stats is None:
        if not PACKAGED_LEVEL_STATS.is_file():
            raise FileNotFoundError(
                f"missing packaged level stats: {PACKAGED_LEVEL_STATS}"
            )
        level_stats = pa.Table.from_pandas(pd.read_csv(PACKAGED_LEVEL_STATS))
    mean_name = "level_mean" if "level_mean" in level_stats.column_names else "obs_mean"
    std_name = (
        "level_stddev" if "level_stddev" in level_stats.column_names else "obs_std"
    )
    gids = np.asarray(level_stats["Global_Channel_ID"], dtype=np.int64)
    levels = np.asarray(level_stats["Level_hPa"], dtype=np.int16)
    means = np.asarray(level_stats[mean_name], dtype=np.float32)
    stddevs = np.asarray(level_stats[std_name], dtype=np.float32)
    return {
        (int(gid), int(level)): (float(mean), float(stddev))
        for gid, level, mean, stddev in zip(gids, levels, means, stddevs)
    }


def build_conv_plevel_channel_table(
    level_stats: pa.Table | None = None,
    base_channel_table: pa.Table | None = None,
    include_local_channel_id: bool = False,
) -> pa.Table:
    """Build the 92-row ``conv-plevel`` channel table.

    ``level_stats`` is keyed by original base conv Global_Channel_ID and Level_hPa.
    For pressure-expanded rows, matching level stats override the base conv
    normalization; missing level rows fall back to base channel stats. The ps row
    has Level_hPa=-1 and always uses the base ps normalization.
    """
    conv_offset = SENSOR_OFFSET["conv"]
    conv_plevel_sensor_id = SENSOR_NAME_TO_ID["conv-plevel"]
    level_norms = _level_norm_lookup(level_stats)
    base_norms = _base_norm_lookup(base_channel_table)

    rows = []
    channel_map = conv_plevel_channel_map().to_pydict()
    for i in range(len(channel_map["Global_Channel_ID"])):
        base_global_id = int(channel_map["base_Global_Channel_ID"][i])
        base_local_channel = base_global_id - conv_offset
        channel = CONV_CHANNELS[base_local_channel]
        level_hpa = int(channel_map["Level_hPa"][i])
        if base_global_id not in base_norms:
            raise KeyError(
                f"no conv normalization for base Global_Channel_ID {base_global_id}"
            )
        base_mean, base_stddev = base_norms[base_global_id]
        if level_hpa < 0:
            mean, stddev = base_mean, base_stddev
        else:
            mean, stddev = level_norms.get(
                (base_global_id, level_hpa), (base_mean, base_stddev)
            )
        rows.append(
            {
                "Global_Channel_ID": int(channel_map["Global_Channel_ID"][i]),
                "min_valid": channel.min_valid,
                "max_valid": channel.max_valid,
                "sensor_id": conv_plevel_sensor_id,
                "is_conv": True,
                "name": channel_map["name"][i],
                "mean": mean,
                "stddev": stddev,
                "local_channel_id": int(channel_map["local_channel_id"][i]),
            }
        )

    names = [field.name for field in get_channel_table_schema()]
    if include_local_channel_id:
        names.append("local_channel_id")
    schema = get_channel_table_schema()
    arrays = [
        pa.array(
            [row["Global_Channel_ID"] for row in rows], type=GLOBAL_CHANNEL_ID.type
        ),
        pa.array([row["min_valid"] for row in rows], type=pa.float32()),
        pa.array([row["max_valid"] for row in rows], type=pa.float32()),
        pa.array([row["sensor_id"] for row in rows], type=SENSOR_ID.type),
        pa.array([row["is_conv"] for row in rows], type=pa.bool_()),
        pa.array([row["name"] for row in rows], type=pa.string()),
        pa.array([row["mean"] for row in rows], type=pa.float32()),
        pa.array([row["stddev"] for row in rows], type=pa.float32()),
    ]
    if include_local_channel_id:
        schema = schema.append(pa.field("local_channel_id", pa.uint16()))
        arrays.append(
            pa.array([row["local_channel_id"] for row in rows], type=pa.uint16())
        )
    return pa.table(arrays, schema=schema).select(names)
