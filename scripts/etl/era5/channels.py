# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout of an ERA5 hourly year file: its channels, their raw sources, and byte geometry.

A channel set is an ordered list of output names, given as a preset (``c73``), a comma
separated list (``sst,sic,z500``), or ``@path`` to a file of names. Surface names resolve
through ``SURFACE_SOURCE``, pressure names by splitting a prefix and a level, so ``z500``
is ``z`` at 500 hPa.

Output names here are the short names of the hourly year files; ``CHANNELS_75`` is in the
CF-style names of ``healda.config.channels``.
"""

import datetime
from dataclasses import dataclass

from healda.config.channels import CHANNELS_75, LEVELS, channel_name

NLAT, NLON = 721, 1440
CHAN_BYTES = NLAT * NLON * 4
HOURS_PER_BLOCK = 24

# Output name -> token in the raw NCAR filename and its netCDF short_name.
SURFACE_SOURCE = {
    "u10m": "10u",
    "v10m": "10v",
    "u100m": "100u",
    "v100m": "100v",
    "t2m": "2t",
    "sp": "sp",
    "msl": "msl",
    "tcwv": "tcwv",
    "tcw": "tcw",
    "sst": "sstk",
    "sic": "ci",
    "skt": "skt",
    "tcc": "tcc",
    "d2m": "2d",
    "lcc": "lcc",
    "mcc": "mcc",
    "hcc": "hcc",
    "stl1": "stl1",
    "stl2": "stl2",
    "swvl1": "swvl1",
    "swvl2": "swvl2",
    "sd": "sd",
}

PRESSURE_PREFIXES = frozenset("uvztqrw")

C73_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def _c73() -> list[str]:
    names = ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv"]
    for prefix in ("u", "v", "z", "t", "q"):
        names += [f"{prefix}{level}" for level in C73_LEVELS]
    return names


# c73 is the 73-channel training set; c75 appends the two
# fields the HPX64 set needs; the rest are companion packs over the same time axis.
PRESETS = {
    "c73": _c73(),
    "c75": _c73() + ["sst", "sic"],
    "sst_sic": ["sst", "sic"],
    "cloud_cover": ["tcc", "lcc", "mcc", "hcc"],
    "land": ["stl1", "stl2", "swvl1", "swvl2", "sd"],
    "w": [f"w{level}" for level in C73_LEVELS],
    "d2m_skt_sp": ["d2m", "skt", "sp"],
}

# Named by count: a store is written once against a frozen channel order, so a later
# set is a new constant beside this one, never an edit to it.
CHANNELS_99 = (
    tuple(CHANNELS_75)
    + tuple(channel_name("W", level) for level in LEVELS)
    + ("d2m", "skt")
    + tuple(PRESETS["cloud_cover"])
    + tuple(PRESETS["land"])
)


@dataclass(frozen=True)
class SourceGroup:
    """One raw variable and the output channels it fills, in ascending channel order."""

    raw_name: str
    product: str
    levels: tuple[int, ...]
    channel_indices: tuple[int, ...]


def parse_channel(name: str) -> tuple[str, str, int | None]:
    if name in SURFACE_SOURCE:
        return "sfc", SURFACE_SOURCE[name], None
    prefix = name.rstrip("0123456789")
    digits = name[len(prefix) :]
    raw_prefix = prefix.lower()
    if not digits or raw_prefix not in PRESSURE_PREFIXES:
        raise ValueError(
            f"unknown channel {name!r}; expected one of "
            f"{sorted(SURFACE_SOURCE)} or <{'|'.join(sorted(PRESSURE_PREFIXES))}><level>"
        )
    return "pl", raw_prefix, int(digits)


def resolve_channels(spec: str) -> list[str]:
    if spec in PRESETS:
        names = list(PRESETS[spec])
    elif spec.startswith("@"):
        with open(spec[1:]) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = [part.strip() for part in spec.split(",") if part.strip()]

    if not names:
        raise ValueError(f"channel set {spec!r} is empty")
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"channel set repeats {sorted(duplicates)}")
    for name in names:
        parse_channel(name)
    return names


def source_groups(channel_names: list[str]) -> list[SourceGroup]:
    """One group per raw variable, in order of first appearance so task ids are stable.

    A pressure chunk spans all 37 levels, so one group must consume every level the set
    wants from that variable; splitting them would inflate the same chunk repeatedly.
    """
    collected: dict[tuple[str, str], tuple[list[int], list[int]]] = {}

    for index, name in enumerate(channel_names):
        product, raw, level = parse_channel(name)
        levels, indices = collected.setdefault((product, raw), ([], []))
        if level is not None:
            levels.append(level)
        indices.append(index)

    return [
        SourceGroup(raw, product, tuple(levels), tuple(indices))
        for (product, raw), (levels, indices) in collected.items()
    ]


def write_plan(
    group: SourceGroup,
    day: int,
    data_offset: int,
    stride: int,
    steps_per_day: int = HOURS_PER_BLOCK,
):
    """Yield ``(hour, byte offset, level keys)`` for each write of one task.

    Tasks partition (variable, day) and runs are contiguous, so the ranges this yields
    across every task of a year tile the dataset exactly once.
    """
    runs = channel_runs(group)
    for hour in range(steps_per_day):
        base = data_offset + (day * steps_per_day + hour) * stride
        for channel, keys in runs:
            yield hour, base + channel * CHAN_BYTES, keys


def channel_runs(group: SourceGroup) -> list[tuple[int, list]]:
    """Split a group into runs of consecutive output channels, one write each.

    Level keys index the dict ``read_era5_source_group`` returns, which uses ``None``
    for a surface variable.
    """
    keys = group.levels or (None,)
    runs: list[tuple[int, list]] = []
    for channel, key in zip(group.channel_indices, keys, strict=True):
        if runs and channel == runs[-1][0] + len(runs[-1][1]):
            runs[-1][1].append(key)
        else:
            runs.append((channel, [key]))
    return runs


# ARCO-ERA5 keys its arrays by the long CDS names; SURFACE_SOURCE above gives the
# short name, this gives the long one.
ARCO_VARIABLE = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "2d": "2m_dewpoint_temperature",
    "2t": "2m_temperature",
    "ci": "sea_ice_cover",
    "hcc": "high_cloud_cover",
    "lcc": "low_cloud_cover",
    "mcc": "medium_cloud_cover",
    "msl": "mean_sea_level_pressure",
    "q": "specific_humidity",
    "sd": "snow_depth",
    "skt": "skin_temperature",
    "sp": "surface_pressure",
    "sstk": "sea_surface_temperature",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
    "t": "temperature",
    "tcc": "total_cloud_cover",
    "tcw": "total_column_water",
    "tcwv": "total_column_water_vapour",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "z": "geopotential",
}

ARCO_EPOCH = datetime.datetime(1900, 1, 1)


def hour_index(timestamp: datetime.datetime) -> int:
    return int((timestamp - ARCO_EPOCH).total_seconds() // 3600)
