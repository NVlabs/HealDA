# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import numpy as np
import pyarrow as pa
from dataclasses import dataclass, field
import pathlib

NORMALIZATION_DIR = pathlib.Path(__file__).parent / "normalizations" / "ufs"


@dataclass
class SensorConfig:
    """
    Sensor metadata that sets up data loading.
    Defines the sensor name, platforms, channels, and normalization stats.
    """

    name: str
    platforms: list[str]
    channels: int
    nc_file_template: str
    means: np.ndarray = field(init=False)
    stds: np.ndarray = field(init=False)
    min_valid: float = 0.0
    max_valid: float = 400.0
    sensor_type: str = "microwave"
    # None means this sensor has no per-sensor stats file; its normalizations come
    # from elsewhere (conv-plevel's are per pressure level, see conv_plevel.py).
    stats_csv: str | None = ""
    raw_to_local: np.ndarray = field(
        init=False
    )  # lookup-table: raw_id →  local channel

    def __post_init__(self):
        if self.stats_csv is None:
            self.means = np.zeros(self.channels, dtype=float)
            self.stds = np.ones(self.channels, dtype=float)
            self.raw_to_local = None
            return

        norm_file = NORMALIZATION_DIR / (
            self.stats_csv or f"{self.name}_normalizations.csv"
        )
        if not norm_file.exists():
            raise FileNotFoundError(
                f"{self.name}: missing normalizations {norm_file}. Set stats_csv=None "
                "if this sensor deliberately has none."
            )
        df = pd.read_csv(norm_file)
        # Col -1 is the avg across all platforms
        channel_col = "Raw_Channel_ID"
        df = df[df["Platform_ID"] == -1].sort_values(channel_col)

        self.means = df["obs_mean"].to_numpy()
        self.stds = df["obs_std"].to_numpy()

        raw_ids = df[channel_col].to_numpy()
        lookup_table = np.full(raw_ids.max() + 1, 0, dtype=int)
        for local_idx, raw in enumerate(raw_ids, start=1):
            lookup_table[raw] = local_idx
        self.raw_to_local = lookup_table


def _build_identity_lut(raw_ids: np.ndarray) -> np.ndarray:
    """Build a 1-indexed identity LUT from observed raw channel IDs.
    Used for initial setup of raw IR sounder data which have non-contiguous channel IDs."""
    raw_ids = np.asarray(raw_ids).ravel()
    unique = np.unique(raw_ids)
    lut = np.zeros(int(unique.max()) + 1, dtype=int)
    for local_idx, raw in enumerate(unique, start=1):
        lut[raw] = local_idx
    return lut


def get_global_channel_id(sensor, raw_channel_ids):
    """Map per-sensor raw channel IDs to unified global IDs (no overlap across sensors)."""
    cfg = SENSOR_CONFIGS[sensor]
    if cfg.raw_to_local is None:
        cfg.raw_to_local = _build_identity_lut(raw_channel_ids)
    raw_to_local = cfg.raw_to_local
    channel_offset = SENSOR_OFFSET[sensor]
    raw_channel_ids = np.asarray(raw_channel_ids)
    safe_ids = np.minimum(raw_channel_ids, len(raw_to_local) - 1)
    local_channels = raw_to_local[safe_ids] - 1
    return (local_channels + channel_offset).astype(np.uint16)


SENSOR_CONFIGS = {
    "atms": SensorConfig(
        name="atms",
        platforms=["npp", "n20", "n21"],
        channels=22,
        nc_file_template="diag_atms_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    "mhs": SensorConfig(
        name="mhs",
        platforms=["metop-a", "metop-b", "metop-c", "n18", "n19"],
        channels=5,
        nc_file_template="diag_mhs_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    "amsua": SensorConfig(
        name="amsua",
        platforms=["metop-a", "metop-b", "metop-c", "n15", "n16", "n17", "n18", "n19"],
        channels=15,
        nc_file_template="diag_amsua_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    "amsub": SensorConfig(
        name="amsub",
        platforms=["n15", "n16", "n17"],
        channels=5,
        nc_file_template="diag_amsub_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    # Raw IR sounders - unused
    "iasi": SensorConfig(
        name="iasi",
        platforms=["metop-a", "metop-b", "metop-c"],
        channels=175,
        nc_file_template="diag_iasi_{platform}_ges.{date}_control.nc4",
        min_valid=150.0,
        max_valid=350.0,
        sensor_type="infrared",
    ),
    "cris-fsr": SensorConfig(
        name="cris-fsr",
        platforms=["npp", "n20"],
        channels=100,
        nc_file_template="diag_cris_fsr_{platform}_ges.{date}_control.nc4",
        min_valid=150.0,
        max_valid=350.0,
        sensor_type="infrared",
    ),
    "conv": SensorConfig(
        name="conv",
        platforms=[],  # platform idea doesn't apply to conv
        channels=8,  # all conv sensors stacked (gps angle, gps temp, gps spfh, ps, q, t, u, v) (see CONV_CHANNELS)
        nc_file_template="conv_{platform}_ges.{date}_control.nc4",
        sensor_type="conv",
    ),
    # Compressed IR sounders — 32 PCA latent channels per footprint
    "iasi-pca": SensorConfig(
        name="iasi-pca",
        platforms=["metop-a", "metop-b", "metop-c"],
        channels=32,
        nc_file_template="",
        min_valid=float("-inf"),
        max_valid=float("inf"),
        sensor_type="infrared",
    ),
    "cris-fsr-pca": SensorConfig(
        name="cris-fsr-pca",
        platforms=["npp", "n20"],
        channels=32,
        nc_file_template="",
        min_valid=float("-inf"),
        max_valid=float("inf"),
        sensor_type="infrared",
    ),
    "airs": SensorConfig(
        name="airs",
        platforms=["aqua"],
        channels=117,
        nc_file_template="diag_airs_{platform}_ges.{date}_control.nc4",
        min_valid=150.0,
        max_valid=350.0,
        sensor_type="infrared",
    ),
    "airs-pca": SensorConfig(
        name="airs-pca",
        platforms=["aqua"],
        channels=32,
        nc_file_template="",
        min_valid=float("-inf"),
        max_valid=float("inf"),
        sensor_type="infrared",
    ),
    "conv-plevel": SensorConfig(
        name="conv-plevel",
        stats_csv=None,
        platforms=[],
        channels=92,  # conv but segmented by pressure level: gps_angle/gps_t/gps_q/q/t/u/v x 13 pressure levels + ps
        nc_file_template="",
        sensor_type="conv",
    ),
}


class QCLimits:
    """Conventional Observation QC filtering limits."""

    # Height limits (meters)
    HEIGHT_MIN = 0
    HEIGHT_MAX = 60000
    # Pressure limits (hPa)
    PRESSURE_MIN_GPS = 0.5
    PRESSURE_MIN_DEFAULT = 200
    PRESSURE_MAX = 1100


# Concept of platform for conv is only used in etl, does not apply outside of etl. All conv obs have platform 0
@dataclass(frozen=True)
class ConvChannel:
    """Conv sensor channel definition, used for ETL and creating channel table"""

    name: str
    platform: str
    nc_column: str
    min_valid: float
    max_valid: float


CONV_CHANNELS = [
    ConvChannel("gps_angle", "gps", "Observation", 0.0, 0.1),
    ConvChannel("gps_t", "gps", "Temperature_at_Obs_Location", 150, 350),
    ConvChannel("gps_q", "gps", "Specific_Humidity_at_Obs_Location", 0.0, 1.0),
    ConvChannel("ps", "ps", "Observation", 500.0, 1100.0),
    ConvChannel("q", "q", "Observation", 0, 1),
    ConvChannel("t", "t", "Observation", 150, 350),
    ConvChannel("u", "uv", "u_Observation", -100, 100),
    ConvChannel("v", "uv", "v_Observation", -100, 100),
]

CONV_CHANNEL_NAMES = [c.name for c in CONV_CHANNELS]
CONV_PLATFORMS = list(dict.fromkeys(c.platform for c in CONV_CHANNELS))
CONV_GPS_CHANNELS = [i for i, c in enumerate(CONV_CHANNELS) if c.platform == "gps"]
CONV_GPS_LEVEL2_CHANNELS = [
    i for i, c in enumerate(CONV_CHANNELS) if c.name in ("gps_t", "gps_q")
]
CONV_UV_CHANNELS = [i for i, c in enumerate(CONV_CHANNELS) if c.platform == "uv"]
CONV_UV_IN_SITU_TYPES = [220, 221, 229, 230, 231, 232, 233, 234, 235, 280, 282]


def _build_conv_channel_map() -> dict[str, int]:
    """Build map from platform name to first channel ID (1-indexed)."""
    channel_map = {}
    for i, channel in enumerate(CONV_CHANNELS, start=1):
        if channel.platform not in channel_map:
            channel_map[channel.platform] = i
    return channel_map


CONV_CHANNEL_MAP = _build_conv_channel_map()


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


PLATFORM_NAME_TO_ID = {
    "aqua": 0,
    "aura": 1,
    "f10": 2,
    "f11": 3,
    "f13": 4,
    "f14": 5,
    "f15": 6,
    "g08": 7,
    "g10": 8,
    "g11": 9,
    "g12": 10,
    "m08": 11,
    "m09": 12,
    "m10": 13,
    "metop-a": 14,
    "metop-b": 15,
    "metop-c": 16,
    "n11": 17,
    "n12": 18,
    "n14": 19,
    "n15": 20,
    "n16": 21,
    "n17": 22,
    "n18": 23,
    "n19": 24,
    "n20": 25,
    "npp": 26,
    "gps": 27,
    "ps": 28,
    "q": 29,
    "t": 30,
    "uv": 31,
    # NOAA-21 carries ATMS from November 2022. may cause issue with old ckpts
    "n21": 32,
}

PLATFORM_ID_TO_NAME = {v: k for k, v in PLATFORM_NAME_TO_ID.items()}

NPLATFORMS = _next_power_of_two(max(len(PLATFORM_NAME_TO_ID), 64))  # 64

SENSOR_OFFSET = {}
offset = 0
for name, cfg in SENSOR_CONFIGS.items():
    SENSOR_OFFSET[name] = offset
    offset += cfg.channels
NCHANNEL = _next_power_of_two(max(offset, 1024))  # 1024

# GPS channel Global_Channel_IDs (for use in SQL queries against parquet)
CONV_GPS_GLOBAL_IDS = [SENSOR_OFFSET["conv"] + i for i in CONV_GPS_CHANNELS]


def conv_var_global_ids(tokens):
    """Resolve conv channel names ('gps_angle', 't', 'q', 'ps', 'u', 'v', ...) or
    platforms ('gps', 'uv', 'ps', 'q', 't') to unified global channel IDs.
    'pres' is accepted as an alias for 'ps'."""
    name_to_idx = {c.name: i for i, c in enumerate(CONV_CHANNELS)}
    plat_to_idxs = {}
    for i, c in enumerate(CONV_CHANNELS):
        plat_to_idxs.setdefault(c.platform, []).append(i)
    idxs = set()
    for token in tokens:
        token = "ps" if token == "pres" else token
        if token in plat_to_idxs:
            idxs.update(plat_to_idxs[token])
        elif token in name_to_idx:
            idxs.add(name_to_idx[token])
        else:
            valid = sorted(set(name_to_idx) | set(plat_to_idxs))
            raise ValueError(
                f"unknown conv var {token!r}; valid: {valid} (+ alias pres)"
            )
    return [SENSOR_OFFSET["conv"] + i for i in sorted(idxs)]


SENSOR_NAME_TO_ID = {name: idx for idx, name in enumerate(SENSOR_CONFIGS.keys())}
SENSOR_ID_TO_NAME = {idx: name for name, idx in SENSOR_NAME_TO_ID.items()}


def build_channel_table() -> pa.Table:
    from healda.observations.preprocessing.conv_plevel import (
        build_conv_plevel_channel_table,
    )
    from healda.observations.schema import get_channel_table_schema

    channel_counts = [config.channels for config in SENSOR_CONFIGS.values()]
    sensor_ids = np.arange(len(channel_counts)).repeat(channel_counts)
    global_ids = np.arange(sensor_ids.size, dtype=np.uint16)
    names = []
    min_valid = []
    max_valid = []
    means = []
    stddevs = []

    for name, config in SENSOR_CONFIGS.items():
        if name == "conv":
            names.extend(channel.name for channel in CONV_CHANNELS)
            min_valid.extend(channel.min_valid for channel in CONV_CHANNELS)
            max_valid.extend(channel.max_valid for channel in CONV_CHANNELS)
        elif name == "conv-plevel":
            table = build_conv_plevel_channel_table()
            names.extend(table["name"].to_pylist())
            min_valid.extend(table["min_valid"].to_pylist())
            max_valid.extend(table["max_valid"].to_pylist())
        else:
            names.extend(f"{name}_{index:03d}" for index in range(config.channels))
            min_valid.extend([config.min_valid] * config.channels)
            max_valid.extend([config.max_valid] * config.channels)

        if name == "conv-plevel":
            means.extend(table["mean"].to_pylist())
            stddevs.extend(table["stddev"].to_pylist())
        else:
            means.extend(config.means)
            stddevs.extend(config.stds)

    conv_sensor_id = SENSOR_NAME_TO_ID["conv"]
    return pa.table(
        [
            global_ids,
            np.asarray(min_valid, dtype=np.float32),
            np.asarray(max_valid, dtype=np.float32),
            sensor_ids,
            sensor_ids == conv_sensor_id,
            names,
            np.asarray(means, dtype=np.float32),
            np.asarray(stddevs, dtype=np.float32),
        ],
        schema=get_channel_table_schema(),
    )
