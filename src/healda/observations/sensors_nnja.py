# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel and sensor vocabulary for the wide NNJA Parquet archive.

Sensor names are the archive's directory names and channel numbers are each instrument's own
full numbering, both taken from the per-sensor normalization CSVs. `sensors.py` instead describes
the GSI replay's assimilated subsets, so the two channel vocabularies are not interchangeable and
a global channel id from one means nothing in the other. Platform ids are shared, and come from
`sensors.PLATFORM_NAME_TO_ID`.

Statistics and validity gates are in brightness temperature, so a loader reading this registry
has to emit Kelvin: IASI and CrIS are stored as spectral radiance and need Planck inversion.
"""

import pathlib

from healda.datasets.base import DatasetMetadata, TimeUnit
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa

from healda.observations.schema import get_channel_table_schema

NORMALIZATION_DIR = pathlib.Path(__file__).parent / "normalizations" / "nnja"

# The satellite sensors bind the end: atms, amsua, mhs and cris stop 2026-07-14,
# cycles run to 07-26, gpsro to 09-12. Obs outside this return empty, not an error.
OBS_COVERAGE = DatasetMetadata(
    name="nnja_obs",
    start="2000-01-01 00:00:00",
    end="2026-07-14 18:00:00",
    time_step=6,
    time_unit=TimeUnit.HOUR,
)

# Declaration order fixes the global channel id layout. Appending a sensor is safe; reordering
# or changing a channel set renumbers everything after it and invalidates trained embeddings.
SENSOR_ORDER = ("atms", "amsua", "amsub", "mhs", "airs", "iasi", "cris")

SENSOR_TYPE = {
    "atms": "microwave",
    "amsua": "microwave",
    "amsub": "microwave",
    "mhs": "microwave",
    "airs": "infrared",
    "iasi": "infrared",
    "cris": "infrared",
}

# Raw IR channels, not PCA/latent scores
DEFAULT_SENSORS = tuple(
    n for n in SENSOR_ORDER if SENSOR_TYPE[n] in ["microwave", "infrared"]
)

# Brightness temperature gates, as in the GSI vocabulary.
GATE_KELVIN = {"microwave": (0.0, 400.0), "infrared": (150.0, 350.0)}

SENSOR_NAME_TO_ID = {name: index for index, name in enumerate(SENSOR_ORDER)}


@dataclass(frozen=True)
class NNJASensorConfig:
    name: str
    sensor_type: str
    channels: np.ndarray  # archive channel numbers, ascending
    platform_ids: tuple[int, ...]
    means: np.ndarray  # over all platforms, per channel, Kelvin
    stds: np.ndarray
    min_valid: float
    max_valid: float


def _read(name: str) -> NNJASensorConfig:
    frame = pd.read_csv(NORMALIZATION_DIR / f"{name}_normalizations.csv")
    # Platform_ID -1 is the all-platform aggregate; the rest are per-platform rows.
    aggregate = frame[frame["Platform_ID"] == -1].sort_values("Raw_Channel_ID")
    # A new sensor's CSV is written by hand, and each of these lands silently: no aggregate
    # gives the sensor an empty axis, a repeated channel breaks the searchsorted id lookup,
    # and a non-finite or zero stddev z-scores the channel to NaN or inf.
    if aggregate.empty:
        raise ValueError(f"{name}: no Platform_ID == -1 aggregate rows")
    if aggregate["Raw_Channel_ID"].duplicated().any():
        raise ValueError(f"{name}: repeated Raw_Channel_ID among the aggregate rows")
    if not np.isfinite(aggregate["obs_mean"]).all():
        raise ValueError(f"{name}: non-finite obs_mean")
    if not (np.isfinite(aggregate["obs_std"]) & (aggregate["obs_std"] > 0)).all():
        raise ValueError(f"{name}: obs_std must be finite and positive")
    sensor_type = SENSOR_TYPE[name]
    min_valid, max_valid = GATE_KELVIN[sensor_type]
    return NNJASensorConfig(
        name=name,
        sensor_type=sensor_type,
        channels=aggregate["Raw_Channel_ID"].to_numpy(),
        # json cannot encode an int64 returned from pandas
        platform_ids=tuple(
            sorted(int(p) for p in frame["Platform_ID"].unique() if p >= 0)
        ),
        means=aggregate["obs_mean"].to_numpy(),
        stds=aggregate["obs_std"].to_numpy(),
        min_valid=min_valid,
        max_valid=max_valid,
    )


SENSOR_CONFIGS = {name: _read(name) for name in SENSOR_ORDER}

SENSOR_OFFSET = {}
_offset = 0
for _name in SENSOR_ORDER:
    SENSOR_OFFSET[_name] = _offset
    _offset += len(SENSOR_CONFIGS[_name].channels)

# Slots the vocabulary occupies, which the model's channel embedding table has to cover.
NCHANNEL = _offset


def get_global_channel_id(sensor: str, raw_channel_ids) -> np.ndarray:
    """Map an instrument's own channel numbers to unified NNJA global ids."""
    channels = SENSOR_CONFIGS[sensor].channels
    raw = np.asarray(raw_channel_ids)
    position = np.clip(np.searchsorted(channels, raw), 0, len(channels) - 1)
    unknown = raw[channels[position] != raw]
    if unknown.size:
        raise ValueError(
            f"{sensor}: channels outside the archive vocabulary: {np.unique(unknown).tolist()}"
        )
    return (position + SENSOR_OFFSET[sensor]).astype(np.uint16)


def channel_table() -> pa.Table:
    """Per-channel stats and gates for this vocabulary, one row per global id in id order.

    IR sounders zscored in Kelvin (needs conversion from spectral radiance)
    """
    names, means, stds, min_valid, max_valid, sensor_id = [], [], [], [], [], []
    for index, name in enumerate(SENSOR_ORDER):
        config = SENSOR_CONFIGS[name]
        count = len(config.channels)
        names.extend(f"{name}_{channel:05d}" for channel in config.channels)
        means.append(config.means)
        stds.append(config.stds)
        min_valid.extend([config.min_valid] * count)
        max_valid.extend([config.max_valid] * count)
        sensor_id.extend([index] * count)
    return pa.table(
        [
            np.arange(NCHANNEL, dtype=np.uint16),
            np.array(min_valid, dtype=np.float32),
            np.array(max_valid, dtype=np.float32),
            np.array(sensor_id, dtype=np.uint16),
            np.zeros(NCHANNEL, dtype=bool),
            names,
            np.concatenate(means).astype(np.float32),
            np.concatenate(stds).astype(np.float32),
        ],
        schema=get_channel_table_schema(),
    )


if __name__ == "__main__":
    import argparse

    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=channel_table.__doc__)
    parser.add_argument("path", help="where to write channel_table.parquet")
    pq.write_table(channel_table(), parser.parse_args().path)
