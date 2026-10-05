# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which obs archive a config reads, and how far it extends.

For NNJA the end is measured from the files, not declared: each stream the config reads
covers up to the end of its unbroken run of files from ``SCAN_START`` (holes up to
``MAX_GAP`` are outages), and a run ends at the earliest of those. A stream with no file
since ``SCAN_START`` is retired and limits nothing.
"""

import dataclasses
import os

import numpy as np
import pandas as pd

from healda.datasets.base import DatasetMetadata
from healda.observations import sensors_nnja
from healda.observations.loaders.combined import NNJAConventionalLoader
from healda.observations.loaders.ufs import UFS_OBS_COVERAGE

# Continuity is checked from here on; earlier years are the bulk archive.
SCAN_START = pd.Timestamp("2025-01-01")
MAX_GAP = pd.Timedelta(days=2)
CYCLE = pd.Timedelta(hours=6)
DAY = pd.Timedelta(days=1)


@dataclasses.dataclass(frozen=True)
class StreamExtent:
    stream: str
    kind: str  # "cycle" (one file per 6 h cycle) or "daily"
    covered_until: pd.Timestamp | None  # None: retired
    outages: int
    files_beyond: int  # files after the first hole longer than MAX_GAP


def stream_paths(loader):
    """(stream, kind, path_for) for every stream an NNJA loader reads."""
    sat = loader.satellite
    for sensor in sat.sensors:
        yield sensor, "daily", lambda t, s=sensor: sat._path(s, f"{t:%Y%m%d}")
    if not isinstance(loader.conventional, NNJAConventionalLoader):
        return
    conv = loader.conventional._loader
    yield "prepbufr", "cycle", conv._path
    if conv.gpsro is not None:
        yield "gpsro", "cycle", conv.gpsro._path
    if conv.satwnd is not None:
        yield "satwnd", "daily", conv.satwnd._path


def unbroken_run(present, step, max_gap):
    """Index of the last file before the first hole longer than max_gap, and the count of
    shorter holes on the way; None if nothing is present."""
    found = np.flatnonzero(present)
    if not len(found):
        return None, 0
    end, outages = found[0], 0
    for a, b in zip(found[:-1], found[1:]):
        if (b - a - 1) * step > max_gap:
            break
        outages += int(b - a > 1)
        end = b
    return end, outages


def stream_extents(loader, until=None) -> list[StreamExtent]:
    until = pd.Timestamp.now(tz="UTC").tz_localize(None) if until is None else until
    extents = []
    for stream, kind, path_for in stream_paths(loader):
        step = CYCLE if kind == "cycle" else DAY
        times = pd.date_range(SCAN_START, until, freq=step)
        present = np.array([os.path.exists(path_for(t)) for t in times])
        end, outages = unbroken_run(present, step, MAX_GAP)
        if end is None:
            extents.append(StreamExtent(stream, kind, None, 0, 0))
            continue
        # A cycle file holds its +-3 h window; a daily file the whole day.
        covered = times[end] + (CYCLE / 2 if kind == "cycle" else DAY)
        extents.append(
            StreamExtent(
                stream,
                kind,
                covered,
                outages,
                int(present[end + 1 :].sum()),
            )
        )
    return extents


def last_analysis(extents, context_end_hours) -> pd.Timestamp:
    """Latest analysis time whose obs window every live stream covers."""
    live = [e.covered_until for e in extents if e.covered_until is not None]
    if not live:
        raise ValueError(f"no live stream among {[e.stream for e in extents]}")
    return (min(live) - pd.Timedelta(hours=context_end_hours)).floor(CYCLE)


def obs_coverage(obs_config) -> DatasetMetadata:
    """Coverage of the obs archive this config reads."""
    if not obs_config.use_nnja_sat:
        return UFS_OBS_COVERAGE
    from healda.datasets.da.tasks import build_obs_loader

    loader = build_obs_loader(obs_config, training=False)
    end = last_analysis(stream_extents(loader), obs_config.context_end)
    return dataclasses.replace(sensors_nnja.OBS_COVERAGE, end=str(end))
