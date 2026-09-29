# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every NNJA archive loader does identically. Reading and screening stay per loader."""

from __future__ import annotations

import functools
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd
import pyarrow as pa

from healda.observations.schema import (
    GLOBAL_CHANNEL_ID,
    SENSOR_ID,
    get_combined_observation_schema,
)

LOCAL_CHANNEL_ID = pa.field("local_channel_id", pa.uint16())

__all__ = [
    "GLOBAL_CHANNEL_ID",
    "LOCAL_CHANNEL_ID",
    "SENSOR_ID",
    "NNJAArchiveLoader",
    "CycleTableSource",
    "SampleDropout",
    "archive_cycle_and_window",
    "row_generator",
]


class NNJAArchiveLoader:
    obs_context_hours: tuple[int, int]
    data_spacing: int
    dropout_scope: str = "row"

    def _dropout(self) -> SampleDropout:
        return SampleDropout(self.dropout_scope)

    @functools.cached_property
    def output_schema(self) -> pa.Schema:
        return (
            get_combined_observation_schema().append(LOCAL_CHANNEL_ID).append(SENSOR_ID)
        )

    def _empty(self) -> pa.Table:
        return pa.table(
            [pa.array([], type=field.type) for field in self.output_schema],
            schema=self.output_schema,
        )

    def _interval_times(self, target: datetime) -> pd.DatetimeIndex:
        # DA windows are end-aligned: the first starts one spacing after the context.
        start, end = self.obs_context_hours
        return pd.date_range(
            target + pd.Timedelta(hours=start + self.data_spacing),
            target + pd.Timedelta(hours=end),
            freq=f"{self.data_spacing}h",
        )

    @staticmethod
    def _numpy(table: pa.Table, name: str, dtype=None) -> np.ndarray:
        values = table[name].to_numpy(zero_copy_only=False)
        return (
            np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
        )

    @staticmethod
    def _cycle_and_half(window: pd.Timestamp) -> tuple[pd.Timestamp, bool]:
        # PrepBUFR and GPS-RO ship one file per 6-hourly cycle, DHR in [-3, +3], so each
        # file serves two of the model's 3-hour windows.
        window = pd.Timestamp(window).tz_localize(None)
        if window.minute or window.second or window.microsecond or window.hour % 3:
            raise ValueError(f"observation window must be 3-hour aligned, got {window}")
        cycle = window.floor("6h")
        return cycle, bool(window.hour % 6)


class CycleTableSource(Protocol):
    """Archive-schema rows of one NCEP cycle file, looked up by that file's cycle.

    What an archive loader reads instead of its parquet files when given one; a dict
    ``{cycle: table}`` such as the ``adapters.e2s_nnja`` output satisfies it.
    """

    def get(self, cycle: pd.Timestamp) -> pa.Table | None: ...


def archive_cycle_and_window(time: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The 6-hourly cycle file holding each time, and its end-aligned 3-hour window.

    NCEP dumps cover ``[cycle - 3h, cycle + 3h)``; the archives window a file's rows as
    ``cycle`` when ``time <= cycle``, else ``cycle + 3h``. Inverse of ``_cycle_and_half``.
    """
    cycle = (time + pd.Timedelta(hours=3)).dt.floor("6h")
    window = cycle.where(time <= cycle, cycle + pd.Timedelta(hours=3))
    return cycle, window


class SampleDropout:
    """One ambient draw per sel_time, so time-parallel ranks stay in lockstep."""

    def __init__(self, scope: str) -> None:
        self._parent = np.random.default_rng(np.random.randint(2**31))
        self._scope = scope

    def rate(self, configured: float) -> float:
        # "sample" spends a draw to pin the rate to 0 or 1; "row" passes it through.
        if configured and self._scope == "sample":
            return float(self._parent.random() < configured)
        return configured

    def seeds(self, units) -> dict:
        # One per file: each is decoded on its own thread and a Generator is not shared.
        return {unit: int(self._parent.integers(2**31)) for unit in units}


def row_generator(seed: int | None) -> np.random.Generator | None:
    return np.random.default_rng(seed) if seed is not None else None
