# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for satwnd time composition.

Misframed historical subsets decode years outside the datetime64[ns] range and
impossible days; numpy's out-of-range datetime cast segfaults rather than
raising. The real-file test pins its input by SHA-256.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from healda.observations.loaders.nnja_base import archive_cycle_and_window
from scripts.etl.nnja.satwnd.run_production_etl import stream_path
from scripts.etl.nnja.satwnd.archive_etl import (
    _compose_utc,
    project_table,
)

SEGFAULT_CYCLE = stream_path("satwnd", 2005, 6, "gdas.20050607.t12z.satwnd.tm00.bufr_d")
SEGFAULT_CYCLE_SHA256 = (
    "0db49710487c6e4d7032579d1b0a666f078f34d3726263695f9c9fffc4e5e21f"
)


def _f(*values: float) -> np.ndarray:
    return np.asarray(values, dtype="float64")


def test_compose_utc__years_outside_ns_range_become_null_not_a_crash():
    """Years numpy cannot represent must yield NaT, never reach the ns cast."""
    # 2005 is a valid timestamp; the others are outside the ns range.
    year = _f(1600, 1677, 2005, 2264, 2497)
    month = _f(6, 6, 6, 6, 6)
    day = _f(7, 7, 7, 7, 7)
    hour = _f(12, 12, 12, 12, 12)
    minute = _f(0, 0, 30, 0, 0)
    second = _f(0, 0, 15, 0, 0)

    out = _compose_utc(year, month, day, hour, minute, second)

    assert np.isnat(out[0]), "year 1600 is before the datetime64[ns] epoch"
    assert np.isnat(out[1]), "year 1677 is before the datetime64[ns] epoch"
    assert not np.isnat(out[2]), "an in-range timestamp must still compose"
    assert out[2] == np.datetime64("2005-06-07T12:30:15", "ns")
    assert np.isnat(out[3]), "year 2264 is past the datetime64[ns] epoch"
    assert np.isnat(out[4]), "year 2497 is past the datetime64[ns] epoch"


def test_compose_utc__impossible_day_is_null_and_does_not_shift_other_rows():
    """Garbage DAYS values are rejected without disturbing neighbouring rows."""
    year = _f(2005, 2005, 2005, 2005)
    month = _f(6, 6, 2, 6)
    day = _f(62, 0, 31, 7)
    hour = _f(12, 12, 12, 12)
    minute = _f(0, 0, 0, 0)
    second = _f(0, 0, 0, 0)

    out = _compose_utc(year, month, day, hour, minute, second)

    assert np.isnat(out[0]), "June has no 62nd day"
    assert np.isnat(out[1]), "day 0 is not a date"
    assert np.isnat(out[2]), "February has no 31st"
    assert out[3] == np.datetime64("2005-06-07T12:00:00", "ns")


def test_da_window_preserves_missing_observation_time():
    times = pd.Series(
        [
            np.datetime64("2025-03-31T23:59", "ns"),
            np.datetime64("2025-04-01T00:01", "ns"),
            np.datetime64("NaT", "ns"),
        ]
    )

    windows = archive_cycle_and_window(times)[1]

    assert windows[0] == np.datetime64("2025-04-01T00:00", "ns")
    assert windows[1] == np.datetime64("2025-04-01T03:00", "ns")
    assert np.isnat(windows[2])


def test_projection_preserves_valid_values_above_legacy_bmiss_threshold():
    # Both rows need a valid time, or `project_table` drops them.
    n = 2
    table = pa.table(
        {
            "SCCF": pa.array([2.65e13, None], type=pa.float64()),
            "YEAR": pa.array([2005] * n, type=pa.float64()),
            "MNTH": pa.array([6] * n, type=pa.float64()),
            "DAYS": pa.array([7] * n, type=pa.float64()),
            "HOUR": pa.array([12] * n, type=pa.float64()),
            "MINU": pa.array([0] * n, type=pa.float64()),
            "SECO": pa.array([0] * n, type=pa.float64()),
        }
    )

    projected = project_table(
        table,
        "3-63-251",
        datetime(2005, 6, 7, 12, tzinfo=timezone.utc),
        active_elements=(("0-02-153", "SCCF"),),
    )

    assert projected.num_rows == n, "quarantine dropped a row with a valid time"
    values = projected.column("satellite_channel_centre_frequency")
    assert values[0].as_py() == 2.65e13
    assert values[1].as_py() is None


@pytest.mark.skipif(
    not SEGFAULT_CYCLE.exists(),
    reason=f"NNJA archive not mounted; {SEGFAULT_CYCLE} is unavailable",
)
def test_real_cycle_2005_06_07_t12z_projects_without_crashing():
    """A cycle with misframed-subset dates projects with every time on its date."""
    from scripts.etl.nnja.satwnd.archive_etl import (
        iter_projected_batches,
    )

    digest = hashlib.sha256()
    with SEGFAULT_CYCLE.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert (
        digest.hexdigest() == SEGFAULT_CYCLE_SHA256
    ), "fixture bytes changed; the regression it pins is no longer covered"

    cycle = datetime(2005, 6, 7, 12, tzinfo=timezone.utc)
    rows = 0
    missing_times = 0
    wrong_days = 0
    expected_day = np.datetime64("2005-06-07", "D")
    for table, _ in iter_projected_batches(str(SEGFAULT_CYCLE), cycle):
        rows += table.num_rows
        times = table.column("time_utc").to_numpy(zero_copy_only=False)
        missing_times += int(np.isnat(times).sum())
        wrong_days += int(
            (times[~np.isnat(times)].astype("datetime64[D]") != expected_day).sum()
        )

    # Row count alone does not detect misframing, so times are checked too.
    assert rows == 93128
    assert missing_times == 0
    assert wrong_days == 0
