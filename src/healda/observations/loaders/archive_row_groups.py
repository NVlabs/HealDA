# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate a DA window in an NNJA archive: from row-group statistics on disk, no data
read, or by ``da_window`` in an in-memory table.

Every archive is written one ``da_window`` per row group. Shared rather than
per-loader: separate copies had drifted onto different index bases (physical
leaf order vs Arrow field order), which agree only until a nested column sits
ahead of ``da_window`` -- a mislabelled window with no error anywhere.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WINDOW_COLUMN = "da_window"


def window_rows(
    table: pa.Table, window: pd.Timestamp, columns: Sequence[str] | None = None
) -> pa.Table:
    """The rows of an in-memory archive table whose ``da_window`` is ``window``."""
    stamp = np.datetime64(pd.Timestamp(window).tz_localize(None).to_datetime64(), "ns")
    in_window = table[WINDOW_COLUMN].cast(pa.timestamp("ns")).to_numpy() == stamp
    if columns is not None:
        table = table.select(columns)
    return table.filter(pa.array(in_window))


def window_row_groups(
    parquet: pq.ParquetFile,
    windows: Sequence[pd.Timestamp],
    path: str = "",
    window_column: str = WINDOW_COLUMN,
) -> Iterator[tuple[pd.Timestamp, int]]:
    """Yield ``(window, row_group)`` for the row groups holding `windows`.

    Raises rather than filtering when a file breaks the one-window-per-group
    contract: such a file was not written by these ETLs, and quietly falling
    back to a row-level filter would hide that while still returning data.
    """
    names = parquet.metadata.schema.names
    if window_column not in names:
        raise ValueError(
            f"{path or 'file'}: no {window_column!r} column, so windows cannot "
            "be located from row-group statistics"
        )
    index = names.index(window_column)
    wanted = {pd.Timestamp(window).tz_localize(None) for window in windows}

    for group in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(group).column(index).statistics
        if statistics is None or not statistics.has_min_max:
            raise ValueError(
                f"{path or 'file'} row group {group}: no {window_column} "
                "statistics, so the window cannot be located"
            )
        if statistics.min != statistics.max:
            raise ValueError(
                f"{path or 'file'} row group {group} spans "
                f"{statistics.min}..{statistics.max}; the archive contract is "
                f"one {window_column} per row group"
            )
        # Archives store the window UTC-aware or naive depending on era; the
        # caller always asks in naive UTC. `replace` handles both, whereas
        # `tz_localize(None)` is only defined one way round.
        stored = pd.Timestamp(statistics.min).replace(tzinfo=None)
        if stored in wanted:
            yield stored, group
