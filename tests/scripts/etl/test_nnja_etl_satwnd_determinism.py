# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archive output must not depend on the decode thread count.

Table content is compared, not file bytes, since Parquet metadata may differ.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import earth2bufr
import pytest

from scripts.etl.nnja.satwnd.amv_writer import write_day_resumable
from scripts.etl.nnja.satwnd.run_production_etl import stream_path

# A pre-2010 day containing corrupt dates.
DAY = datetime(2005, 6, 7)
EXPECTED_ROWS = 301_328
CYCLE_SHA256 = {
    "12": "0db49710487c6e4d7032579d1b0a666f078f34d3726263695f9c9fffc4e5e21f",
}


def _cycle_paths(day: datetime) -> dict[str, str]:
    out = {}
    for hour in ("00", "06", "12", "18"):
        name = f"gdas.{day:%Y%m%d}.t{hour}z.satwnd.tm00.bufr_d"
        path = stream_path("satwnd", day.year, day.month, name)
        if path.exists():
            out[hour] = str(path)
    return out


def _content_digest(path: Path) -> tuple[int, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    digest = hashlib.sha256()
    digest.update(str(table.num_rows).encode())
    for index, field in enumerate(table.schema):
        digest.update(f"{field.name}|{field.type}".encode())
        column = table.column(index).combine_chunks()
        for start in range(0, table.num_rows, 100_000):
            digest.update(repr(column.slice(start, 100_000).to_pylist()).encode())
    return table.num_rows, digest.hexdigest()


@pytest.mark.skipif(
    len(_cycle_paths(DAY)) < 4, reason="the day's four cycles are not staged"
)
def test_output_is_identical_across_thread_counts(tmp_path: Path):
    """The same day written at 1 and 8 decode threads must be identical."""
    paths = _cycle_paths(DAY)
    for hour, path in paths.items():
        expected = CYCLE_SHA256.get(hour)
        if expected is None:
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, f"t{hour}z fixture bytes changed"

    digests = {}
    default_threads = earth2bufr.get_num_threads()
    try:
        for threads in (1, 8):
            earth2bufr.set_num_threads(threads)
            root = tmp_path / f"threads{threads}"
            # Fixed revisions, so both runs write identical footers.
            write_day_resumable(
                DAY,
                paths,
                str(root),
                sensor="satwnd",
                decoder_revision="determinism-fixed-decoder",
                etl_revision="determinism-fixed-etl",
            )
            parquet = root / "satwnd" / f"{DAY:%Y%m%d}.parquet"
            assert parquet.exists(), f"no output at {threads} threads"
            digests[threads] = _content_digest(parquet)
    finally:
        earth2bufr.set_num_threads(default_threads)

    assert digests[1][0] == EXPECTED_ROWS
    assert digests[1] == digests[8], (
        "archive content depends on thread count; resumability and re-runs are "
        "not reproducible"
    )
