# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream eras, stream provenance through the writer, and footer revisions.

The Arrow IPC spool allows one non-delta dictionary per field, so a per-batch
`source_stream` dictionary fails only when a later batch has another value.
This is tested through the real writer with a read-back from the Parquet file.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.etl.nnja.satwnd.archive_etl import (
    SOURCE_STREAM_TYPE,
    primary_stream_for_year,
    source_stream_array,
    streams_for_year,
)
from scripts.etl.nnja.satwnd.amv_writer import (
    AmvArchiveDayWriter,
    AmvFileProvenance,
)


def test_stream_eras_cover_the_archive_and_reject_the_unlisted():
    assert streams_for_year(1979) == ("merged",)
    assert streams_for_year(2019) == ("merged",)
    assert primary_stream_for_year(2019) == "merged"
    assert streams_for_year(2020) == ("satwnd",)
    assert streams_for_year(2025) == ("satwnd",)
    assert primary_stream_for_year(2025) == "satwnd"
    with pytest.raises(Exception):
        streams_for_year(1900)


def _provenance(**overrides):
    base = dict(
        utc_day=datetime(2019, 8, 1, tzinfo=timezone.utc),
        source_files=({"cycle": "12", "path": "/dev/null"},),
        source_structures={},
        decoder_version="0.1.0",
        etl_git_sha="testsha",
        decoder_revision="testrev",
        etl_revision="testetl",
    )
    base.update(overrides)
    return AmvFileProvenance(**base)


def _stream_batch(stream, n, start_hour):
    times = [np.datetime64(f"2019-08-01T{start_hour:02d}:00:00", "ns")] * n
    return pa.table(
        {
            "time_utc": pa.array(np.array(times, dtype="datetime64[ns]")).cast(
                pa.timestamp("ns", tz="UTC")
            ),
            "da_window": pa.array(
                [datetime(2019, 8, 1, 12, tzinfo=timezone.utc)] * n,
                pa.timestamp("ns", tz="UTC"),
            ),
            "latitude": pa.array(np.arange(n, dtype="float32")),
            "source_stream": source_stream_array(stream, n),
            "also_in_other_stream": pa.array([False] * n, pa.bool_()),
        }
    )


def test_source_stream_dictionary_survives_multiple_batches_through_the_writer(
    tmp_path,
):
    """Batches alternate stream values, which a per-batch dictionary would fail on."""
    out = tmp_path / "day.parquet"
    schema = _stream_batch("satwnd", 1, 12).schema
    writer = AmvArchiveDayWriter(
        out, schema=schema, provenance=_provenance(), scratch_dir=tmp_path
    )
    try:
        writer.write(_stream_batch("satwnd", 3, 12))
        writer.write(_stream_batch("merged", 2, 12))
        writer.write(_stream_batch("satwnd", 1, 12))
        writer.finish_windows([datetime(2019, 8, 1, 12, tzinfo=timezone.utc)])
    finally:
        writer.close()

    table = pq.read_table(out)
    assert table.num_rows == 6
    field = table.schema.field("source_stream")
    assert pa.types.is_dictionary(field.type), f"lost dictionary encoding: {field.type}"
    assert field.type.index_type == SOURCE_STREAM_TYPE.index_type
    counts = table.column("source_stream").to_pylist()
    assert counts.count("satwnd") == 4
    assert counts.count("merged") == 2


def test_project_table_quarantines_rows_with_no_usable_time():
    """A real cycle containing undecodable times yields no null time_utc or da_window."""
    from scripts.etl.nnja.satwnd.archive_etl import (
        iter_projected_batches,
    )

    # This cycle contains NC005064 reports with no usable time.
    from scripts.etl.nnja.satwnd.run_production_etl import stream_path

    path = stream_path("merged", 2005, 3, "gdas.20050329.t12z.satwnd.tm00.bufr_d")
    if not path.exists():
        pytest.skip("source cycle not staged")

    cycle = datetime(2005, 3, 29, 12, tzinfo=timezone.utc)
    seen = 0
    for table, _ in iter_projected_batches(str(path), cycle):
        if not table.num_rows:
            continue
        seen += table.num_rows
        times = (
            table.column("time_utc")
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype("datetime64[ns]")
        )
        windows = (
            table.column("da_window")
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype("datetime64[ns]")
        )
        # Both are required archive fields; neither may contain nulls.
        assert not np.isnat(times).any(), "null time_utc survived quarantine"
        assert not np.isnat(windows).any(), "null da_window reached the writer"
    assert seen > 0, "no rows decoded, test proves nothing"


def test_etl_revision_is_semantic_not_a_git_sha():
    """The resume cache key is a chosen revision, separate from the git SHA."""
    import re
    from scripts.etl.nnja.satwnd.archive_etl import (
        ETL_CONTENT_REVISION,
        etl_git_sha,
    )

    assert not re.fullmatch(r"[0-9a-f]{7,12}", ETL_CONTENT_REVISION), (
        "ETL_CONTENT_REVISION looks like a git SHA; it is the resume cache key "
        "and must be a semantic revision"
    )
    assert ETL_CONTENT_REVISION.startswith("amv-etl-")

    sha = etl_git_sha()
    assert sha, "git sha must never be empty"
    assert sha != ETL_CONTENT_REVISION


def test_footer_records_content_revision_and_git_sha_separately(tmp_path):
    out = tmp_path / "day.parquet"
    schema = _stream_batch("satwnd", 1, 12).schema
    provenance = _provenance(etl_git_sha="abc1234")
    writer = AmvArchiveDayWriter(
        out, schema=schema, provenance=provenance, scratch_dir=tmp_path
    )
    try:
        writer.write(_stream_batch("satwnd", 2, 12))
        writer.finish_windows([datetime(2019, 8, 1, 12, tzinfo=timezone.utc)])
    finally:
        writer.close()

    meta = pq.ParquetFile(out).schema_arrow.metadata
    assert meta[b"amv.etl_revision"] == b"testetl"
    assert meta[b"amv.etl_git_sha"] == b"abc1234"
    assert meta[b"amv.etl_revision"] != meta[b"amv.etl_git_sha"]
    assert meta[b"amv.dedup_dropped_rows"] == b"0"
    assert b"EUMETSAT" in meta[b"amv.data_attribution"]
