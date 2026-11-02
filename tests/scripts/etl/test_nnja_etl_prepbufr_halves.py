# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from healda.observations import schema

from scripts.etl.nnja.conv import nnja_prepbufr_to_parquet as etl
from scripts.etl.nnja.conv.nnja_prepbufr_to_parquet import (
    HALF_GT_T0,
    HALF_LE_T0,
    CycleParquetBuilder,
    batch_to_halves,
    empty_meta_table,
    message_type_names,
)

SOURCE = "gdas.20250102.t06z.prepbufr.nr"


class FakePrepbufr:
    """Stands in for earth2bufr.open_bufr: one data message per Table A sequence."""

    def __init__(self, sequences: dict[str, str]):
        self.sequences = sequences
        messages = [
            SimpleNamespace(
                source_message_index=index,
                data_category=int(descriptor[-3:]),
                descriptors=(descriptor,),
            )
            for index, descriptor in enumerate(sequences)
        ]
        self.info = SimpleNamespace(data_messages=messages)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def source_structure(self, source_message_index):
        return {
            "active_sequences": [
                {"descriptor": d, "mnemonic": m} for d, m in self.sequences.items()
            ]
        }


@pytest.mark.parametrize(
    "sequences",
    [
        {"3-60-230": "ADPUPA", "3-60-231": "AIRCAR"},
        {"3-48-102": "ADPUPA", "3-48-103": "AIRCAR"},
    ],
)
def test_category_names_come_from_the_file_table_a(monkeypatch, sequences):
    monkeypatch.setattr(etl.earth2bufr, "open_bufr", lambda _: FakePrepbufr(sequences))
    names = message_type_names(SOURCE)
    adpupa = min(names)
    assert names == {adpupa: "ADPUPA", adpupa + 1: "AIRCAR"}

    batch = pa.RecordBatch.from_pydict(
        {"_data_category": pa.array([float(adpupa)] * 2), "DHR": [-1.0, 1.0]}
    )
    halves = batch_to_halves(batch, SOURCE, "20250102", names)
    for half in halves.values():
        assert half["data_category"].to_pylist() == [adpupa]
        assert half["data_category_name"].to_pylist() == ["ADPUPA"]


def pandas_reference(batch: pa.RecordBatch) -> dict[str, pa.Table]:
    """The pandas pass batch_to_halves stands in for."""
    df = batch.to_pandas()
    df["obs_date"] = "20250102"
    df["source_file"] = SOURCE
    df["cycle_date"] = "20250102"
    df["cycle_hour"] = "06"
    df["data_category"] = int(df["_data_category"].iloc[0])
    df["data_category_name"] = "ADPSFC"
    cycle = pd.Timestamp("2025-01-02 06:00", tz="UTC").as_unit("ns")
    df["obs_time"] = cycle + pd.to_timedelta(df["DHR"], unit="h")
    df = df.astype({c: "float32" for c in df.select_dtypes("float64").columns})
    le = df["obs_time"] <= cycle
    return {
        HALF_LE_T0: pa.Table.from_pandas(df[le], preserve_index=False),
        HALF_GT_T0: pa.Table.from_pandas(df[~le], preserve_index=False),
    }


def struct_lists(rows, field):
    entry = pa.struct([(field, pa.float64())])
    return pa.array(rows, type=pa.list_(entry))


def test_batch_to_halves_matches_the_pandas_pass():
    batch = pa.RecordBatch.from_pydict(
        {
            "_data_category": pa.array([109.0] * 4),
            "SID": pa.array(["A", None, "  B", ""]),
            "DHR": pa.array([-1.5, 0.0, 2.0, 0.25]),
            "XOB": pa.array([1.1, None, float("nan"), 3.3]),
            "COUNT": pa.array([1, None, 3, 4], type=pa.int64()),
            "INDEX": pa.array([1, 2, 3, 4], type=pa.uint64()),
            "FLAG": pa.array([True, None, False, True]),
            "FILLED": struct_lists([[{"A": 1.5}], [], None, [{"A": None}]], "A"),
            "EMPTY": struct_lists([[], [], None, []], "B"),
            "LATE": struct_lists([None, None, [{"C": 2.0}], None], "C"),
        }
    )
    expected = pandas_reference(batch)
    actual = batch_to_halves(batch, SOURCE, "20250102", {109: "ADPSFC"})

    assert actual.keys() == expected.keys()
    for key, want in expected.items():
        got = actual[key]
        assert got.schema.metadata == want.schema.metadata
        strings = pa.schema(
            [
                f.with_type(pa.string()) if f.type == pa.large_string() else f
                for f in want.schema
            ],
            metadata=want.schema.metadata,
        )
        assert got.equals(want.cast(strings), check_metadata=True)


def test_written_cycles_have_the_reader_types(tmp_path):
    batch = pa.RecordBatch.from_pydict(
        {
            "_data_category": pa.array([120.0, 120.0]),
            "SID": pa.array(["72403", None]),
            "DHR": pa.array([-1.0, 1.0]),
            **{
                name: pa.array([1.0, None])
                for name in ("XOB", "YOB", "POB", "TYP", "TOB")
            },
        }
    )
    base = schema.table_schema("prepbufr")
    full = CycleParquetBuilder(string_type=pa.string(), base_schema=base)
    full.add_halves(batch_to_halves(batch, SOURCE, "20250102", {120: "ADPUPA"}))
    full.write(tmp_path / "full.parquet")
    empty = CycleParquetBuilder(string_type=pa.string(), base_schema=base)
    empty.write(tmp_path / "empty.parquet", empty_table=empty_meta_table())

    # A cycle writes the base columns it decoded; an empty cycle writes all of them.
    assert set(base.names) <= set(pq.read_schema(tmp_path / "empty.parquet").names)
    assert set(empty_meta_table().column_names) <= set(
        pq.read_schema(tmp_path / "empty.parquet").names
    )
    for name in ("full", "empty"):
        written = pq.read_table(tmp_path / f"{name}.parquet")
        for field in base:
            if field.name in written.column_names:
                assert written.schema.field(field.name).type == field.type, field.name
                assert written.schema.field(field.name).nullable == field.nullable
        schema.validate(written, "prepbufr")
