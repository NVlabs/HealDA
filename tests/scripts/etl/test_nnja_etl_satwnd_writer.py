# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMV writer contract, resumability, and failure-ledger regressions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import earth2bufr
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from healda.observations import schema as reader_schema

from scripts.etl.nnja.archive_common.hashing import sha256_file
from scripts.etl.nnja.satwnd import amv_writer
from scripts.etl.nnja.satwnd import archive_etl
from scripts.etl.nnja.satwnd.amv_writer import (
    AmvDayTask,
    AmvFileProvenance,
    _validate_reusable_output,
    bind_amv_file_metadata,
    clear_amv_failure,
    write_amv_failure,
)


def _write_contract_file(path: Path, source: Path, schema: pa.Schema) -> None:
    sha256, size_bytes = sha256_file(str(source))
    record = {"path": str(source), "sha256": sha256, "size_bytes": size_bytes}
    provenance = AmvFileProvenance(
        utc_day=datetime(2025, 4, 1, tzinfo=timezone.utc),
        source_files=({**record, "cycle": "00"},),
        decoder_version="0.1.0",
        etl_git_sha="abc1234",
        decoder_revision="decoder-a",
        etl_revision="etl-a",
    )
    window = pa.array([datetime(2025, 4, 1)], type=pa.timestamp("ns"))
    arrays = [window if f.name == "da_window" else pa.nulls(1, f.type) for f in schema]
    schema = bind_amv_file_metadata(schema, provenance)
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def test_reusable_output_requires_exact_sources_and_code_revisions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bufr"
    source.write_bytes(b"BUFR source A")
    output = tmp_path / "day.parquet"
    _write_contract_file(output, source, archive_etl.load_declared_schema())
    cycles = {"00": str(source)}

    contract = _validate_reusable_output(
        output,
        cycles,
        decoder_revision="decoder-a",
        etl_revision="etl-a",
    )
    assert contract["rows"] == 1

    with pytest.raises(AssertionError, match="decoder_revision"):
        _validate_reusable_output(
            output,
            cycles,
            decoder_revision="decoder-b",
            etl_revision="etl-a",
        )

    moved = tmp_path / "elsewhere.bufr"
    moved.write_bytes(source.read_bytes())
    _validate_reusable_output(
        output, {"00": str(moved)}, decoder_revision="decoder-a", etl_revision="etl-a"
    )

    source.write_bytes(b"BUFR source B")
    with pytest.raises(AssertionError, match="sources"):
        _validate_reusable_output(
            output,
            cycles,
            decoder_revision="decoder-a",
            etl_revision="etl-a",
        )


def test_reusable_output_requires_the_declared_columns(tmp_path: Path) -> None:
    source = tmp_path / "source.bufr"
    source.write_bytes(b"BUFR source A")
    output = tmp_path / "day.parquet"
    declared = archive_etl.load_declared_schema()
    _write_contract_file(
        output, source, declared.remove(declared.get_field_index("wind_speed"))
    )
    with pytest.raises(AssertionError, match="columns"):
        _validate_reusable_output(
            output,
            {"00": str(source)},
            decoder_revision="decoder-a",
            etl_revision="etl-a",
        )


def test_write_day_aborts_and_propagates_decode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bufr"
    source.write_bytes(b"BUFR")
    state = {"aborted": False}

    class FakeWriter:
        populated_windows = ()

        def __init__(self, *_args, **_kwargs):
            pass

        def abort(self) -> None:
            state["aborted"] = True

    def fail_decode(*_args, **_kwargs):
        raise ValueError("bad template")
        yield  # pragma: no cover

    monkeypatch.setattr(amv_writer, "read_source_structures", lambda _path: {})
    monkeypatch.setattr(
        amv_writer,
        "load_declared_schema",
        lambda: pa.schema([pa.field("da_window", pa.timestamp("ns"))]),
    )
    monkeypatch.setattr(amv_writer, "AmvArchiveDayWriter", FakeWriter)
    monkeypatch.setattr(amv_writer, "iter_projected_batches", fail_decode)

    with pytest.raises(ValueError, match="bad template"):
        amv_writer.write_day(
            datetime(2025, 4, 1, tzinfo=timezone.utc),
            {"00": str(source)},
            str(tmp_path / "out"),
            decoder_revision="decoder-a",
            etl_revision="etl-a",
        )
    assert state["aborted"]


def test_amv_failure_marker_is_atomic_and_cleared_after_success(tmp_path: Path) -> None:
    task = AmvDayTask(
        day=datetime(2025, 4, 1, tzinfo=timezone.utc),
        cycle_paths={"00": "source.bufr"},
        output_root=str(tmp_path / "archive"),
        decoder_revision="decoder-a",
        etl_revision="etl-a",
    )
    marker = write_amv_failure(tmp_path / "errors", task, ValueError("bad template"))

    assert marker.exists()
    assert not list(marker.parent.glob(f".{marker.name}.*.tmp"))
    clear_amv_failure(tmp_path / "errors", task)
    assert not marker.exists()


def test_source_structure_identity_includes_full_operator_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def message(index: int, descriptors: tuple[str, ...]) -> earth2bufr.MessageInfo:
        return earth2bufr.MessageInfo(
            **dict.fromkeys(earth2bufr.MessageInfo.__dataclass_fields__, 0)
            | {
                "source_message_index": index,
                "data_category": 5,
                "subsets": 1,
                "descriptors": descriptors,
                "table_epoch": 0,
            }
        )

    file_info = earth2bufr.FileInfo(
        messages=(
            message(0, ("2-01-130", "3-63-251")),
            message(1, ("2-01-131", "3-63-251")),
        ),
        table_epochs=(earth2bufr.TableEpoch("a" * 64, ()),),
    )

    class Handle:
        info = file_info

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            pass

        def source_structure(self, source_message_index: int):
            return {
                "embedded_table_sha256": "a" * 64,
                "active_sequences": [
                    {"descriptor": "3-63-251", "mnemonic": "NC005064"}
                ],
                "active_elements": [{"descriptor": "0-04-001", "mnemonic": "YEAR"}],
                "decoded_physical_schema": [{"name": "YEAR", "arrow_type": "double"}],
                "effective_occurrences": [
                    {
                        "descriptor": "0-04-001",
                        "mnemonic": "YEAR",
                        "decoder_path": "YEAR",
                    }
                ],
            }

    monkeypatch.setattr(archive_etl.earth2bufr, "open_bufr", lambda _path: Handle())

    structures = archive_etl.read_source_structures("operators.bufr")

    assert len(structures) == 2
    assert {
        tuple(structure["observation_descriptors"]) for structure in structures.values()
    } == {
        ("2-01-130", "3-63-251"),
        ("2-01-131", "3-63-251"),
    }
    assert {structure["root_sequence"] for structure in structures.values()} == {
        "3-63-251"
    }


def test_declared_schema_has_unique_catalogued_legacy_fields() -> None:
    schema = archive_etl.load_declared_schema()
    assert len(schema.names) == len(set(schema.names))
    assert "unclassified__ICLI" not in schema.names
    assert "unclassified__SUWS" not in schema.names
    assert schema.field("bulletin_origin_icao").type == pa.string()
    assert schema.field("source_wind_speed_units").type == pa.uint32()


def test_declared_schema_has_the_reader_fields() -> None:
    declared = archive_etl.load_declared_schema()
    for field in reader_schema.table_schema("satwnd"):
        assert declared.field(field.name).type == field.type, field.name
        assert declared.field(field.name).nullable == field.nullable, field.name
    reader_schema.validate(declared.empty_table(), "satwnd")
