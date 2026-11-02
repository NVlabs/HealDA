# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for persistent NNJA sounder archive orchestration."""

from __future__ import annotations

# ruff: noqa: S101
import json
from dataclasses import replace
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from scripts.etl.nnja.sat import archive_runner
from scripts.etl.nnja.sat.archive_runner import (
    ArchiveSourceCycle,
    SensorDayResult,
    SensorDayTask,
    discover_sensor_day_tasks,
    initialize_archive_worker,
    process_sensor_day,
    recover_sensor_day,
    run_sensor_days,
)
from scripts.etl.nnja.sat.archive_schema import (
    SourceStructure,
    decode_message_header_manifest,
)
from scripts.etl.nnja.archive_common.archive_writer import WriteSummary

_REVISION = "e" * 40


def _header(index: int, subset_count: int):
    return archive_runner.BufrMessageHeader(
        index=index,
        edition=4,
        master_table=0,
        center=7,
        sub_center=0,
        update_sequence=0,
        optional_section_present=False,
        reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc)
        + timedelta(minutes=index),
        data_category=21,
        international_sub_category=0,
        local_sub_category=0,
        master_table_version=41,
        local_table_version=0,
        subset_count=subset_count,
        compressed=True,
        section3_flags=0x40,
        descriptors=("3-61-234",),
    )


def test_message_header_manifest_round_trips_exact_ordered_metadata() -> None:
    manifest = archive_runner._message_header_manifest(
        (_header(8, 17), _header(9, 23), _header(10, 23))
    )

    decoded = decode_message_header_manifest(manifest)
    assert decoded["index"] == [8, 9, 10]
    assert decoded["subset_count"] == [17, 23, 23]
    assert decoded["center"] == [7, 7, 7]
    assert decoded["descriptors"] == [["3-61-234"]] * 3
    assert manifest["columns"]["index"]["encoding"] == "linear"
    assert manifest["columns"]["center"]["encoding"] == "constant"


_STRUCTURE = SourceStructure(
    sensor="mhs",
    embedded_table_sha256="b" * 64,
    observation_descriptors=("3-61-234",),
    active_sequences=({"descriptor": "3-61-234"},),
    active_elements=({"descriptor": "0-12-999"},),
    effective_occurrences=({"mnemonic": "TMBR"},),
    decoded_physical_schema=({"name": "TMBR"},),
)


def _task(tmp_path: Path, day: date) -> SensorDayTask:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cycles = tuple(
        datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
        for hour in (0, 6, 12, 18)
    )
    paths = tuple(tmp_path / f"mhs-{cycle:%Y%m%d%H}.bufr" for cycle in cycles)
    for path in paths:
        path.write_bytes(b"BUFR")
    return SensorDayTask(
        sensor="mhs",
        utc_day=day,
        sources=tuple(
            ArchiveSourceCycle(path=path, cycle=cycle)
            for path, cycle in zip(paths, cycles, strict=True)
        ),
        output_path=tmp_path / "mhs" / f"{day:%Y%m%d}.parquet",
        release=f"mhs-{day:%Y%m%d}",
        decoder_revision=_REVISION,
    )


def test_worker_initializer_sets_thread_budgets() -> None:
    import earth2bufr
    import torch

    saved = earth2bufr.get_num_threads(), torch.get_num_threads(), pa.cpu_count()
    try:
        initialize_archive_worker()
        assert earth2bufr.get_num_threads() == archive_runner.DECODE_THREADS
        assert torch.get_num_threads() == archive_runner.TORCH_THREADS
        assert pa.cpu_count() == archive_runner.ARROW_THREADS
    finally:
        earth2bufr.set_num_threads(saved[0])
        torch.set_num_threads(saved[1])
        pa.set_cpu_count(saved[2])


def test_process_sensor_day_streams_four_cycles_and_validates_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path, date(2024, 1, 1))
    schema = pa.schema([pa.field("value", pa.int32(), nullable=False)])
    writers = []

    class FakeSource:
        message_headers = (_header(0, 1),)

        def source_structure_bindings(self, sensor):
            assert sensor == "mhs"
            return {header.index: _STRUCTURE for header in self.message_headers}

        def decode_failures(self):
            return ()

    @contextmanager
    def fake_open_source(_path):
        yield FakeSource()

    class FakeWriter:
        def __init__(self, path, **options):
            self.path = path
            self.options = options
            self.tables = []
            self.finished = []
            self.aborted = False
            writers.append(self)

        def write(self, table):
            self.tables.append(table)

        def finish_windows(self, windows):
            self.finished.append(tuple(windows))

        @property
        def populated_windows(self):
            return tuple(window for pair in self.finished for window in pair)

        def bind_provenance(self, provenance):
            self.provenance = provenance

        def close(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"PAR1")
            windows = tuple(
                window.isoformat() for pair in self.finished for window in pair
            )
            return WriteSummary(
                path=str(self.path),
                rows=len(self.tables),
                row_groups=len(windows),
                row_group_policy="one_complete_da_window",
                windows=windows,
                contract={"path": str(self.path), "row_groups": 8},
            )

        def abort(self):
            self.aborted = True

    def fake_batches(_source, **options):
        assert options["structures"] == {0: _STRUCTURE}
        ordinal = options["source_file_ordinal"]
        yield SimpleNamespace(
            layout=SimpleNamespace(schema=schema, source_structure=_STRUCTURE),
            table=pa.table({"value": pa.array([ordinal], type=pa.int32())}),
        )

    monkeypatch.setattr(archive_runner, "open_source", fake_open_source)
    monkeypatch.setattr(archive_runner, "SounderArchiveDayWriter", FakeWriter)
    monkeypatch.setattr(
        archive_runner, "iter_open_sounder_archive_batches", fake_batches
    )

    result = process_sensor_day(task)

    assert len(writers) == 1
    assert len(writers[0].tables) == 4
    assert writers[0].options["schema"].equals(schema)
    assert writers[0].finished == [
        (source.cycle, source.cycle + timedelta(hours=3)) for source in task.sources
    ]
    provenance = writers[0].provenance
    assert [source.cycle for source in provenance.source_files] == [
        source.cycle.isoformat() for source in task.sources
    ]
    assert provenance.source_structures == (_STRUCTURE,)
    assert provenance.source_subsets_encountered == 4
    assert provenance.empty_windows == ()
    assert result.rows == 4
    assert result.row_groups == 8
    assert result.contract == {"path": str(task.output_path), "row_groups": 8}


def test_recovery_validates_published_day_unless_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path, date(2024, 1, 1))
    task.output_path.parent.mkdir(parents=True)
    task.output_path.write_bytes(b"PAR1")
    monkeypatch.setattr(
        archive_runner,
        "_validate_day_output",
        lambda task: {"rows": 4, "row_groups": 8, "windows": ["w"]},
    )

    result = recover_sensor_day(task)

    assert (result.utc_day, result.rows, result.windows) == ("2024-01-01", 4, ("w",))
    assert recover_sensor_day(replace(task, overwrite=True)) is None


def test_run_sensor_days_executes_single_worker_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path, date(2024, 1, 1))
    calls = []
    expected = SensorDayResult(
        sensor="mhs",
        utc_day="2024-01-01",
        output_path=str(task.output_path),
        rows=4,
        row_groups=8,
        output_bytes=4,
        elapsed_seconds=1.0,
        subsets_per_second=4.0,
        source_bytes=16,
        worker_peak_rss_mib=32.0,
        windows=(),
        contract=None,
    )
    monkeypatch.setattr(
        archive_runner,
        "initialize_archive_worker",
        lambda: calls.append(("initialize",)),
    )
    monkeypatch.setattr(
        archive_runner,
        "process_sensor_day",
        lambda value: calls.append(("process", value)) or expected,
    )

    results = run_sensor_days((task,), max_workers=1, progress=None)

    assert results == (expected,)
    assert calls == [
        ("initialize",),
        ("process", task),
    ]


def test_discovery_builds_complete_and_partial_days(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw_bufr"
    source_dir = raw_root / "mhs" / "2024" / "01" / "bufr"
    source_dir.mkdir(parents=True)
    for hour in (0, 6, 12, 18):
        (source_dir / f"gdas.20240101.t{hour:02d}z.1bmhs.tm00.bufr_d").write_bytes(
            b"BUFR"
        )
    (source_dir / "gdas.20240102.t00z.1bmhs.tm00.bufr_d").write_bytes(b"BUFR")

    discovered = discover_sensor_day_tasks(
        raw_root,
        tmp_path / "archive",
        decoder_revision=_REVISION,
        etl_revision="test",
        sensors=("mhs",),
    )

    assert len(discovered.tasks) == 2
    complete, partial = discovered.tasks
    assert complete.utc_day == date(2024, 1, 1)
    assert complete.output_path == tmp_path / "archive" / "mhs" / "20240101.parquet"
    assert [source.cycle.hour for source in complete.sources] == [0, 6, 12, 18]
    assert all(
        source.object_key.startswith("mhs/2024/01/bufr/") for source in complete.sources
    )
    assert partial.utc_day == date(2024, 1, 2)
    assert [source.cycle.hour for source in partial.sources] == [0]
    assert discovered.issues == ()


def test_discovery_reports_zero_source_days_in_explicit_range(tmp_path: Path) -> None:
    """An explicit range cannot silently omit an entirely absent sensor-day."""

    discovered = discover_sensor_day_tasks(
        tmp_path / "raw_bufr",
        tmp_path / "archive",
        decoder_revision=_REVISION,
        etl_revision="test",
        sensors=("mhs", "atms"),
        start_day=date(2024, 1, 1),
        end_day=date(2024, 1, 2),
    )

    assert discovered.tasks == ()
    assert [(issue.sensor, issue.utc_day) for issue in discovered.issues] == [
        ("atms", date(2024, 1, 1)),
        ("atms", date(2024, 1, 2)),
        ("mhs", date(2024, 1, 1)),
        ("mhs", date(2024, 1, 2)),
    ]
    assert all(issue.message == "no source cycles found" for issue in discovered.issues)


def test_discovery_classifies_all_zero_byte_cycles_as_upstream_warning(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw_bufr"
    source_dir = raw_root / "airs" / "2024" / "01" / "bufr"
    source_dir.mkdir(parents=True)
    for hour in (0, 6, 12, 18):
        (source_dir / f"airs_disc_final.20240101.t{hour:02d}z.bufr").touch()

    discovered = discover_sensor_day_tasks(
        raw_root,
        tmp_path / "archive",
        decoder_revision=_REVISION,
        etl_revision="test",
        sensors=("airs",),
    )

    assert discovered.tasks == ()
    assert len(discovered.issues) == 1
    issue = discovered.issues[0]
    assert issue.message == "all source cycles are zero-byte"
    assert len(issue.source_paths) == 4

    warning = archive_runner.write_unavailable_day_warning(tmp_path / "archive", issue)
    payload = json.loads(warning.read_text())
    assert payload["issue_types"] == [
        "missing_source_cycles",
        "zero_byte_source_objects",
    ]
    assert payload["missing_cycles"] == ["00Z", "06Z", "12Z", "18Z"]
    assert len(payload["zero_byte_sources"]) == 4


def test_task_index_migrates_legacy_zero_byte_issue(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_bufr"
    source_dir = raw_root / "airs" / "2024" / "01" / "bufr"
    source_dir.mkdir(parents=True)
    source_paths = []
    for hour in (0, 6, 12, 18):
        source = source_dir / f"airs_disc_final.20240101.t{hour:02d}z.bufr"
        source.touch()
        source_paths.append(str(source))
    index = tmp_path / "airs.jsonl"
    archive_runner.write_task_index(
        index,
        archive_runner.DiscoveryResult(
            tasks=(),
            issues=(
                archive_runner.DiscoveryIssue(
                    sensor="airs",
                    utc_day=date(2024, 1, 1),
                    message="no source cycles found",
                    source_paths=tuple(source_paths),
                ),
            ),
        ),
        raw_root=raw_root,
        sensors=("airs",),
    )

    loaded = archive_runner.load_task_index(
        index,
        raw_root=raw_root,
        output_root=tmp_path / "archive",
        sensors=("airs",),
        start_day=None,
        end_day=None,
        task_options={},
    )

    assert loaded.issues[0].message == "all source cycles are zero-byte"


def test_failure_writes_one_atomic_sensor_day_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path / "input", date(2024, 1, 1))

    def fail(_task):
        raise ValueError("source template changed")

    monkeypatch.setattr(archive_runner, "initialize_archive_worker", lambda: None)
    monkeypatch.setattr(archive_runner, "process_sensor_day", fail)
    error_root = tmp_path / "archive" / "_errors"

    results = run_sensor_days(
        (task,),
        max_workers=1,
        error_root=error_root,
        continue_on_error=True,
        progress=None,
    )

    assert results == ()
    payload = json.loads((error_root / "mhs" / "20240101.json").read_text())
    assert payload["stage"] == "conversion"
    assert payload["exception_type"] == "ValueError"
    assert payload["message"] == "source template changed"


def test_main_rejects_ambiguous_ranges_and_missing_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    roots = ["--raw-root", str(tmp_path), "--output-root", str(tmp_path)]
    common = ["--workers", "1", "--decoder-revision", _REVISION]
    reindex = ["--reindex", "--task-index", str(tmp_path / "i.jsonl")]
    monkeypatch.setattr(archive_runner, "NNJA_ROOT", "")
    for argv, message in (
        (common, "set NNJA_ROOT"),
        (
            [*roots, *common, "--start-day", "2024-01-02", "--end-day", "2024-01-01"],
            "must not be after",
        ),
        ([*roots, *common, *reindex, "--start-day", "2024-01-01"], "whole task index"),
    ):
        with pytest.raises(SystemExit):
            archive_runner.main(argv)
        assert message in capsys.readouterr().err


def test_rerun_keeps_warnings_of_published_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw_bufr"
    output_root = tmp_path / "archive"
    source_dir = raw_root / "mhs" / "2024" / "01" / "bufr"
    source_dir.mkdir(parents=True)
    for day in (1, 2):
        (source_dir / f"gdas.2024010{day}.t00z.1bmhs.tm00.bufr_d").write_bytes(b"BUFR")
    (output_root / "mhs").mkdir(parents=True)
    (output_root / "mhs" / "20240101.parquet").write_bytes(b"PAR1")
    published_warning = output_root / "_warnings" / "mhs" / "20240101.json"
    published_warning.parent.mkdir(parents=True)
    published_warning.write_text("validated")

    submitted = []
    monkeypatch.setattr(
        archive_runner,
        "run_sensor_days",
        lambda tasks, **_options: submitted.extend(tasks) or (),
    )

    archive_runner.main(
        [
            *("--raw-root", str(raw_root), "--output-root", str(output_root)),
            *("--workers", "1", "--decoder-revision", _REVISION, "--sensors", "mhs"),
        ]
    )

    assert [task.utc_day for task in submitted] == [date(2024, 1, 2)]
    assert published_warning.read_text() == "validated"
    pending_warning = json.loads(
        (output_root / "_warnings" / "mhs" / "20240102.json").read_text()
    )
    assert pending_warning["missing_cycles"] == ["06Z", "12Z", "18Z"]
