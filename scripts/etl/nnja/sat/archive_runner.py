# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent process orchestration for daily NNJA sounder archive ETL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import socket
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import earth2bufr
import pyarrow as pa

from healda.config.environment import NNJA_ROOT, nnja_archive
from healda.utils.profiling import (
    CPU_TIMING_ENABLED,
    cpu_timing_range,
    cpu_timing_reset,
    cpu_timing_report,
)

from .archive_etl import (
    DECODER_NAME,
    BufrMessageHeader,
    initialize_hpx2048,
    iter_open_sounder_archive_batches,
    open_source,
    validate_archive_layout_compatibility,
)
from .archive_schema import (
    CATALOG_PATH,
    MESSAGE_HEADER_FIELDS,
    FileProvenance,
    SENSOR_SPECS,
    SourceFile,
    SourceStructure,
)
from ..archive_common.archive_runner import (
    exception_traceback,
    run_tasks,
    write_json_atomic,
    write_text_atomic,
)
from ..archive_common.archive_writer import WriteSummary
from ..archive_common.hashing import sha256_file
from .archive_writer import SounderArchiveDayWriter, validate_parquet_contract

# Per worker process.
DECODE_THREADS = 4
TORCH_THREADS = 2
ARROW_THREADS = 2
DEFAULT_SENSORS = ("amsua", "amsub", "atms", "mhs", "iasi", "cris", "airs")
_CYCLE_PATTERN = re.compile(
    r"(?P<day>\d{8}).*?[._]t(?P<hour>00|06|12|18)z(?:[._]|$)", re.IGNORECASE
)


def _json_header_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value


def _compact_header_column(values: Sequence[Any]) -> dict[str, Any]:
    if all(value == values[0] for value in values[1:]):
        return {"encoding": "constant", "value": values[0]}
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        step = values[1] - values[0]
        if all(value == values[0] + step * index for index, value in enumerate(values)):
            return {"encoding": "linear", "start": values[0], "step": step}
    runs: list[list[Any]] = []
    for value in values:
        if runs and runs[-1][1] == value:
            runs[-1][0] += 1
        else:
            runs.append([1, value])
    rle = {"encoding": "rle", "runs": runs}
    plain = {"encoding": "plain", "values": list(values)}
    encoded_rle = json.dumps(rle, sort_keys=True, separators=(",", ":"))
    encoded_plain = json.dumps(plain, sort_keys=True, separators=(",", ":"))
    return rle if len(encoded_rle) < len(encoded_plain) else plain


def _header_column(headers: Sequence[BufrMessageHeader], name: str) -> list[Any]:
    values = [getattr(header, name) for header in headers]
    if name in ("reference_time", "descriptors"):
        return [_json_header_value(value) for value in values]
    return values


def _message_header_manifest(
    headers: Sequence[BufrMessageHeader],
) -> dict[str, Any]:
    """Compact exact ordered Section 1/3 headers for one source object."""

    if not headers:
        raise ValueError("cannot build an empty message-header manifest")
    return {
        "format_version": 1,
        "count": len(headers),
        "columns": {
            name: _compact_header_column(_header_column(headers, name))
            for name in MESSAGE_HEADER_FIELDS
        },
    }


@dataclass(frozen=True)
class ArchiveSourceCycle:
    """One six-hour BUFR source cycle contributing to a daily archive."""

    path: Path
    cycle: datetime
    object_key: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class SensorDayTask:
    """Complete, picklable input contract for one sensor/day conversion."""

    sensor: str
    utc_day: date
    sources: tuple[ArchiveSourceCycle, ...]
    output_path: Path
    release: str
    decoder_revision: str
    zero_byte_source_paths: tuple[str, ...] = ()
    etl_revision: str = "working-tree"
    scratch_dir: Path | None = None
    overwrite: bool = False

    def validate(self) -> None:
        if self.sensor not in SENSOR_SPECS:
            raise ValueError(f"unsupported sensor {self.sensor!r}")
        if not 1 <= len(self.sources) <= 4:
            raise ValueError("one sensor-day task requires one to four source cycles")
        expected_cycles = tuple(
            datetime(
                self.utc_day.year,
                self.utc_day.month,
                self.utc_day.day,
                hour,
                tzinfo=timezone.utc,
            )
            for hour in (0, 6, 12, 18)
        )
        if any(source.cycle.tzinfo is None for source in self.sources):
            raise ValueError("source cycle timestamps must be timezone-aware")
        actual_cycles = tuple(
            source.cycle.astimezone(timezone.utc) for source in self.sources
        )
        actual_set = set(actual_cycles)
        expected_subset = tuple(
            cycle for cycle in expected_cycles if cycle in actual_set
        )
        if actual_cycles != expected_subset:
            raise ValueError(
                "source cycles must be a unique ordered subset of 00Z, 06Z, 12Z, "
                f"and 18Z for {self.utc_day.isoformat()}"
            )
        if not self.release:
            raise ValueError("release must not be empty")
        if not self.decoder_revision:
            raise ValueError("decoder_revision must not be empty")
        if not self.etl_revision:
            raise ValueError("etl_revision must not be empty")


@dataclass(frozen=True)
class SensorDayResult:
    """Small parent-process result for progress and audit logging."""

    sensor: str
    utc_day: str
    output_path: str
    rows: int
    row_groups: int
    output_bytes: int
    elapsed_seconds: float
    subsets_per_second: float
    source_bytes: int
    worker_peak_rss_mib: float
    windows: tuple[str, ...]
    contract: dict[str, Any] | None
    cpu_profile: str | None = None


@dataclass(frozen=True)
class DiscoveryIssue:
    """One raw-archive day that cannot form an unambiguous four-cycle task."""

    sensor: str
    utc_day: date
    message: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryResult:
    tasks: tuple[SensorDayTask, ...]
    issues: tuple[DiscoveryIssue, ...]


ProgressCallback = Callable[[SensorDayResult, int, int], None]


def _missing_cycle_labels(sources: Sequence[ArchiveSourceCycle]) -> tuple[str, ...]:
    present = {source.cycle.astimezone(timezone.utc).hour for source in sources}
    return tuple(f"{hour:02d}Z" for hour in (0, 6, 12, 18) if hour not in present)


def _selected_source_paths(
    sensor_root: Path,
    start_day: date | None,
    end_day: date | None,
) -> Iterable[Path]:
    """Return source candidates without scanning irrelevant months."""

    if start_day is None or end_day is None:
        yield from sensor_root.glob("*/*/bufr/*")
        return
    year = start_day.year
    month = start_day.month
    while (year, month) <= (end_day.year, end_day.month):
        yield from (sensor_root / f"{year:04d}" / f"{month:02d}" / "bufr").glob("*")
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def _parse_source_cycle(path: Path) -> datetime | None:
    match = _CYCLE_PATTERN.search(path.name)
    if match is None:
        return None
    day = datetime.strptime(match.group("day"), "%Y%m%d").date()
    return datetime(
        day.year,
        day.month,
        day.day,
        int(match.group("hour")),
        tzinfo=timezone.utc,
    )


def discover_sensor_day_tasks(
    raw_root: Path,
    output_root: Path,
    *,
    decoder_revision: str,
    etl_revision: str,
    sensors: Sequence[str] = DEFAULT_SENSORS,
    start_day: date | None = None,
    end_day: date | None = None,
    scratch_dir: Path | None = None,
    overwrite: bool = False,
) -> DiscoveryResult:
    """Scan the downloaded NNJA hierarchy and build complete sensor-day tasks."""

    selected = tuple(dict.fromkeys(sensors))
    unknown = set(selected) - SENSOR_SPECS.keys()
    if unknown:
        raise ValueError(f"unsupported sensors: {sorted(unknown)}")
    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError("start_day must not be after end_day")
    grouped: dict[tuple[str, date], dict[int, list[Path]]] = {}
    ignored_sources: dict[tuple[str, date], list[str]] = {}
    for sensor in selected:
        sensor_root = raw_root / sensor
        file_count = 0
        misplaced_count = 0
        for path in _selected_source_paths(sensor_root, start_day, end_day):
            if not path.is_file():
                continue
            file_count += 1
            cycle = _parse_source_cycle(path)
            if cycle is None:
                continue
            day = cycle.date()
            expected_parent = (
                sensor_root / f"{day.year:04d}" / f"{day.month:02d}" / "bufr"
            )
            if path.parent != expected_parent:
                misplaced_count += 1
                continue
            if start_day is not None and day < start_day:
                continue
            if end_day is not None and day > end_day:
                continue
            if path.stat().st_size == 0:
                grouped.setdefault((sensor, day), {})
                ignored_sources.setdefault((sensor, day), []).append(str(path))
                continue
            grouped.setdefault((sensor, day), {}).setdefault(cycle.hour, []).append(
                path
            )
        print(
            f"DISCOVERY_SENSOR sensor={sensor} files={file_count:,} "
            f"misplaced_ignored={misplaced_count:,}",
            flush=True,
        )
    if start_day is not None and end_day is not None:
        day = start_day
        while day <= end_day:
            for sensor in selected:
                grouped.setdefault((sensor, day), {})
            day += timedelta(days=1)

    tasks: list[SensorDayTask] = []
    issues: list[DiscoveryIssue] = []
    for (sensor, day), by_hour in sorted(grouped.items()):
        paths = tuple(
            str(path) for hour in sorted(by_hour) for path in sorted(by_hour[hour])
        ) + tuple(ignored_sources.get((sensor, day), ()))
        duplicates = {
            hour: len(by_hour.get(hour, ()))
            for hour in (0, 6, 12, 18)
            if len(by_hour.get(hour, ())) > 1
        }
        if not by_hour or duplicates:
            message = (
                (
                    "all source cycles are zero-byte"
                    if ignored_sources.get((sensor, day))
                    else "no source cycles found"
                )
                if not by_hour
                else f"duplicate source cycles={duplicates}"
            )
            issues.append(
                DiscoveryIssue(
                    sensor=sensor,
                    utc_day=day,
                    message=message,
                    source_paths=paths,
                )
            )
            continue
        output = output_root / sensor / f"{day:%Y%m%d}.parquet"
        sources = tuple(
            ArchiveSourceCycle(
                path=by_hour[hour][0],
                cycle=datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc),
                object_key=by_hour[hour][0].relative_to(raw_root).as_posix(),
                size_bytes=by_hour[hour][0].stat().st_size,
            )
            for hour in sorted(by_hour)
        )
        tasks.append(
            SensorDayTask(
                sensor=sensor,
                utc_day=day,
                sources=sources,
                output_path=output,
                release=f"{sensor}-nnja-native-v1",
                decoder_revision=decoder_revision,
                zero_byte_source_paths=tuple(ignored_sources.get((sensor, day), ())),
                etl_revision=etl_revision,
                scratch_dir=scratch_dir,
                overwrite=overwrite,
            )
        )
    tasks.sort(key=lambda task: (task.sensor, task.utc_day))
    return DiscoveryResult(tasks=tuple(tasks), issues=tuple(issues))


def _validate_day_output(task: SensorDayTask) -> dict[str, Any]:
    expected_sources = [
        {
            "object_key": source.object_key,
            "cycle": source.cycle.isoformat(),
            "size_bytes": (
                source.size_bytes
                if source.size_bytes is not None
                else source.path.stat().st_size
            ),
        }
        for source in task.sources
    ]
    return validate_parquet_contract(
        task.output_path,
        require_complete_day=True,
        expected_sensor=task.sensor,
        expected_day=task.utc_day,
        expected_sources=expected_sources,
    )


def _filter_existing_outputs(
    discovery: DiscoveryResult,
    output_root: Path,
    overwrite: bool,
    *,
    validate_existing: bool = True,
) -> DiscoveryResult:
    """Validate completed outputs and return pending source tasks."""

    if overwrite:
        return discovery
    pending = []
    issues = list(discovery.issues)
    rebuild_reasons: Counter[str] = Counter()
    for task in discovery.tasks:
        output = task.output_path
        if not output.exists():
            pending.append(task)
            continue
        if not validate_existing:
            clear_task_failure(output_root / "_errors", task)
            continue
        try:
            contract = _validate_day_output(task)
        except Exception as exc:
            message = str(exc)
            if "Expected all lists to be of size=" in message:
                reason = "unreadable_nested_columns"
            elif "empty-window provenance mismatch" in message:
                reason = "missing_empty_window_provenance"
            elif "source object identity differs" in message:
                reason = "source_set_changed"
            else:
                reason = "contract_validation"
            rebuild_reasons[reason] += 1
            pending.append(replace(task, overwrite=True))
        else:
            refresh_archive_warning(
                output_root,
                sensor=task.sensor,
                utc_day=task.utc_day,
                output_path=output,
                contract=contract,
            )
            clear_task_failure(output_root / "_errors", task)
    if rebuild_reasons:
        details = " ".join(
            f"{reason}={count:,}" for reason, count in sorted(rebuild_reasons.items())
        )
        print(
            f"OUTPUT_REBUILD_SUMMARY total={sum(rebuild_reasons.values()):,} "
            f"{details}",
            flush=True,
        )
    return DiscoveryResult(tasks=tuple(pending), issues=tuple(issues))


def _failure_path(error_root: Path, sensor: str, utc_day: date) -> Path:
    return error_root / sensor / f"{utc_day:%Y%m%d}.json"


def _warning_path(output_root: Path, sensor: str, utc_day: date) -> Path:
    return output_root / "_warnings" / sensor / f"{utc_day:%Y%m%d}.json"


def _issue_type(message: str, *, stage: str) -> str:
    if message.startswith("duplicate source cycles"):
        return "duplicate_source_cycles"
    if message == "no source cycles found":
        return "no_source_cycles"
    if "No BUFR messages found" in message:
        return "no_bufr_messages"
    if "source observation time" in message or "reference time" in message:
        return "invalid_source_time"
    if "source structure" in message or "contract mnemonics" in message:
        return "unsupported_source_structure"
    return f"{stage}_error"


def write_partial_day_warning(output_root: Path, task: SensorDayTask) -> Path:
    """Record one searchable non-fatal partial-source warning."""

    path = _warning_path(output_root, task.sensor, task.utc_day)
    missing = _missing_cycle_labels(task.sources)
    issue_types = ["missing_source_cycles"]
    if task.zero_byte_source_paths:
        issue_types.append("zero_byte_source_objects")
    write_json_atomic(
        path,
        {
            "format_version": 1,
            "status": "warning",
            "stage": "discovery",
            "sensor": task.sensor,
            "utc_day": task.utc_day.isoformat(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "message": f"partial source day; missing cycles={list(missing)}",
            "issue_types": issue_types,
            "available_cycles": [
                f"{source.cycle.astimezone(timezone.utc).hour:02d}Z"
                for source in task.sources
            ],
            "missing_cycles": list(missing),
            "source_paths": [str(source.path) for source in task.sources],
            "zero_byte_sources": [
                {"path": path, "size_bytes": 0} for path in task.zero_byte_source_paths
            ],
        },
    )
    return path


def refresh_archive_warning(
    output_root: Path,
    *,
    sensor: str,
    utc_day: date,
    output_path: Path,
    contract: Mapping[str, Any],
) -> Path | None:
    """Synchronize one warning record with validated archive provenance."""

    path = _warning_path(output_root, sensor, utc_day)
    missing = list(contract.get("missing_cycles", ()))
    empty = list(contract.get("empty_windows", ()))
    missing_template_fields = list(contract.get("missing_template_fields", ()))
    unusable_source_objects = list(contract.get("unusable_source_objects", ()))
    quarantined_source_messages = list(contract.get("quarantined_source_messages", ()))
    if (
        not missing
        and not empty
        and not missing_template_fields
        and not unusable_source_objects
        and not quarantined_source_messages
    ):
        path.unlink(missing_ok=True)
        return None
    zero_byte_sources = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            zero_byte_sources = existing.get("zero_byte_sources", ())
            if not zero_byte_sources:
                zero_byte_sources = [
                    {"path": value, "size_bytes": 0}
                    for value in existing.get("ignored_source_paths", ())
                ]
        except (OSError, json.JSONDecodeError):
            pass
    issue_types = []
    if missing:
        issue_types.append("missing_source_cycles")
    if empty:
        issue_types.append("empty_da_windows")
    if zero_byte_sources:
        issue_types.append("zero_byte_source_objects")
    if missing_template_fields:
        issue_types.append("missing_template_fields")
    if unusable_source_objects:
        issue_types.extend(
            sorted({str(source["reason"]) for source in unusable_source_objects})
        )
    if quarantined_source_messages:
        issue_types.append("truncated_source_messages")
    write_json_atomic(
        path,
        {
            "format_version": 1,
            "status": "warning",
            "stage": "validated_output",
            "sensor": sensor,
            "utc_day": utc_day.isoformat(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "message": (
                f"archive has missing_cycles={missing}, empty_windows={empty}, "
                f"missing_template_fields={missing_template_fields}, "
                f"unusable_source_objects={len(unusable_source_objects)}, "
                f"quarantined_source_messages={len(quarantined_source_messages)}"
            ),
            "issue_types": issue_types,
            "missing_cycles": missing,
            "empty_windows": empty,
            "missing_template_fields": missing_template_fields,
            "missing_template_fields_by_source": contract.get(
                "missing_template_fields_by_source", {}
            ),
            "unusable_source_objects": unusable_source_objects,
            "quarantined_source_messages": quarantined_source_messages,
            "output_path": str(output_path),
            "source_files": [
                source["object_key"] for source in contract.get("source_files", ())
            ],
            "zero_byte_sources": zero_byte_sources,
        },
    )
    return path


def write_task_index(
    path: Path,
    discovery: DiscoveryResult,
    *,
    raw_root: Path,
    sensors: Sequence[str],
) -> None:
    """Atomically persist one immutable source task snapshot."""

    records: list[dict[str, Any]] = [
        {
            "record_type": "metadata",
            "format_version": 1,
            "raw_root": str(raw_root.resolve()),
            "sensors": list(sensors),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    ]
    for task in discovery.tasks:
        records.append(
            {
                "record_type": "task",
                "sensor": task.sensor,
                "utc_day": task.utc_day.isoformat(),
                "zero_byte_source_paths": list(task.zero_byte_source_paths),
                "sources": [
                    {
                        "object_key": source.object_key,
                        "cycle": source.cycle.isoformat(),
                        "size_bytes": source.size_bytes,
                    }
                    for source in task.sources
                ],
            }
        )
    for issue in discovery.issues:
        records.append(
            {
                "record_type": "issue",
                "sensor": issue.sensor,
                "utc_day": issue.utc_day.isoformat(),
                "message": issue.message,
                "source_paths": list(issue.source_paths),
            }
        )
    write_text_atomic(
        path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )


def load_task_index(
    path: Path,
    *,
    raw_root: Path,
    output_root: Path,
    sensors: Sequence[str],
    start_day: date | None,
    end_day: date | None,
    task_options: dict[str, Any],
) -> DiscoveryResult:
    """Load source tasks and apply current runtime/output configuration."""

    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records or records[0].get("record_type") != "metadata":
        raise ValueError(f"invalid task index: {path}")
    metadata = records[0]
    if metadata.get("format_version") != 1:
        raise ValueError(f"unsupported task index format: {path}")
    if Path(metadata["raw_root"]).resolve() != raw_root.resolve():
        raise ValueError("task index raw root differs; rerun with --reindex")
    indexed_sensors = set(metadata["sensors"])
    if not set(sensors).issubset(indexed_sensors):
        raise ValueError(
            "task index is missing requested sensors; rerun with --reindex"
        )

    tasks = []
    issues = []
    selected = set(sensors)
    for record in records[1:]:
        sensor = str(record["sensor"])
        utc_day = date.fromisoformat(record["utc_day"])
        if sensor not in selected:
            continue
        if start_day is not None and utc_day < start_day:
            continue
        if end_day is not None and utc_day > end_day:
            continue
        if record["record_type"] == "issue":
            source_paths = tuple(str(value) for value in record["source_paths"])
            message = str(record["message"])
            candidates = tuple(
                Path(value) if Path(value).is_absolute() else raw_root / value
                for value in source_paths
            )
            if (
                message == "no source cycles found"
                and candidates
                and all(
                    candidate.is_file() and candidate.stat().st_size == 0
                    for candidate in candidates
                )
            ):
                message = "all source cycles are zero-byte"
            issues.append(
                DiscoveryIssue(
                    sensor=sensor,
                    utc_day=utc_day,
                    message=message,
                    source_paths=source_paths,
                )
            )
            continue
        sources = tuple(
            ArchiveSourceCycle(
                path=raw_root / source["object_key"],
                cycle=datetime.fromisoformat(source["cycle"]),
                object_key=str(source["object_key"]),
                size_bytes=int(source["size_bytes"]),
            )
            for source in record["sources"]
        )
        tasks.append(
            SensorDayTask(
                sensor=sensor,
                utc_day=utc_day,
                sources=sources,
                output_path=output_root / sensor / f"{utc_day:%Y%m%d}.parquet",
                release=f"{sensor}-nnja-native-v1",
                zero_byte_source_paths=tuple(
                    str(value)
                    for value in record.get(
                        "zero_byte_source_paths",
                        record.get("ignored_source_paths", ()),
                    )
                ),
                **task_options,
            )
        )
    tasks.sort(key=lambda task: (task.sensor, task.utc_day))
    return DiscoveryResult(tasks=tuple(tasks), issues=tuple(issues))


def write_discovery_issue(error_root: Path, issue: DiscoveryIssue) -> Path:
    path = _failure_path(error_root, issue.sensor, issue.utc_day)
    write_json_atomic(
        path,
        {
            "format_version": 1,
            "status": "failed",
            "stage": "discovery",
            "issue_type": _issue_type(issue.message, stage="discovery"),
            "sensor": issue.sensor,
            "utc_day": issue.utc_day.isoformat(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "message": issue.message,
            "source_paths": list(issue.source_paths),
        },
    )
    return path


def write_unavailable_day_warning(output_root: Path, issue: DiscoveryIssue) -> Path:
    path = _warning_path(output_root, issue.sensor, issue.utc_day)
    write_json_atomic(
        path,
        {
            "format_version": 1,
            "status": "warning",
            "stage": "discovery",
            "sensor": issue.sensor,
            "utc_day": issue.utc_day.isoformat(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "message": issue.message,
            "issue_types": [
                "missing_source_cycles",
                "zero_byte_source_objects",
            ],
            "available_cycles": [],
            "missing_cycles": ["00Z", "06Z", "12Z", "18Z"],
            "source_paths": list(issue.source_paths),
            "zero_byte_sources": [
                {"path": source, "size_bytes": 0} for source in issue.source_paths
            ],
        },
    )
    return path


def write_task_failure(
    error_root: Path, task: SensorDayTask, exception: BaseException
) -> Path:
    path = _failure_path(error_root, task.sensor, task.utc_day)
    write_json_atomic(
        path,
        {
            "format_version": 1,
            "status": "failed",
            "stage": "conversion",
            "sensor": task.sensor,
            "utc_day": task.utc_day.isoformat(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "parent_pid": os.getpid(),
            "output_path": str(task.output_path),
            "source_paths": [str(source.path) for source in task.sources],
            "source_cycles": [source.cycle.isoformat() for source in task.sources],
            "decoder_revision": task.decoder_revision,
            "etl_revision": task.etl_revision,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "issue_type": _issue_type(str(exception), stage="conversion"),
            "traceback": exception_traceback(exception),
        },
    )
    return path


def clear_task_failure(error_root: Path, task: SensorDayTask) -> None:
    _failure_path(error_root, task.sensor, task.utc_day).unlink(missing_ok=True)


def _set_torch_threads(count: int) -> None:
    import torch  # noqa: PLC0415

    torch.set_num_threads(count)


def initialize_archive_worker() -> None:
    """Set the worker's thread budgets and warm the HEALPix lookup."""

    earth2bufr.set_num_threads(DECODE_THREADS)
    _set_torch_threads(TORCH_THREADS)
    pa.set_cpu_count(ARROW_THREADS)
    initialize_hpx2048()


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_provenance(
    task: SensorDayTask,
    source_files: Sequence[SourceFile],
    structures: Sequence[SourceStructure],
    source_subsets: int,
    populated_windows: Sequence[datetime],
    unusable_source_objects: Sequence[Mapping[str, Any]],
    quarantined_source_messages: Sequence[Mapping[str, Any]],
) -> FileProvenance:
    unique = {structure.fingerprint: structure for structure in structures}
    ordered_structures = tuple(unique[key] for key in sorted(unique))
    populated = {window.astimezone(timezone.utc) for window in populated_windows}
    available_cycles = tuple(
        datetime.fromisoformat(source.cycle).astimezone(timezone.utc)
        for source in source_files
    )
    expected_windows = tuple(
        cycle + offset
        for cycle in available_cycles
        for offset in (timedelta(0), timedelta(hours=3))
    )
    return FileProvenance(
        utc_day=task.utc_day,
        source_files=tuple(source_files),
        source_structures=ordered_structures,
        decoder_name=DECODER_NAME,
        decoder_version=earth2bufr.__version__,
        decoder_revision=task.decoder_revision,
        etl_revision=task.etl_revision,
        effective_packing_ledger_sha256=_hash_payload(
            [
                {
                    "source_structure_sha256": structure.fingerprint,
                    "effective_occurrences": list(structure.effective_occurrences),
                }
                for structure in ordered_structures
            ]
        ),
        code_flag_catalog_sha256=sha256_file(CATALOG_PATH)[0],
        source_subsets_encountered=source_subsets,
        quarantined_subsets=sum(
            int(record["subset_count"]) for record in quarantined_source_messages
        ),
        quarantine_manifest_sha256=(
            _hash_payload(list(quarantined_source_messages))
            if quarantined_source_messages
            else None
        ),
        quarantined_source_messages=tuple(quarantined_source_messages),
        missing_cycles=tuple(
            f"{hour:02d}Z"
            for hour in (0, 6, 12, 18)
            if hour not in {cycle.hour for cycle in available_cycles}
        ),
        empty_windows=tuple(
            window.isoformat() for window in expected_windows if window not in populated
        ),
        unusable_source_objects=tuple(unusable_source_objects),
    )


def _hash_source(path: Path) -> tuple[str, int]:
    with cpu_timing_range("source.sha256"):
        return sha256_file(path)


def _write_sensor_day(task: SensorDayTask) -> WriteSummary:
    """Derive each structure from its open source and publish one faithful day."""

    writer: SounderArchiveDayWriter | None = None
    writer_layout = None
    compatible_structure_hashes: set[str] = set()
    source_files: list[SourceFile] = []
    structures: list[SourceStructure] = []
    unusable_source_objects: list[dict[str, Any]] = []
    quarantined_source_messages: list[dict[str, Any]] = []
    source_subsets = 0
    # Hashes each source while it decodes.
    hash_pool = ThreadPoolExecutor(max_workers=1)
    try:
        for source in task.sources:
            if not source.path.is_file():
                raise FileNotFoundError(source.path)
            object_key = source.object_key or source.path.name
            source_hash = hash_pool.submit(_hash_source, source.path)
            with open_source(source.path) as opened:
                with cpu_timing_range("source.headers_structure"):
                    try:
                        headers = opened.message_headers
                    except ValueError as exc:
                        if str(exc) != "source contains no meaningful data messages":
                            raise
                        digest, size_bytes = source_hash.result()
                        unusable_source_objects.append(
                            {
                                "object_key": object_key,
                                "cycle": source.cycle.isoformat(),
                                "size_bytes": size_bytes,
                                "sha256": digest,
                                "reason": "no_observation_subsets",
                            }
                        )
                        continue
                    source_file_ordinal = len(source_files)
                    bindings = opened.source_structure_bindings(task.sensor)
                    file_structures = tuple(
                        {
                            structure.fingerprint: structure
                            for structure in bindings.values()
                        }.values()
                    )
                structures.extend(file_structures)
                source_subsets += sum(header.subset_count for header in headers)
                converted_any = False
                for converted in iter_open_sounder_archive_batches(
                    opened,
                    sensor=task.sensor,
                    cycle=source.cycle,
                    source_file_ordinal=source_file_ordinal,
                    release=task.release,
                    structures=bindings,
                ):
                    converted_any = True
                    if writer is None:
                        writer_layout = converted.layout
                        writer = SounderArchiveDayWriter(
                            task.output_path,
                            sensor=task.sensor,
                            schema=converted.layout.schema,
                            scratch_dir=task.scratch_dir,
                            require_complete_day=True,
                        )
                        compatible_structure_hashes.add(
                            converted.layout.source_structure.fingerprint
                        )
                    else:
                        fingerprint = converted.layout.source_structure.fingerprint
                        if fingerprint not in compatible_structure_hashes:
                            validate_archive_layout_compatibility(
                                writer_layout, converted.layout
                            )
                            compatible_structure_hashes.add(fingerprint)
                    writer.write(converted.table)
                for failure in opened.decode_failures():
                    quarantined_source_messages.append(
                        {
                            "object_key": object_key,
                            "cycle": source.cycle.isoformat(),
                            "source_file_ordinal": source_file_ordinal,
                            **failure,
                        }
                    )
                if not converted_any:
                    raise RuntimeError(
                        f"decoder produced no {task.sensor} rows for {source.path}"
                    )
            with cpu_timing_range("source.sha256_wait"):
                digest, size_bytes = source_hash.result()
            structure_hashes = tuple(
                structure.fingerprint for structure in file_structures
            )
            source_files.append(
                SourceFile(
                    object_key=object_key,
                    sha256=digest,
                    size_bytes=size_bytes,
                    cycle=source.cycle.isoformat(),
                    message_type=task.sensor,
                    source_structure_sha256=(
                        structure_hashes[0] if len(structure_hashes) == 1 else None
                    ),
                    message_header_manifest=_message_header_manifest(headers),
                    source_structure_sha256s=(
                        structure_hashes if len(structure_hashes) > 1 else ()
                    ),
                )
            )
            writer.finish_windows((source.cycle, source.cycle + timedelta(hours=3)))
        if writer is None:
            raise RuntimeError(f"decoder produced no {task.sensor} rows")
        with cpu_timing_range("writer.bind_provenance"):
            provenance = _file_provenance(
                task,
                source_files,
                structures,
                source_subsets,
                writer.populated_windows,
                unusable_source_objects,
                quarantined_source_messages,
            )
            writer.bind_provenance(provenance)
        return writer.close()
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    finally:
        hash_pool.shutdown(wait=True, cancel_futures=True)


def process_sensor_day(task: SensorDayTask) -> SensorDayResult:
    """Convert one to four source cycles into one atomic daily Parquet archive."""

    task.validate()
    if task.output_path.exists() and not task.overwrite:
        raise FileExistsError(task.output_path)
    if CPU_TIMING_ENABLED:
        cpu_timing_reset()
    started = time.perf_counter()
    summary = _write_sensor_day(task)
    elapsed = time.perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    cpu_profile = cpu_timing_report(None) if CPU_TIMING_ENABLED else None
    if cpu_profile is not None:
        print(
            "CPU_PROFILE "
            + json.dumps(
                {
                    "sensor": task.sensor,
                    "utc_day": task.utc_day.isoformat(),
                    "elapsed_seconds": elapsed,
                    "stages": cpu_profile,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return SensorDayResult(
        sensor=task.sensor,
        utc_day=task.utc_day.isoformat(),
        output_path=str(task.output_path),
        rows=summary.rows,
        row_groups=summary.row_groups,
        output_bytes=task.output_path.stat().st_size,
        elapsed_seconds=elapsed,
        subsets_per_second=summary.rows / elapsed,
        source_bytes=sum(source.path.stat().st_size for source in task.sources),
        worker_peak_rss_mib=peak_rss,
        windows=summary.windows,
        contract=summary.contract,
        cpu_profile=cpu_profile,
    )


def recover_sensor_day(task: SensorDayTask) -> SensorDayResult | None:
    """Validate a day published just before its worker process exited."""

    if task.overwrite:
        return None
    contract = _validate_day_output(task)
    return SensorDayResult(
        sensor=task.sensor,
        utc_day=task.utc_day.isoformat(),
        output_path=str(task.output_path),
        rows=contract["rows"],
        row_groups=contract["row_groups"],
        output_bytes=task.output_path.stat().st_size,
        elapsed_seconds=0.0,
        subsets_per_second=0.0,
        source_bytes=sum(source.path.stat().st_size for source in task.sources),
        worker_peak_rss_mib=0.0,
        windows=tuple(contract["windows"]),
        contract=contract,
    )


def _format_elapsed(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def report_sensor_day_progress(
    result: SensorDayResult, completed: int, total: int
) -> None:
    """Print one immediately flushed completion record in the parent process."""

    mib = result.output_bytes / 2**20
    print(
        f"[{completed}/{total}] {result.sensor} {result.utc_day}: "
        f"{result.rows:,} rows, {mib:,.1f} MiB, "
        f"{result.elapsed_seconds:,.1f} s, "
        f"{result.subsets_per_second:,.0f} rows/s, "
        f"peak_rss={result.worker_peak_rss_mib:,.1f} MiB -> {result.output_path}",
        flush=True,
    )


def run_sensor_days(
    tasks: Sequence[SensorDayTask],
    *,
    max_workers: int,
    progress: ProgressCallback | None = report_sensor_day_progress,
    error_root: Path | None = None,
    continue_on_error: bool = False,
) -> tuple[SensorDayResult, ...]:
    """Run one sensor-day per task on the shared spawn pool."""

    return run_tasks(
        tasks,
        process=process_sensor_day,
        max_workers=max_workers,
        initializer=initialize_archive_worker,
        progress=progress,
        describe=lambda task: f"{task.sensor} {task.utc_day.isoformat()}",
        error_root=error_root,
        continue_on_error=continue_on_error,
        on_failure=write_task_failure,
        on_success=clear_task_failure,
        recover_completed=recover_sensor_day,
    )


def _parse_day(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert downloaded NNJA satellite BUFR into daily Parquet"
    )
    parser.add_argument("--raw-root", type=Path, help="Default: $NNJA_ROOT/bufr")
    parser.add_argument("--output-root", type=Path, help="Default: $NNJA_ROOT/parquet")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--sensors", nargs="+", default=list(DEFAULT_SENSORS))
    parser.add_argument("--start-day", type=_parse_day)
    parser.add_argument("--end-day", type=_parse_day)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--decoder-revision",
        default=os.environ.get("EARTH2BUFR_REVISION"),
        required="EARTH2BUFR_REVISION" not in os.environ,
    )
    parser.add_argument(
        "--etl-revision", default=os.environ.get("HEALDA_ETL_REVISION", "working-tree")
    )
    parser.add_argument("--progress-interval", type=float, default=60.0)
    # Task index: one immutable source snapshot per sensor, so full-archive runs do not
    # rescan the raw tree; --reindex --index-only builds it.
    parser.add_argument("--task-index", type=Path)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.raw_root is None or args.output_root is None:
        if not NNJA_ROOT:
            parser.error("set NNJA_ROOT or pass --raw-root and --output-root")
        args.raw_root = args.raw_root or Path(nnja_archive("bufr"))
        args.output_root = args.output_root or Path(nnja_archive("parquet"))
    if args.start_day and args.end_day and args.start_day > args.end_day:
        parser.error("--start-day must not be after --end-day")
    if args.reindex and args.task_index is None:
        parser.error("--reindex requires --task-index")
    if args.reindex and (args.start_day or args.end_day):
        parser.error(
            "--reindex rebuilds the whole task index; drop --start-day/--end-day"
        )
    if args.index_only and not args.reindex:
        parser.error("--index-only requires --reindex")
    task_options = {
        "decoder_revision": args.decoder_revision,
        "etl_revision": args.etl_revision,
        "scratch_dir": args.scratch_dir,
        "overwrite": args.overwrite,
    }
    scan_started = time.perf_counter()
    print(
        f"DISCOVERY_START sensors={','.join(args.sensors)} "
        f"start={args.start_day} end={args.end_day} raw_root={args.raw_root}",
        flush=True,
    )
    reuse_task_index = (
        args.task_index is not None and args.task_index.exists() and not args.reindex
    )
    if reuse_task_index:
        print(f"TASK_INDEX_LOAD path={args.task_index}", flush=True)
        source_discovery = load_task_index(
            args.task_index,
            raw_root=args.raw_root,
            output_root=args.output_root,
            sensors=args.sensors,
            start_day=args.start_day,
            end_day=args.end_day,
            task_options=task_options,
        )
    else:
        source_discovery = discover_sensor_day_tasks(
            args.raw_root,
            args.output_root,
            sensors=args.sensors,
            start_day=args.start_day,
            end_day=args.end_day,
            **task_options,
        )
        # A range-limited scan would truncate the index to that range.
        if args.task_index is not None and not (args.start_day or args.end_day):
            write_task_index(
                args.task_index,
                source_discovery,
                raw_root=args.raw_root,
                sensors=args.sensors,
            )
            print(
                f"TASK_INDEX_WRITE path={args.task_index} "
                f"tasks={len(source_discovery.tasks):,} "
                f"issues={len(source_discovery.issues):,}",
                flush=True,
            )
    if args.index_only:
        print(
            f"INDEX_ONLY_DONE tasks={len(source_discovery.tasks):,} "
            f"issues={len(source_discovery.issues):,}",
            flush=True,
        )
        return 0
    discovery = _filter_existing_outputs(
        source_discovery,
        args.output_root,
        args.overwrite,
        validate_existing=args.audit_existing,
    )
    # Published days keep the warnings refreshed from their validated output.
    partial_days = 0
    for task in discovery.tasks:
        if _missing_cycle_labels(task.sources):
            write_partial_day_warning(args.output_root, task)
            partial_days += 1
        else:
            _warning_path(args.output_root, task.sensor, task.utc_day).unlink(
                missing_ok=True
            )
    if partial_days:
        print(
            f"DISCOVERY_WARNINGS partial_days={partial_days:,} "
            f"path={args.output_root / '_warnings'}",
            flush=True,
        )
    print(
        f"DISCOVERY_DONE tasks={len(discovery.tasks):,} "
        f"issues={len(discovery.issues):,} "
        f"elapsed={time.perf_counter() - scan_started:,.1f}s",
        flush=True,
    )
    error_root = args.output_root / "_errors"
    unavailable_issues = [
        issue
        for issue in discovery.issues
        if issue.message == "all source cycles are zero-byte"
    ]
    fatal_issues = [
        issue
        for issue in discovery.issues
        if issue.message != "all source cycles are zero-byte"
    ]
    for issue in unavailable_issues:
        path = write_unavailable_day_warning(args.output_root, issue)
        _failure_path(error_root, issue.sensor, issue.utc_day).unlink(missing_ok=True)
        print(
            f"DISCOVERY WARNING {issue.sensor} {issue.utc_day.isoformat()} -> {path}",
            flush=True,
        )
    for issue in fatal_issues:
        path = write_discovery_issue(error_root, issue)
        print(
            f"DISCOVERY FAILED {issue.sensor} {issue.utc_day.isoformat()} -> {path}",
            flush=True,
        )
    print(
        f"Discovered {len(discovery.tasks):,} pending sensor-days, "
        f"{partial_days + len(unavailable_issues):,} source warnings, and "
        f"{len(fatal_issues):,} ambiguous/invalid days",
        flush=True,
    )

    totals_by_sensor = Counter(task.sensor for task in discovery.tasks)
    completed_by_sensor: Counter[str] = Counter()
    last_progress = 0.0
    run_started = time.monotonic()
    successful_rows = 0
    successful_files = 0

    def progress(result: SensorDayResult, completed: int, total: int) -> None:
        nonlocal last_progress, successful_rows, successful_files
        completed_by_sensor[result.sensor] += 1
        successful_rows += result.rows
        successful_files += len((result.contract or {}).get("source_files", ()))
        if result.contract is not None:
            refresh_archive_warning(
                args.output_root,
                sensor=result.sensor,
                utc_day=date.fromisoformat(result.utc_day),
                output_path=Path(result.output_path),
                contract=result.contract,
            )
        sensor_completed = completed_by_sensor[result.sensor]
        sensor_total = totals_by_sensor[result.sensor]
        now = time.monotonic()
        if (
            completed == 1
            or now - last_progress >= args.progress_interval
            or sensor_completed == sensor_total
        ):
            elapsed = now - run_started
            successful = sum(completed_by_sensor.values())
            failed = completed - successful
            task_rate = completed / elapsed
            file_rate = successful_files / elapsed
            row_rate = successful_rows / elapsed
            eta = (total - completed) / task_rate
            print(
                f"[{completed:,}/{total:,}] {result.sensor} "
                f"done={sensor_completed:,}/{sensor_total:,} failed={failed:,} "
                f"elapsed={_format_elapsed(elapsed)} "
                f"rate={task_rate:,.2f} tasks/s {file_rate:,.2f} files/s "
                f"{row_rate:,.0f} rows/s eta={_format_elapsed(eta)} "
                f"latest={result.utc_day} "
                f"task={result.elapsed_seconds:,.1f}s",
                flush=True,
            )
            last_progress = now

    results = run_sensor_days(
        discovery.tasks,
        max_workers=args.workers,
        progress=progress,
        error_root=error_root,
        continue_on_error=True,
    )
    failures = len(fatal_issues) + len(discovery.tasks) - len(results)
    elapsed = time.monotonic() - run_started
    completed_files = sum(
        len((result.contract or {}).get("source_files", ())) for result in results
    )
    file_rate = completed_files / elapsed if elapsed else 0.0
    row_rate = sum(result.rows for result in results) / elapsed if elapsed else 0.0
    print(
        f"Completed {len(results):,}/{len(discovery.tasks):,} submitted sensor-days; "
        f"failures={failures:,}; elapsed={_format_elapsed(elapsed)}; "
        f"throughput={file_rate:,.2f} files/s {row_rate:,.0f} rows/s",
        flush=True,
    )
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
