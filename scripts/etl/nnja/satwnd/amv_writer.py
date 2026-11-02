# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical Parquet writer for the NNJA SATWND (AMV) archive contract.

Spooling, one row group per DA window and atomic publication come from the
shared ``archive_common`` day writer. ``AmvArchiveDayWriter`` supplies the
AMV contract validator and footer provenance.
"""

from __future__ import annotations

import json
import os
import resource

import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import earth2bufr
import pyarrow as pa
import pyarrow.parquet as pq

from ..archive_common.archive_runner import (
    exception_traceback,
    run_tasks,
    write_json_atomic,
)
from ..archive_common.hashing import sha256_file
from ..archive_common.archive_writer import (
    ArchiveDayWriter as _SharedArchiveDayWriter,
    _schema_without_metadata,
)
from .archive_etl import (
    CATALOG_PATH,
    DECLARED_SCHEMA_PATH,
    iter_projected_batches,
    load_declared_schema,
    read_source_structures,
    DERIVED_COLUMNS,
    GSI_TYPE_TABLE_PATH,
    WIND_SLOT_GATE_PATH,
    SCHEMA_VERSION,
    footer_source_structures,
    etl_git_sha,
    primary_stream_for_year,
)

ROW_GROUP_POLICY = "one_complete_da_window"

# One stream is read per era, so no rows are deduplicated across streams.
DEDUP_FOOTER = {
    "amv.dedup_shared_positions": "0",
    "amv.dedup_identical_rows": "0",
    "amv.dedup_conflicting_rows": "0",
    "amv.dedup_dropped_rows": "0",
}


def _default_scratch_dir() -> Path | None:
    """Spool directory for per-window IPC files: SLURM_TMPDIR, then /dev/shm."""
    for candidate in (os.environ.get("SLURM_TMPDIR"), "/dev/shm"):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


@dataclass
class AmvFileProvenance:
    """Per-file provenance for one AMV archive day; the shared writer calls ``validate()``."""

    utc_day: datetime
    decoder_version: str
    # Exact commit, provenance only; not compared on resume.
    etl_git_sha: str
    source_files: tuple[Mapping[str, Any], ...] = ()
    source_structures: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    decoder_revision: str = ""
    etl_revision: str = ""
    schema_version: str = SCHEMA_VERSION
    created_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self) -> None:
        if not self.source_files:
            raise ValueError("source_files must not be empty")
        if not self.decoder_revision:
            raise ValueError("decoder_revision must not be empty")
        if not self.etl_revision:
            raise ValueError("etl_revision must not be empty")

    def as_footer_metadata(self) -> dict[str, str]:
        return {
            "amv.schema_version": self.schema_version,
            "amv.decoder_version": self.decoder_version,
            "amv.decoder_revision": self.decoder_revision,
            "amv.etl_revision": self.etl_revision,
            "amv.etl_git_sha": self.etl_git_sha,
            "amv.declared_schema_sha256": sha256_file(str(DECLARED_SCHEMA_PATH))[0],
            "amv.build_timestamp_utc": self.created_utc.isoformat(),
            "amv.row_group_policy": ROW_GROUP_POLICY,
            "amv.utc_day": self.utc_day.strftime("%Y-%m-%d"),
            "amv.source_cycles": json.dumps(list(self.source_files)),
            **DEDUP_FOOTER,
            "amv.data_attribution": DATA_ATTRIBUTION,
            # Descriptors each template carries, from the file's own DX tables.
            "amv.source_structures": json.dumps(
                footer_source_structures(self.source_structures)
            ),
            # Pins the catalogue that per-field metadata came from.
            "amv.descriptor_catalog": CATALOG_PATH.name,
            "amv.descriptor_catalog_sha256": sha256_file(str(CATALOG_PATH))[0],
            "amv.derived_columns": json.dumps(DERIVED_COLUMNS),
            "amv.gsi_type_table": GSI_TYPE_TABLE_PATH.name,
            "amv.gsi_type_table_sha256": sha256_file(str(GSI_TYPE_TABLE_PATH))[0],
            "amv.wind_slot_gate": WIND_SLOT_GATE_PATH.name,
            "amv.wind_slot_gate_sha256": sha256_file(str(WIND_SLOT_GATE_PATH))[0],
        }


DATA_ATTRIBUTION = (
    "Contains EUMETSAT Atmospheric Motion Vectors Climate Data Record Release 2 "
    "(MFG and MSG, 0 degree), EUMETSAT (2021), "
    "DOI: 10.15770/EUM_SEC_CLM_0020, licensed CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/). "
    "Also contains NCEP operational satwnd BUFR dumps."
)

REQUIRED_FOOTER_KEYS = (
    b"amv.schema_version",
    b"amv.decoder_version",
    b"amv.decoder_revision",
    b"amv.etl_revision",
    b"amv.declared_schema_sha256",
    b"amv.source_cycles",
    b"amv.source_structures",
    b"amv.descriptor_catalog_sha256",
    b"amv.derived_columns",
    b"amv.gsi_type_table_sha256",
    b"amv.row_group_policy",
    b"amv.utc_day",
)


def bind_amv_file_metadata(
    schema: pa.Schema, provenance: AmvFileProvenance
) -> pa.Schema:
    provenance.validate()
    existing = dict(schema.metadata or {})
    existing.update(
        {
            key.encode(): value.encode()
            for key, value in provenance.as_footer_metadata().items()
        }
    )
    return schema.with_metadata(existing)


def validate_amv_parquet_contract(path: Path) -> dict[str, Any]:
    """Reopen a written AMV day file and check it is publishable.

    Checks required footer keys and one ``da_window`` per row group in strictly
    increasing order, from the footer alone.
    """
    parquet = pq.ParquetFile(str(path))
    # The file-level footer: every key bound at open plus those added at close.
    metadata = parquet.metadata.metadata or {}
    missing = [key for key in REQUIRED_FOOTER_KEYS if key not in metadata]
    if missing:
        raise AssertionError(
            f"missing required footer metadata: {sorted(k.decode() for k in missing)}"
        )

    da_index = parquet.schema_arrow.get_field_index("da_window")
    if da_index < 0:
        raise AssertionError("archive is missing the da_window column")

    windows: list[Any] = []
    for group_index in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(group_index).column(da_index).statistics
        if statistics is None:
            raise AssertionError(f"row group {group_index} has no da_window statistics")
        if statistics.min != statistics.max:
            raise AssertionError(f"row group {group_index} crosses da_window")
        windows.append(statistics.min)

    if len(set(windows)) != len(windows):
        raise AssertionError("more than one row group represents a da_window")
    if windows != sorted(windows):
        raise AssertionError("row groups are not in chronological da_window order")

    return {
        "path": str(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "row_group_policy": ROW_GROUP_POLICY,
        "windows": [str(value) for value in windows],
        "schema_version": metadata[b"amv.schema_version"].decode(),
        "decoder_version": metadata[b"amv.decoder_version"].decode(),
        "decoder_revision": metadata[b"amv.decoder_revision"].decode(),
        "etl_revision": metadata[b"amv.etl_revision"].decode(),
        "declared_schema_sha256": metadata[b"amv.declared_schema_sha256"].decode(),
        "source_cycles": json.loads(metadata[b"amv.source_cycles"]),
    }


class AmvArchiveDayWriter(_SharedArchiveDayWriter):
    """Shared day writer with the AMV contract's hooks bound."""

    def bind_metadata(
        self, schema: pa.Schema, provenance: AmvFileProvenance
    ) -> pa.Schema:
        return bind_amv_file_metadata(schema, provenance)

    def validate_contract(self, path: Path) -> dict[str, Any]:
        return validate_amv_parquet_contract(path)


def write_day(
    day: datetime,
    cycle_paths: dict[str, str],
    output_root: str,
    sensor: str = "satwnd",
    *,
    decoder_revision: str,
    etl_revision: str,
) -> dict:
    """Decode one UTC day's four synoptic cycles into a single Parquet file.

    Writes ``<output_root>/<sensor>/<YYYYMMDD>.parquet`` with one row group per
    3h ``da_window``.
    """
    if not decoder_revision:
        raise ValueError("decoder_revision must not be empty")
    if not etl_revision:
        raise ValueError("etl_revision must not be empty")
    out_dir = Path(output_root) / sensor
    out_path = out_dir / f"{day:%Y%m%d}.parquet"

    per_cycle_stats: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    source_structures: dict[str, Any] = {}

    # Pass 1: source identity and per-template structure, without decoding data.
    cycles: list[tuple[str, str, datetime]] = []
    for hh, path in sorted(cycle_paths.items()):
        cycle = datetime(day.year, day.month, day.day, int(hh), tzinfo=timezone.utc)
        if not Path(path).exists():
            per_cycle_stats.append({"cycle": hh, "status": "missing"})
            continue
        sha256, size_bytes = sha256_file(path)
        source_files.append(
            {"path": path, "sha256": sha256, "size_bytes": size_bytes, "cycle": hh}
        )
        # One record per (template, table hash), so a mid-day structure change
        # shows up as two records.
        for structure_key, structure in read_source_structures(path).items():
            root = structure.get("root_sequence", structure_key.split("@", 1)[0])
            key = (
                structure_key
                if "@" in structure_key
                else f"{root}@{structure.get('embedded_table_sha256', 'unknown')[:12]}"
            )
            source_structures.setdefault(key, structure)
        cycles.append((hh, path, cycle))

    if not cycles:
        return {"status": "no_data", "per_cycle": per_cycle_stats}

    # Every file carries the archive-wide declared schema.
    unified_schema = load_declared_schema()

    provenance = AmvFileProvenance(
        utc_day=day,
        source_files=tuple(source_files),
        source_structures=source_structures,
        decoder_version=earth2bufr.__version__,
        decoder_revision=decoder_revision,
        etl_revision=etl_revision,
        etl_git_sha=etl_git_sha(),
    )
    source_stream = primary_stream_for_year(day.year)

    t_write = time.time()
    sealed: set = set()
    quarantined = 0
    writer = AmvArchiveDayWriter(
        out_path,
        schema=unified_schema,
        provenance=provenance,
        scratch_dir=_default_scratch_dir(),
    )
    try:
        # Pass 2: bounded decode, projected and written batch by batch.
        for hh, path, cycle in cycles:
            started = time.time()
            cycle_rows = 0
            for table, dropped in iter_projected_batches(
                path, cycle, source_stream=source_stream
            ):
                writer.write(table)
                cycle_rows += table.num_rows
                quarantined += dropped
                del table
            per_cycle_stats.append(
                {
                    "cycle": hh,
                    "status": "ok",
                    "decode_project_sec": round(time.time() - started, 2),
                    "rows": cycle_rows,
                }
            )
            # Windows up to this cycle are complete, since later cycles only
            # produce later windows. finish_windows rejects an already sealed
            # window, hence the `sealed` set.
            boundary = np.datetime64(cycle.replace(tzinfo=None), "ns")
            ready = sorted(
                w for w in writer.populated_windows if w <= boundary and w not in sealed
            )
            if ready:
                writer.finish_windows(ready)
                sealed.update(ready)
        written = sum(stats["rows"] for stats in per_cycle_stats if "rows" in stats)
        writer.closing_metadata = {
            "amv.source_subsets_encountered": str(written + quarantined),
            "amv.quarantined_subsets": str(quarantined),
        }
        summary = writer.close()
    except Exception:
        writer.abort()
        raise
    write_sec = time.time() - t_write

    return {
        "status": "ok",
        "output_path": str(out_path),
        # Process lifetime high-water mark: under a persistent pool, the worst
        # day this worker has handled so far.
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "n_row_groups": summary.row_groups,
        "total_rows": summary.rows,
        "write_sec": round(write_sec, 2),
        "parquet_bytes": out_path.stat().st_size,
        "source_files": source_files,
        "per_cycle": per_cycle_stats,
        "contract": summary.contract,
    }


# ---------------------------------------------------------------------------
# Resume: the atomically published day file is the completion record; only
# failures are recorded, as per-day markers.
# ---------------------------------------------------------------------------


def _source_identity(records) -> list[tuple[str, str, int]]:
    # By content, not path: the same BUFR bytes at another location are the same source.
    return sorted((r["cycle"], r["sha256"], int(r["size_bytes"])) for r in records)


def _expected_sources(cycle_paths: Mapping[str, str]) -> list[tuple[str, str, int]]:
    records = []
    for hh, path in cycle_paths.items():
        if Path(path).exists():
            sha256, size_bytes = sha256_file(path)
            records.append({"cycle": hh, "sha256": sha256, "size_bytes": size_bytes})
    return _source_identity(records)


def _validate_reusable_output(
    path: Path,
    cycle_paths: Mapping[str, str],
    *,
    decoder_revision: str,
    etl_revision: str,
) -> dict[str, Any]:
    contract = validate_amv_parquet_contract(path)
    found = (
        contract["decoder_revision"],
        contract["etl_revision"],
        _source_identity(contract["source_cycles"]),
    )
    expected = (decoder_revision, etl_revision, _expected_sources(cycle_paths))
    names = ("decoder_revision", "etl_revision", "sources")
    mismatched = [name for name, a, b in zip(names, found, expected) if a != b]
    if not _schema_without_metadata(pq.read_schema(path)).equals(
        _schema_without_metadata(load_declared_schema())
    ):
        mismatched.append("columns")
    if mismatched:
        raise AssertionError(f"existing AMV archive contract differs: {mismatched}")
    return contract


def write_day_resumable(
    day: datetime,
    cycle_paths: dict[str, str],
    output_root: str,
    sensor: str = "satwnd",
    *,
    decoder_revision: str,
    etl_revision: str,
) -> dict:
    """``write_day``, skipping days whose existing output validates against the inputs.

    An existing file that fails validation is rewritten.
    """
    out_path = Path(output_root) / sensor / f"{day:%Y%m%d}.parquet"
    if out_path.exists():
        try:
            contract = _validate_reusable_output(
                out_path,
                cycle_paths,
                decoder_revision=decoder_revision,
                etl_revision=etl_revision,
            )
        except Exception as exc:
            print(f"{day:%Y-%m-%d}: rebuilding {out_path}: {exc}", flush=True)
        else:
            return {
                "status": "skipped_validated",
                "output_path": str(out_path),
                "parquet_bytes": out_path.stat().st_size,
                "contract": contract,
            }
    return write_day(
        day,
        cycle_paths,
        output_root,
        sensor,
        decoder_revision=decoder_revision,
        etl_revision=etl_revision,
    )


# ---------------------------------------------------------------------------
# Multi-process day scheduling (pool is archive_common.archive_runner.run_tasks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmvDayTask:
    day: datetime
    cycle_paths: Mapping[str, str]
    output_root: str
    decoder_revision: str
    etl_revision: str
    sensor: str = "satwnd"

    @property
    def output_path(self) -> Path:
        return Path(self.output_root) / self.sensor / f"{self.day:%Y%m%d}.parquet"

    def validate(self) -> None:
        if not self.cycle_paths:
            raise ValueError(f"{self.day:%Y-%m-%d}: no source cycles")
        if not self.decoder_revision:
            raise ValueError("decoder_revision must not be empty")
        if not self.etl_revision:
            raise ValueError("etl_revision must not be empty")


def initialize_amv_worker(arrow_threads: int) -> None:
    earth2bufr.set_num_threads(4)
    pa.set_cpu_count(arrow_threads)


def process_amv_day(task: AmvDayTask) -> dict:
    """Convert one AMV day. Module-level so spawned workers can import it."""
    return write_day_resumable(
        task.day,
        dict(task.cycle_paths),
        task.output_root,
        task.sensor,
        decoder_revision=task.decoder_revision,
        etl_revision=task.etl_revision,
    )


def _amv_failure_path(error_root: Path, task: AmvDayTask) -> Path:
    return Path(error_root) / f"{task.sensor}.{task.day:%Y%m%d}.error.json"


def write_amv_failure(error_root: Path, task: AmvDayTask, exc: BaseException) -> Path:
    marker = _amv_failure_path(error_root, task)
    write_json_atomic(
        marker,
        {
            "day": f"{task.day:%Y-%m-%d}",
            "sensor": task.sensor,
            "output_path": str(task.output_path),
            "cycles": sorted(task.cycle_paths),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": exception_traceback(exc)[-4000:],
        },
        indent=1,
        sort_keys=False,
    )
    return marker


def clear_amv_failure(error_root: Path, task: AmvDayTask) -> None:
    _amv_failure_path(error_root, task).unlink(missing_ok=True)


def recover_amv_day(task: AmvDayTask) -> dict[str, Any]:
    """Validate an output published just before its worker process exited."""
    contract = _validate_reusable_output(
        task.output_path,
        task.cycle_paths,
        decoder_revision=task.decoder_revision,
        etl_revision=task.etl_revision,
    )
    return {
        "status": "recovered_after_worker_exit",
        "output_path": str(task.output_path),
        "parquet_bytes": task.output_path.stat().st_size,
        "total_rows": contract["rows"],
        "n_row_groups": contract["row_groups"],
        "contract": contract,
    }


def run_days(
    tasks: Sequence[AmvDayTask],
    *,
    max_workers: int,
    arrow_threads: int,
    progress: Callable[[dict, int, int], None] | None = None,
    error_root: Path | str | None = None,
    continue_on_error: bool = False,
) -> tuple[dict, ...]:
    """Convert many AMV days across a process pool.

    With ``continue_on_error`` a failed day writes a marker under
    ``error_root`` instead of aborting the pool.
    """
    for task in tasks:
        task.validate()
    return run_tasks(
        tasks,
        process=process_amv_day,
        max_workers=max_workers,
        initializer=initialize_amv_worker,
        initargs=(arrow_threads,),
        progress=progress,
        describe=lambda task: f"{task.sensor} {task.day:%Y-%m-%d}",
        error_root=Path(error_root) if error_root is not None else None,
        continue_on_error=continue_on_error,
        on_failure=write_amv_failure,
        on_success=clear_amv_failure,
        recover_completed=recover_amv_day,
    )
