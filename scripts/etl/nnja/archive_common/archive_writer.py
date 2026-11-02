#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor-agnostic physical Parquet writer for NNJA archive contracts.

Per-window IPC spooling, one row group per complete DA window, and atomic
tmp+validate+replace publication. Sensor packages subclass
``ArchiveDayWriter`` and implement ``bind_metadata`` and ``validate_contract``.
"""

from __future__ import annotations

import os
import uuid
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from healda.utils.profiling import cpu_timing_range

# Duck-typed: needs .validate().
FileProvenance = Any


def _schema_without_metadata(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [pa.field(field.name, field.type, nullable=field.nullable) for field in schema]
    )


@dataclass(frozen=True)
class WriteSummary:
    path: str
    rows: int
    row_groups: int
    row_group_policy: str
    windows: tuple[str, ...]
    contract: dict[str, Any] | None = None


@dataclass
class _WindowSpool:
    path: Path
    sink: pa.NativeFile
    writer: Any

    def close(self) -> None:
        self.writer.close()
        self.sink.close()


def parquet_leaf_paths(schema: pa.Schema) -> tuple[str, ...]:
    """Physical Parquet leaf column paths written for an Arrow schema."""

    paths: list[str] = []

    def visit(data_type: pa.DataType, path: tuple[str, ...]) -> None:
        if pa.types.is_struct(data_type):
            for child in data_type:
                visit(child.type, (*path, child.name))
            return
        if (
            pa.types.is_list(data_type)
            or pa.types.is_large_list(data_type)
            or pa.types.is_fixed_size_list(data_type)
        ):
            visit(data_type.value_type, (*path, "list", "element"))
            return
        if pa.types.is_map(data_type):
            visit(data_type.key_type, (*path, "key_value", "key"))
            visit(data_type.item_type, (*path, "key_value", "value"))
            return
        paths.append(".".join(path))

    for field in schema:
        visit(field.type, (field.name,))
    return tuple(paths)


def writer_options(
    schema: pa.Schema, delta_encoded_columns: Sequence[str] = ()
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "compression": "zstd",
        "compression_level": 3,
        "data_page_version": "2.0",
        "write_statistics": ["da_window"],
        "use_byte_stream_split": False,
    }
    if not delta_encoded_columns:
        options["use_dictionary"] = True
        return options
    # PyArrow rejects a per-column encoding while dictionary encoding is global,
    # so every other physical leaf, nested ones included, is listed explicitly.
    delta = set(delta_encoded_columns)
    options["use_dictionary"] = [
        path for path in parquet_leaf_paths(schema) if path not in delta
    ]
    options["column_encoding"] = dict.fromkeys(
        delta_encoded_columns, "DELTA_BINARY_PACKED"
    )
    return options


def _window_values(table: pa.Table) -> list[datetime]:
    column = table["da_window"]
    if column.null_count:
        raise ValueError("required archive field 'da_window' contains nulls")
    values = sorted(pc.unique(column.combine_chunks()).to_pylist())
    for value in values:
        if value.minute or value.second or value.microsecond:
            raise ValueError(f"da_window is not aligned to an hour: {value}")
        if value.hour % 3:
            raise ValueError(f"da_window is not three-hour aligned: {value}")
    return values


def _window_parts(
    table: pa.Table, windows: list[datetime]
) -> Iterable[tuple[datetime, pa.Table]]:
    """Yield zero-copy source-ordered runs for the open DA windows."""

    if len(windows) == 1:
        yield windows[0], table
        return

    encoded = pc.run_end_encode(table["da_window"].combine_chunks())
    start = 0
    for end, window in zip(
        encoded.run_ends.to_pylist(), encoded.values.to_pylist(), strict=True
    ):
        yield window, table.slice(start, end - start)
        start = end


class ArchiveDayWriter:
    """IPC-spooled writer for one sensor/day archive file.

    Sensor packages subclass this and override ``bind_metadata`` and
    ``validate_contract``; everything else is inherited unchanged.
    ``provenance`` may be None when it is known only after streaming; its footer
    keys then go in ``closing_metadata`` before ``close()``.
    """

    def bind_metadata(self, schema: pa.Schema, provenance: Any) -> pa.Schema:
        """Attach per-file provenance to the schema footer."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement bind_metadata()"
        )

    def validate_contract(self, path: Path) -> dict[str, Any]:
        """Reopen and check the written file before it is published."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement validate_contract()"
        )

    def __init__(
        self,
        path: Path,
        *,
        schema: pa.Schema,
        provenance: FileProvenance | None,
        scratch_dir: Path | None = None,
        delta_encoded_columns: Sequence[str] = (),
    ) -> None:
        self.path = path
        # Unique per attempt: a retried day or a second process on the same
        # output root must not unlink another writer's in-progress file.
        self.temporary = path.with_name(
            f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp"
        )
        self.schema = schema
        if provenance is None:
            self.output_schema = schema
        else:
            provenance.validate()
            self.output_schema = self.bind_metadata(schema, provenance)
        # Footer keys known only once every row is written; set before close().
        self.closing_metadata: dict[str, str] = {}
        self.provenance = provenance
        self._expected_schema = _schema_without_metadata(schema)
        self._spools: dict[datetime, _WindowSpool] = {}
        self._windows: set[datetime] = set()
        self._last_finished_window: datetime | None = None
        self._flushed_windows: set[datetime] = set()
        # Sealed windows are written on one thread, in order, while the caller
        # converts the next cycle; Parquet encoding releases the GIL.
        self._flush_thread = ThreadPoolExecutor(max_workers=1)
        self._flushes: list[Future] = []
        self._rows = 0
        self._row_groups = 0
        self._closed = False
        self._summary: WriteSummary | None = None

        path.parent.mkdir(parents=True, exist_ok=True)
        if scratch_dir is None:
            scratch_dir = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self._spool_root = Path(
            tempfile.mkdtemp(prefix=f"{path.name}.spool-", dir=scratch_dir)
        )
        try:
            self._writer = pq.ParquetWriter(
                self.temporary,
                self.output_schema,
                **writer_options(schema, delta_encoded_columns),
            )
        except Exception:
            self._flush_thread.shutdown()
            shutil.rmtree(self._spool_root, ignore_errors=True)
            raise

    def __enter__(self) -> ArchiveDayWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> None:
        if exception_type is None:
            self.close()
        else:
            self.abort()

    @property
    def populated_windows(self) -> tuple[datetime, ...]:
        return tuple(sorted(self._windows))

    @property
    def summary(self) -> WriteSummary:
        if self._summary is None:
            raise RuntimeError("writer has not closed successfully")
        return self._summary

    def _normalize_table(self, batch: pa.Table | pa.RecordBatch) -> pa.Table:
        table = (
            batch
            if isinstance(batch, pa.Table)
            else pa.Table.from_batches([batch], schema=batch.schema)
        )
        actual = _schema_without_metadata(table.schema)
        if not actual.equals(self._expected_schema):
            raise TypeError(
                "table schema does not match archive contract:\n"
                f"{actual}\n{self._expected_schema}"
            )
        for field in self.schema:
            if not field.nullable and table[field.name].null_count:
                raise ValueError(
                    f"required archive field {field.name!r} contains nulls"
                )
        return pa.Table.from_arrays(table.columns, schema=self.output_schema)

    def write(self, batch: pa.Table | pa.RecordBatch) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        with cpu_timing_range("writer.normalize"):
            table = self._normalize_table(batch)
        with cpu_timing_range("writer.route_ipc"):
            windows = _window_values(table)
            self._rows += table.num_rows
            for window, part in _window_parts(table, windows):
                if (
                    self._last_finished_window is not None
                    and window <= self._last_finished_window
                ):
                    raise ValueError(
                        f"da_window {window.isoformat()} was already finalized; "
                        "source cycles must be finalized in chronological order"
                    )
                self._windows.add(window)
                spool = self._spools.get(window)
                if spool is None:
                    path = self._spool_root / window.strftime("%Y%m%dT%H%M%S%z.arrow")
                    sink = pa.OSFile(str(path), "wb")
                    spool = _WindowSpool(
                        path=path,
                        sink=sink,
                        writer=pa.ipc.new_file(sink, self.output_schema),
                    )
                    self._spools[window] = spool
                spool.writer.write_table(part)

    def _write_spool(self, window: datetime, spool: _WindowSpool) -> None:
        source = pa.memory_map(str(spool.path), "r")
        try:
            with cpu_timing_range("writer.ipc_read", group="writer-thread"):
                table = pa.ipc.open_file(source).read_all()
            with cpu_timing_range("writer.parquet_write", group="writer-thread"):
                self._writer.write_table(table, row_group_size=table.num_rows)
            self._row_groups += 1
            self._flushed_windows.add(window)
        finally:
            source.close()
            spool.path.unlink(missing_ok=True)

    def _flush(self, window: datetime) -> None:
        spool = self._spools.pop(window, None)
        if spool is None:
            return
        with cpu_timing_range("writer.ipc_close"):
            spool.close()
        for flush in self._flushes:
            if flush.done():
                flush.result()
        self._flushes.append(
            self._flush_thread.submit(self._write_spool, window, spool)
        )

    def _wait_for_flushes(self) -> None:
        with cpu_timing_range("writer.flush_wait"):
            for flush in self._flushes:
                flush.result()

    def finish_windows(self, windows: Iterable[datetime]) -> None:
        """Seal complete windows after their source cycle has been consumed.

        NCEP messages can revisit both three-hour windows within one six-hour
        cycle, so every batch of a cycle must be written before its windows are
        sealed. A sealed window rejects further rows.
        """

        if self._closed:
            raise RuntimeError("writer is closed")
        requested = list(windows)
        if requested != sorted(set(requested)):
            raise ValueError("windows must be unique and chronologically ordered")
        for window in requested:
            if (
                self._last_finished_window is not None
                and window <= self._last_finished_window
            ):
                raise ValueError(
                    f"da_window {window.isoformat()} was already finalized"
                )
            earlier = sorted(value for value in self._spools if value < window)
            if earlier:
                raise ValueError(
                    f"cannot finalize {window.isoformat()} before buffered window "
                    f"{earlier[0].isoformat()}"
                )
            self._flush(window)
            self._last_finished_window = window

    def close(self) -> WriteSummary:
        """Flush, validate the written file, and atomically publish."""

        if self._closed:
            return self.summary
        try:
            self.finish_windows(sorted(self._spools))
            self._wait_for_flushes()
            if self.closing_metadata:
                # Lands in the Parquet key-value footer, not the embedded Arrow
                # schema, which is fixed when the writer opens.
                self._writer.add_key_value_metadata(self.closing_metadata)
            with cpu_timing_range("writer.parquet_close"):
                self._writer.close()
            with cpu_timing_range("writer.reopen_validate"):
                contract = self.validate_contract(self.temporary)
            with cpu_timing_range("writer.publish"):
                self.temporary.replace(self.path)
            shutil.rmtree(self._spool_root, ignore_errors=True)
            self._summary = WriteSummary(
                path=str(self.path),
                rows=self._rows,
                row_groups=self._row_groups,
                row_group_policy="one_complete_da_window",
                windows=tuple(window.isoformat() for window in sorted(self._windows)),
                contract=contract,
            )
            return self._summary
        except Exception:
            try:
                self._flush_thread.shutdown(cancel_futures=True)
                self._writer.close()
            finally:
                self.temporary.unlink(missing_ok=True)
                self._cleanup_spools()
            raise
        finally:
            self._flush_thread.shutdown()
            self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._flush_thread.shutdown(cancel_futures=True)
            self._writer.close()
        finally:
            self.temporary.unlink(missing_ok=True)
            self._cleanup_spools()
            self._closed = True

    def _cleanup_spools(self) -> None:
        for spool in self._spools.values():
            try:
                spool.close()
            except Exception:
                pass
        self._spools.clear()
        shutil.rmtree(self._spool_root, ignore_errors=True)
