#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert NNJA PrepBUFR cycle files to Parquet with earth2bufr.

One flat Parquet file per PrepBUFR cycle covering every data category. For each
observed variable the PREPRO value is kept - the bottom of the NCEP event stack,
the observation before any QC or correction (pre-VIRTMP for temperature) -
together with its quality mark and program/reason codes. Radiosondes are
flattened like every other category.

Each file holds exactly two row groups relative to the cycle analysis time t0:
  row group 0 -> observations at/before t0 (DHR <= 0)
  row group 1 -> observations after t0     (DHR > 0)

A report with missing DHR is an error.

Layout:
  <output-root>/cycles/<year>/<cycle>.parquet   unified flat prepro decode

Training can treat the directory as a virtual dataset via pyarrow.dataset.

Examples:
  python -m scripts.etl.nnja.conv.nnja_prepbufr_to_parquet --year 2021 --input-root <bufr/prepbufr> --output-root <parquet>
  python -m scripts.etl.nnja.conv.nnja_prepbufr_to_parquet ... --resume
  python -m scripts.etl.nnja.conv.nnja_prepbufr_to_parquet ... --max-cycles 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import earth2bufr
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from healda.observations.schema import table_schema

from scripts.etl.nnja.conv.prepbufr_flatten import recursive_flatten

WORKER_RESTART_EVERY = 10
CYCLE_RE = re.compile(
    r"gdas\.(?P<date>\d{8})\.t(?P<hour>\d{2})z\.prepbufr\.nr$", re.IGNORECASE
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NNJA PrepBUFR cycle files to per-cycle Parquet datasets."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Calendar year to process (any year with data under --input-root)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="holds <year>/**/gdas.*.prepbufr.nr",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cycles-dir",
        type=Path,
        default=None,
        help="Output dir (default: <output-root>/cycles/<year>)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for decoding (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cycles whose output parquet files already exist",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Process at most N cycle files (for testing)",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    year_dir = args.input_root / str(args.year)
    if not year_dir.is_dir():
        raise FileNotFoundError(f"Input year directory not found: {year_dir}")

    cycles_dir = args.cycles_dir or (args.output_root / "cycles" / str(args.year))
    cycles_dir.mkdir(parents=True, exist_ok=True)
    return year_dir, cycles_dir


def list_prepbufr_files(year_dir: Path) -> list[Path]:
    files = sorted(year_dir.rglob("gdas.*.prepbufr.nr"))
    if not files:
        raise FileNotFoundError(f"No PrepBUFR files found under {year_dir}")
    return files


def parse_cycle_metadata(path: Path) -> tuple[str, str]:
    match = CYCLE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unexpected PrepBUFR filename: {path.name}")
    return match.group("date"), match.group("hour")


def empty_meta_table() -> pa.Table:
    return pa.table(
        {
            "obs_date": pa.array([], type=pa.string()),
            "obs_time": pa.array([], type=pa.timestamp("ns", tz="UTC")),
            "source_file": pa.array([], type=pa.string()),
            "cycle_date": pa.array([], type=pa.string()),
            "cycle_hour": pa.array([], type=pa.string()),
            "data_category": pa.array([], type=pa.int64()),
            "data_category_name": pa.array([], type=pa.string()),
        }
    )


def cycle_time(cycle_date: str, cycle_hour: str) -> pd.Timestamp:
    return pd.to_datetime(cycle_date + cycle_hour, format="%Y%m%d%H", utc=True).as_unit(
        "ns"
    )


# Row-group keys ordered so the at/before-t0 group is written first.
HALF_LE_T0 = "0_le_t0"
HALF_GT_T0 = "1_gt_t0"


def split_frame_at_t0(df: pd.DataFrame) -> dict[str, pa.Table]:
    """Split a cycle frame into two tables relative to the analysis time t0.

    Rows with obs_time <= cycle time go to the at/before-t0 half; rows with
    obs_time > cycle time (or no obs_time) go to the after-t0 half. Every row
    of a frame belongs to one cycle.
    """
    cycle = cycle_time(str(df["cycle_date"].iloc[0]), str(df["cycle_hour"].iloc[0]))
    le_mask = df["obs_time"] <= cycle

    out: dict[str, pa.Table] = {}
    if le_mask.any():
        out[HALF_LE_T0] = pa.Table.from_pandas(df[le_mask], preserve_index=False)
    if (~le_mask).any():
        out[HALF_GT_T0] = pa.Table.from_pandas(df[~le_mask], preserve_index=False)
    return out


def _float32(array: pa.Array) -> pa.Array:
    array = pc.cast(array, pa.float32(), safe=False)
    if pc.any(pc.is_nan(array)).as_py():
        array = pc.if_else(pc.is_nan(array), pa.scalar(None, pa.float32()), array)
    return array


def _archive_frame(
    df: pd.DataFrame,
    code: int,
    category_name: str,
    source_file: str,
    obs_date: str,
    cycle: pd.Timestamp,
) -> pd.DataFrame:
    df["obs_date"] = obs_date
    df["source_file"] = source_file
    df["cycle_date"] = f"{cycle:%Y%m%d}"
    df["cycle_hour"] = f"{cycle:%H}"
    df["data_category"] = code
    df["data_category_name"] = category_name
    df["obs_time"] = cycle + pd.to_timedelta(df["DHR"], unit="h")
    float64_columns = df.select_dtypes(include=["float64"]).columns
    return df.astype({col: "float32" for col in float64_columns})


def batch_to_halves(
    batch: pa.RecordBatch,
    source_file: str,
    obs_date: str,
    category_names: dict[int, str],
) -> dict[str, pa.Table]:
    """One flattened category batch as archive tables split at t0.

    ``category_names`` maps data category to message type mnemonic, as
    ``message_type_names`` reads it from the file.

    The archive's types and ``pandas`` footer are those of the batch converted
    to pandas, given the cycle columns and obs_time, downcast to float32 and
    converted back per half with from_pandas. The bulk columns are converted
    in Arrow by the same rules (floats, and integers with nulls, become float32
    with NaN as null); list columns, whose type from_pandas infers from the
    values, and the footer, which depends on the frame's dtypes, still go
    through pandas.
    """
    cycle_date, cycle_hour = parse_cycle_metadata(Path(source_file))
    cycle = cycle_time(cycle_date, cycle_hour)
    code = int(batch.column("_data_category")[0].as_py())
    category_name = category_names[code]
    dhr = batch.column("DHR")
    if dhr.null_count:
        raise ValueError(
            f"{source_file}: {dhr.null_count} reports with null DHR in "
            f"category {category_name} ({code})"
        )
    n = batch.num_rows

    # The frame's dtypes: to_pandas gives integer and boolean columns with nulls
    # another dtype, so the sample holds the first null of each.
    sample_rows = {0}
    for column in batch.columns:
        t = column.type
        if column.null_count and (pa.types.is_integer(t) or pa.types.is_boolean(t)):
            sample_rows.add(int(np.argmin(pc.is_valid(column).to_numpy(False))))
    sample = _archive_frame(
        batch.take(sorted(sample_rows)).to_pandas(),
        code,
        category_name,
        source_file,
        obs_date,
        cycle,
    )
    obs_time = cycle + pd.to_timedelta(dhr.to_numpy(zero_copy_only=False), unit="h")

    def constant(value: str) -> pa.Array:
        return pa.repeat(pa.scalar(value, pa.string()), n)

    columns: dict[str, pa.Array] = {}
    from_pandas: dict[str, pd.Series] = {}
    for name, column in zip(batch.schema.names, batch.columns):
        t = column.type
        if pa.types.is_list(t) and not pc.sum(pc.list_value_length(column)).as_py():
            # Lists without entries: from_pandas gives list<null>.
            columns[name] = pa.ListArray.from_arrays(
                column.offsets, pa.nulls(0), mask=column.is_null()
            )
        elif pa.types.is_list(t):
            from_pandas[name] = column.to_pandas()
        elif pa.types.is_floating(t) or (pa.types.is_integer(t) and column.null_count):
            columns[name] = _float32(pc.cast(column, pa.float64()))
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            columns[name] = pc.cast(column, pa.string())
        elif pa.types.is_integer(t) or pa.types.is_boolean(t) or pa.types.is_null(t):
            columns[name] = column
        else:
            raise TypeError(f"{name}: {t} has no archive conversion")
    columns.update(
        obs_date=constant(obs_date),
        source_file=constant(source_file),
        cycle_date=constant(cycle_date),
        cycle_hour=constant(cycle_hour),
        data_category=pa.repeat(pa.scalar(code, pa.int64()), n),
        data_category_name=constant(category_name),
        obs_time=pa.array(obs_time, type=pa.timestamp("ns", tz="UTC")),
    )
    order = list(sample.columns)
    le_mask = np.asarray(obs_time <= cycle)

    def half(mask: np.ndarray) -> pa.Table:
        keep = pa.array(mask)
        arrays = [
            pa.array(from_pandas[name][mask], from_pandas=True)
            if name in from_pandas
            else pc.filter(columns[name], keep)
            for name in order
        ]
        # from_pandas gives a column that is null in every row the null type.
        arrays = [
            pa.nulls(len(array))
            if pa.types.is_list(array.type) and array.null_count == len(array)
            else array
            for array in arrays
        ]
        part = pa.Table.from_arrays(arrays, names=order)
        # from_pandas would type str columns as it does for the sample.
        sample_types = pa.Schema.from_pandas(sample, preserve_index=False)
        schema = pa.schema(
            [
                sample_types.field(f.name) if pa.types.is_string(f.type) else f
                for f in part.schema
            ]
        )
        metadata = pa.Table.from_pandas(
            sample.iloc[:0], schema=schema, preserve_index=False
        ).schema.metadata
        return part.replace_schema_metadata(metadata)

    out: dict[str, pa.Table] = {}
    if le_mask.any():
        out[HALF_LE_T0] = half(le_mask)
    if (~le_mask).any():
        out[HALF_GT_T0] = half(~le_mask)
    return out


def conform_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reindex a table to match a target schema, filling missing columns with nulls."""
    arrays = []
    columns = set(table.column_names)
    for field in schema:
        if field.name in columns:
            col = table.column(field.name)
            if col.type != field.type:
                col = col.cast(field.type)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.table(arrays, schema=schema)


def with_base_types(schema: pa.Schema, base_schema: pa.Schema) -> pa.Schema:
    fields = []
    for field in schema:
        if field.name in base_schema.names:
            base = base_schema.field(field.name)
            field = field.with_type(base.type).with_nullable(base.nullable)
        fields.append(field)
    return pa.schema(fields, metadata=schema.metadata)


def with_base_columns(table: pa.Table, base_schema: pa.Schema) -> pa.Table:
    extra = [field for field in table.schema if field.name not in base_schema.names]
    schema = pa.schema([field.remove_metadata() for field in base_schema] + extra)
    return conform_to_schema(table, schema)


class CycleParquetBuilder:
    """Accumulate cycle table parts, then write two row groups (<=t0, >t0).

    Columns `base_schema` names are written with its types; the rest keep the decoded ones.
    """

    def __init__(self, string_type: pa.DataType, base_schema: pa.Schema) -> None:
        # pandas 2 converts str columns to string, pandas 3 to large_string.
        self.string_type = string_type
        self.base_schema = base_schema
        self._windows: dict[str, list[pa.Table]] = defaultdict(list)
        self.rows = 0

    def add_table(self, table: pa.Table) -> None:
        if table.num_rows:
            self.add_frame(table.to_pandas())

    def add_frame(self, df: pd.DataFrame) -> None:
        if not df.empty:
            self.add_halves(split_frame_at_t0(df))

    def add_halves(self, halves: dict[str, pa.Table]) -> None:
        for key, part in halves.items():
            self._windows[key].append(part)
            self.rows += part.num_rows

    def _output_schema(self, schema: pa.Schema) -> pa.Schema:
        strings = pa.schema(
            [
                f.with_type(self.string_type)
                if f.type in (pa.string(), pa.large_string())
                else f
                for f in schema
            ],
            metadata=schema.metadata,
        )
        return with_base_types(strings, self.base_schema)

    def write(self, output_path: Path, *, empty_table: pa.Table | None = None) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".parquet.tmp")

        if not self._windows:
            stub = with_base_columns(
                empty_table
                if empty_table is not None
                else pa.table({"_empty": pa.array([], type=pa.bool())}),
                self.base_schema,
            )
            stub = stub.cast(self._output_schema(stub.schema))
            pq.write_table(stub, tmp_path, compression="zstd")
            tmp_path.replace(output_path)
            size_mb = output_path.stat().st_size / (1024**2)
            return {
                "path": str(output_path),
                "rows": 0,
                "row_groups": 0,
                "size_mb": size_mb,
            }

        window_tables = {
            key: pa.concat_tables(self._windows[key], promote_options="default")
            for key in sorted(self._windows)
        }

        schema = self._output_schema(
            pa.unify_schemas([t.schema for t in window_tables.values()])
        )
        rows = 0
        n_groups = 0
        writer = pq.ParquetWriter(
            tmp_path, schema=schema, compression="zstd", use_dictionary=True
        )
        try:
            for key in sorted(window_tables):
                part = conform_to_schema(window_tables[key], schema)
                writer.write_table(part, row_group_size=part.num_rows)
                rows += part.num_rows
                n_groups += 1
        finally:
            writer.close()

        tmp_path.replace(output_path)
        size_mb = output_path.stat().st_size / (1024**2)
        return {
            "path": str(output_path),
            "rows": rows,
            "row_groups": n_groups,
            "size_mb": size_mb,
        }


def read_nested_batches(input_path: Path) -> list:
    """Return nested (flatten=False) batches for all categories in the cycle."""
    try:
        return earth2bufr.read_bufr(input_path)
    except ValueError as exc:
        if "No matching data messages found" in str(exc):
            return []
        raise


def message_type_names(input_path: Path) -> dict[int, str]:
    """Data category -> Table A mnemonic of each message family in a PrepBUFR file.

    The mnemonic comes from the file's own DX tables: files before 2007 number
    the families 230-249 (sequences 3-60-2xx), later files 102-121 (3-48-1xx),
    with the same mnemonics.
    """
    with earth2bufr.open_bufr(input_path) as source:
        first_message = {}
        for message in source.info.data_messages:
            first_message.setdefault(message.data_category, message)
        names = {}
        for category, message in first_message.items():
            structure = source.source_structure(message.source_message_index)
            sequences = {
                s["descriptor"]: s["mnemonic"] for s in structure["active_sequences"]
            }
            table_a = next(d for d in message.descriptors if d.startswith("3-"))
            names[category] = sequences[table_a]
    return names


def decode_cycle(input_path: Path, cycles_dir: Path) -> dict:
    """Decode one PrepBUFR cycle into a single unified flat prepro parquet file.

    Every category is decoded nested, then flattened keeping the bottom-of-stack
    (PREPRO / raw) event per variable (take='last'). All categories are merged
    into one file with 2 row groups split at t0.
    """
    obs_date, _cycle_hour = parse_cycle_metadata(input_path)
    cycle_id = input_path.name

    builder = CycleParquetBuilder(
        string_type=pa.string(), base_schema=table_schema("prepbufr")
    )
    category_names = message_type_names(input_path)
    for batch in read_nested_batches(input_path):
        flat = recursive_flatten(batch, take="last")
        if flat.num_rows:
            builder.add_halves(
                batch_to_halves(flat, cycle_id, obs_date, category_names)
            )

    cycles_path = cycles_dir / f"{cycle_id}.parquet"
    cycles_info = builder.write(cycles_path, empty_table=empty_meta_table())

    return {
        "cycle": cycle_id,
        "cycles": cycles_info,
        "rows": cycles_info["rows"],
    }


def timed_cycle(
    decode: Callable[[Path, Path], dict], input_path: str, out_dir: str
) -> dict:
    t0 = time.perf_counter()
    info = decode(Path(input_path), Path(out_dir))
    info["elapsed_s"] = time.perf_counter() - t0
    return info


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"processed_cycles": []}


def process_cycles(
    cycle_files: list[Path],
    out_dir: Path,
    decode: Callable[[Path, Path], dict],
    *,
    workers: int,
    resume: bool,
    manifest_path: Path,
) -> list[dict]:
    """Run decode(input_path, out_dir) on each cycle, recording the manifest."""
    manifest = load_manifest(manifest_path)
    done = set(manifest.get("processed_cycles", []))
    if resume:
        pending = [
            p
            for p in cycle_files
            if p.name not in done and not (out_dir / f"{p.name}.parquet").exists()
        ]
    else:
        pending = cycle_files
    results: list[dict] = []
    total = len(pending)

    def record(info: dict, n: int) -> None:
        results.append(info)
        done.add(info["cycle"])
        manifest["processed_cycles"] = sorted(done)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(
            f"[{n}/{total}] {info['cycle']}: "
            f"{info['rows']:,} rows, "
            f"{info['cycles']['row_groups']} row groups, "
            f"{info['elapsed_s']:.1f}s",
            flush=True,
        )

    if workers <= 1:
        for i, path in enumerate(pending, 1):
            record(timed_cycle(decode, str(path), str(out_dir)), i)
        return results

    # A fresh pool per batch of cycles: worker processes exit between batches and
    # return their memory to the OS.
    batch_size = workers * WORKER_RESTART_EVERY
    done_n = 0
    for start in range(0, total, batch_size):
        batch = pending[start : start + batch_size]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(timed_cycle, decode, str(path), str(out_dir))
                for path in batch
            ]
            for fut in as_completed(futures):
                done_n += 1
                record(fut.result(), done_n)
    return results


def write_manifest(
    manifest_path: Path,
    *,
    year: int,
    results: list[dict],
) -> None:
    manifest = load_manifest(manifest_path)
    manifest.update(
        {
            "year": year,
            "layout": "per_cycle_unified_flat",
            "value_kept": "prepro (bottom-of-stack raw value + QC mark)",
            "chunking": "split_at_t0",
            "row_groups": ["le_t0 (DHR<=0)", "gt_t0 (DHR>0)"],
            "cycles_dir": f"cycles/{year}",
            "float_dtype": "float32",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cycles": results,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))


def clean_outputs(cycles_dir: Path, manifest_path: Path) -> None:
    """Delete prior cycle outputs and reset the manifest before a non-resume run."""
    for child in cycles_dir.iterdir():
        if child.suffix in (".parquet", ".tmp"):
            child.unlink()

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        manifest["processed_cycles"] = []
        manifest_path.write_text(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        year_dir, cycles_dir = resolve_paths(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    manifest_dir = args.output_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.year}.manifest.json"
    cycle_files = list_prepbufr_files(year_dir)
    if args.max_cycles is not None:
        cycle_files = cycle_files[: args.max_cycles]

    print(f"Year:       {args.year}")
    print(f"Input:      {year_dir} ({len(cycle_files):,} cycle files)")
    print(f"Output:     {cycles_dir}")
    print("Values:     PREPRO (raw, bottom-of-stack) value + QC mark, all sensors")
    print("Chunking:   2 row groups per cycle (<=t0: DHR<=0, >t0: DHR>0)")
    print(
        f"Workers:    {args.workers} "
        f"(fresh pool every {args.workers * WORKER_RESTART_EVERY} cycles)"
    )

    if not args.resume:
        clean_outputs(cycles_dir, manifest_path)

    results = process_cycles(
        cycle_files,
        cycles_dir,
        decode_cycle,
        workers=args.workers,
        resume=args.resume,
        manifest_path=manifest_path,
    )

    total_rows = sum(r.get("rows") or 0 for r in results)
    print(f"Done: {len(results):,} cycles, {total_rows:,} rows")

    write_manifest(manifest_path, year=args.year, results=results)
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
