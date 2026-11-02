#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert NNJA GPSRO BUFR cycle files to per-cycle Parquet.

One Parquet file per cycle:
  <output-root>/gpsro_v3/<year>/<cycle>.parquet
with one row per (occultation, vertical level).

Keeps the DRF16BIT bending-angle observation (rawest L1b) plus derived
source-only pressure/height columns from DRF16BIT_2 refractivity (HEIT+ARFR)
and header GEODU/ELRC: ``impact_height_m`` (IMPP-ELRC, the GSI-diagnostic-
convention height, the model's height coordinate), ``heit_at_impact_m``
(refraction-corrected true height, used only for the standard-atmosphere
lookup), ``dry_pressure_hpa``, ``standard_atmosphere_pressure_hpa`` (at
``heit_at_impact_m``) and ``blended_pressure_5km_hpa``. The background-influenced
1D-Var profile (DRF16BIT_3) is dropped.

Each file has two row groups split at the cycle analysis time t0
(obs_time <= t0 vs > t0), matching the conventional prepbufr layout.

GPSRO BUFR embeds its NCEP DX tables, which earth2bufr.read_bufr reads from the
file.

Examples:
  python -m scripts.etl.nnja.gpsro.nnja_gpsro_to_parquet --year 2025 --input-root <bufr/gpsro> --output-root <parquet>
  python -m scripts.etl.nnja.gpsro.nnja_gpsro_to_parquet ... --max-cycles 4 --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import earth2bufr
import pyarrow as pa
from healda.observations.schema import table_schema

from scripts.etl.nnja.conv.nnja_prepbufr_to_parquet import (
    CycleParquetBuilder,
    clean_outputs,
    load_manifest,
    process_cycles,
)
from scripts.etl.nnja.gpsro.gpsro_projection import (
    GPSRO_CODE,
    empty_gpsro_table,
    project_gpsro_batch,
)

PRODUCT = "gpsro_v3"
CYCLE_RE = re.compile(
    r"gdas\.(?P<date>\d{8})\.t(?P<hour>\d{2})z\.gpsro\.tm00\.bufr_d$", re.IGNORECASE
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert NNJA GPSRO BUFR to per-cycle Parquet."
    )
    p.add_argument("--year", type=int, required=True)
    p.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="holds <year>/**/gdas.*.gpsro.tm00.bufr_d",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpsro-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-cycles", type=int, default=None)
    return p.parse_args(argv)


def parse_cycle_metadata(path: Path) -> tuple[str, str]:
    m = CYCLE_RE.search(path.name)
    if not m:
        raise ValueError(f"unexpected GPSRO filename: {path.name}")
    return m.group("date"), m.group("hour")


def list_gpsro_files(year_dir: Path) -> list[Path]:
    files = sorted(year_dir.rglob("gdas.*.gpsro.tm00.bufr_d"))
    if not files:
        raise FileNotFoundError(f"No GPSRO files under {year_dir}")
    return files


def decode_cycle(input_path: Path, gpsro_dir: Path) -> dict:
    cycle_date, cycle_hour = parse_cycle_metadata(input_path)
    cycle_id = input_path.name
    builder = CycleParquetBuilder(
        string_type=pa.large_string(), base_schema=table_schema("gpsro")
    )
    try:
        batches = earth2bufr.read_bufr(input_path, data_category_filter=[GPSRO_CODE])
    except ValueError as exc:
        if "No matching data messages found" in str(exc):
            batches = []
        else:
            raise
    for batch in batches:
        builder.add_table(
            project_gpsro_batch(
                batch,
                source_file=cycle_id,
                cycle_date=cycle_date,
                cycle_hour=cycle_hour,
            )
        )

    out_path = gpsro_dir / f"{cycle_id}.parquet"
    info = builder.write(out_path, empty_table=empty_gpsro_table())
    return {"cycle": cycle_id, "cycles": info, "rows": info["rows"]}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    year_dir = args.input_root / str(args.year)
    if not year_dir.is_dir():
        print(f"error: input year dir not found: {year_dir}", file=sys.stderr)
        return 1
    gpsro_dir = args.gpsro_dir or (args.output_root / PRODUCT / str(args.year))
    gpsro_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{PRODUCT}_{args.year}.manifest.json"

    cycle_files = list_gpsro_files(year_dir)
    if args.max_cycles is not None:
        cycle_files = cycle_files[: args.max_cycles]

    print(f"Year:       {args.year}")
    print(f"Input:      {year_dir} ({len(cycle_files):,} GPSRO cycle files)")
    print(f"Output:     {gpsro_dir}")
    print(
        "Values:     DRF16BIT bending angle + impact_height_m/heit_at_impact_m/"
        "dry_pressure_hpa/standard_atmosphere_pressure_hpa/blended_pressure_5km_hpa "
        "from DRF16BIT_2; 1D-Var (DRF16BIT_3) dropped"
    )
    print("Chunking:   2 row groups per cycle (<=t0 / >t0)")
    print(f"Workers:    {args.workers}", flush=True)

    if not args.resume:
        clean_outputs(gpsro_dir, manifest_path)

    results = process_cycles(
        cycle_files,
        gpsro_dir,
        decode_cycle,
        workers=args.workers,
        resume=args.resume,
        manifest_path=manifest_path,
    )
    total_rows = sum(r.get("rows") or 0 for r in results)
    print(f"Done: {len(results):,} cycles, {total_rows:,} rows")

    manifest = load_manifest(manifest_path)
    manifest.update(
        {
            "year": args.year,
            "product": PRODUCT,
            "layout": "per_cycle_bending_angle_with_dry_pressure_v3",
            "value_kept": (
                "DRF16BIT bending angle (IMPP + BNDA + L1/L2 + FOST error); "
                "impact_height_m = IMPP-ELRC, GSI-diagnostic-convention height "
                "(recommended model height coordinate); "
                "heit_at_impact_m = true refraction-corrected height, HEIT "
                "interpolated to IMPP via Bouguer's rule (pressure-lookup input only); "
                "dry_pressure_hpa from DRF16BIT_2 HEIT/ARFR + GEODU/ELRC; "
                "standard_atmosphere_pressure_hpa at heit_at_impact_m (fixed from v2, "
                "which incorrectly used impact_height_m); "
                "blended_pressure_5km_hpa = 0.8*standard_atmosphere_pressure_hpa + "
                "0.2*dry_pressure_hpa below 5km, dry_pressure_hpa above (validated "
                "best single source-only pressure candidate)"
            ),
            "chunking": "split_at_t0",
            "float_dtype": "float32",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cycles": results,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
