# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production AMV archive build: full years, one stream per era.

The stream for each year comes from `primary_stream_for_year`.

Resumable: days whose published Parquet validates against the same revisions
are skipped, and the writer publishes atomically. Eras share nothing and can
run on separate nodes.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from healda.config.environment import nnja_archive

CYCLES = ("00", "06", "12", "18")


# The layout pull_amv_bufr.sh mirrors from the NOAA bucket.
def stream_path(stream: str, year: int, month: int, name: str) -> Path:
    return Path(nnja_archive("bufr", stream, f"{year}", f"{month:02d}", "bufr", name))


def days_between(first: datetime, last: datetime):
    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--log-json", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arrow-threads", type=int, default=2)
    ap.add_argument("--decoder-revision", required=True)
    ap.add_argument(
        "--sensor",
        default="satwnd",
        help="Output subdirectory under --output-root; files land in "
        "<output-root>/<sensor>/. The driver appends this itself, so pass "
        "the parent directory as --output-root: `--output-root .../parquet "
        "--sensor satwnd_v2` writes .../parquet/satwnd_v2/. Including the "
        "sensor in --output-root as well doubles it.",
    )
    ap.add_argument(
        "--etl-revision",
        default=None,
        help="Archive content revision, the resume cache key. Defaults to "
        "ETL_CONTENT_REVISION. It is not a build label: changing it "
        "invalidates every published day. The exact commit is recorded "
        "separately as amv.etl_git_sha.",
    )
    args = ap.parse_args()
    if args.start_year > args.end_year:
        ap.error(f"--start-year {args.start_year} is after --end-year {args.end_year}")

    from scripts.etl.nnja.satwnd.archive_etl import (
        ETL_CONTENT_REVISION,
        etl_git_sha,
        primary_stream_for_year,
    )

    if args.etl_revision is None:
        args.etl_revision = ETL_CONTENT_REVISION
    elif args.etl_revision != ETL_CONTENT_REVISION:
        # Resume compares against the revision stamped in the files on disk,
        # so an override can still skip days written under that revision.
        print(
            f"NOTE --etl-revision={args.etl_revision!r} overrides "
            f"ETL_CONTENT_REVISION={ETL_CONTENT_REVISION!r}. Days are resumed "
            f"only where the published footer carries this same value; any "
            f"day stamped differently is rebuilt. Check the first few "
            f"lines say 'skipped_validated' before walking away.",
            flush=True,
        )
    from scripts.etl.nnja.satwnd.amv_writer import (
        AmvDayTask,
        run_days,
    )

    tasks: list[AmvDayTask] = []
    per_year: dict[int, dict] = {}
    for year in range(args.start_year, args.end_year + 1):
        stream = primary_stream_for_year(year)
        counts = {"days": 0, "cycles": 0, "stream": stream}
        for day in days_between(datetime(year, 1, 1), datetime(year, 12, 31)):
            cycle_paths = {}
            for hour in CYCLES:
                name = f"gdas.{day:%Y%m%d}.t{hour}z.satwnd.tm00.bufr_d"
                p = stream_path(stream, year, day.month, name)
                if p.exists():
                    cycle_paths[hour] = str(p)
            if not cycle_paths:
                continue
            counts["days"] += 1
            counts["cycles"] += len(cycle_paths)
            tasks.append(
                AmvDayTask(
                    day=day,
                    cycle_paths=cycle_paths,
                    output_root=args.output_root,
                    sensor=args.sensor,
                    decoder_revision=args.decoder_revision,
                    etl_revision=args.etl_revision,
                )
            )
        per_year[year] = counts

    empty = [
        f"{year} ({stream_path(c['stream'], year, 1, '').parent.parent})"
        for year, c in sorted(per_year.items())
        if not c["days"]
    ]
    if empty:
        raise SystemExit(f"no source cycles on disk for {', '.join(empty)}")

    print(
        f"years={args.start_year}-{args.end_year} days={len(tasks)} "
        f"workers={args.workers}",
        flush=True,
    )
    for year, c in sorted(per_year.items()):
        print(
            f"  {year}: days={c['days']:4d} stream={c['stream']:7s} "
            f"cycles={c['cycles']:5d}",
            flush=True,
        )
    print(f"output_root={args.output_root}", flush=True)
    print(
        f"etl_revision={args.etl_revision} (cache key)  "
        f"git_sha={etl_git_sha()} (provenance)",
        flush=True,
    )

    error_root = Path(args.output_root) / "_errors"
    results_path = Path(args.log_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def report(result: dict, completed: int, total: int) -> None:
        elapsed = time.time() - started
        rate = completed / elapsed if elapsed else 0
        eta = (total - completed) / rate / 3600 if rate else 0
        print(
            f"[{completed}/{total} {elapsed/60:7.1f}m eta={eta:5.2f}h] "
            f"{result.get('status', 'unknown'):24s} "
            f"rows={result.get('total_rows') or 0:>10,} "
            f"parquet={result.get('parquet_bytes', 0)/1e6:7.1f}MB",
            flush=True,
        )
        with results_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {k: v for k, v in result.items() if k != "per_cycle"}, default=str
                )
                + "\n"
            )

    run_days(
        tasks,
        max_workers=args.workers,
        arrow_threads=args.arrow_threads,
        progress=report,
        error_root=error_root,
        continue_on_error=True,
    )
    failed = [
        task.day.strftime("%Y-%m-%d")
        for task in tasks
        if (error_root / f"{task.sensor}.{task.day:%Y%m%d}.error.json").exists()
    ]
    if failed:
        raise SystemExit(
            f"{len(failed)} of {len(tasks)} days failed (markers in {error_root}): "
            f"{', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}"
        )
    print("ETL_DONE", flush=True)


if __name__ == "__main__":
    main()
