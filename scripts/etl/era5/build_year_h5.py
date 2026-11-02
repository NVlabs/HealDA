#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ERA5 HDF5 year files from the raw NCAR netCDFs (ds633.0).

One contiguous, uncompressed float32 dataset ``fields`` of shape
``(hours, channels, 721, 1440)`` per year, a pure repack of the raw netCDFs.

Layout rules are in ``scripts.etl.era5.year_h5``.

One task is one source variable for one day, which keeps each task inside a single
source and inflates each pressure chunk once.

Run ``--init-only`` once to create the file, then launch the writers.

The raw files are read from ``--sources``, a TOML file of ``[[source]]`` tables, each
used from its ``start`` date until the next one starts:

    [[source]]
    start = 1940-01-01
    pl = "s3://nsf-ncar-era5/e5.oper.an.pl"    # pressure-level files
    sfc = "s3://nsf-ncar-era5/e5.oper.an.sfc"  # surface files
    monthly = true   # files under <root>/YYYYMM/; false: directly under <root>
    profile = ""     # storage profile (healda.utils.storage); empty: anonymous

Without ``--sources``, everything is read from NCAR's public bucket as above. Remote
files are copied to ``--cache-dir``; local roots are read in place.

This is the training store. ARCO (write_arco_year_h5.py) can carry ERA5T and is
used for scoring references only.
"""

import argparse
import calendar
import datetime
import logging
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.etl.common.array_job import LaunchMode, WorkManager, state_dir_for
from healda.utils.cache import AtomicFileCache
from scripts.etl.era5.channels import (
    CHAN_BYTES,
    HOURS_PER_BLOCK,
    resolve_channels,
    source_groups,
    write_plan,
)
from scripts.etl.era5.year_h5 import (
    cadence_name,
    create_output,
    pwrite_all,
    sidecar_document,
    write_h5_attrs,
    write_sidecar,
)
from scripts.etl.era5.era5_source import Era5FileIndex, read_era5_source_group
from scripts.etl.common.tasks import task_event_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    start: datetime.date
    pl: str
    sfc: str
    monthly: bool = True
    profile: str | None = None

    @property
    def remote(self) -> bool:
        return self.pl.startswith("s3://")


NCAR_PUBLIC = Source(
    start=datetime.date(1940, 1, 1),
    pl="s3://nsf-ncar-era5/e5.oper.an.pl",
    sfc="s3://nsf-ncar-era5/e5.oper.an.sfc",
)


def read_sources(path: str | None) -> list[Source]:
    if path is None:
        return [NCAR_PUBLIC]
    with open(path, "rb") as f:
        tables = tomllib.load(f)["source"]
    sources = []
    for table in tables:
        unknown = set(table) - {"start", "pl", "sfc", "monthly", "profile"}
        if unknown:
            raise SystemExit(f"{path}: unknown source keys {sorted(unknown)}")
        start = table["start"]
        if type(start) is not datetime.date:
            raise SystemExit(f"{path}: start must be a bare date, got {start!r}")
        source = Source(
            start=start,
            pl=table["pl"],
            sfc=table["sfc"],
            monthly=table.get("monthly", True),
            profile=table.get("profile") or None,
        )
        # Roots are s3:// (cached) or local paths (read in place), both alike.
        for root in (source.pl, source.sfc):
            if "://" in root and not root.startswith("s3://"):
                raise SystemExit(f"{path}: {root} is neither s3:// nor a local path")
        if source.remote != source.sfc.startswith("s3://"):
            raise SystemExit(
                f"{path}: pl and sfc of one source are both s3:// or both local"
            )
        sources.append(source)
    return sorted(sources, key=lambda source: source.start)


def source_for(date: datetime.date, sources: list[Source]) -> int:
    """Index of the source that serves ``date``: the last one started by then."""
    started = [i for i, source in enumerate(sources) if source.start <= date]
    if not started:
        raise SystemExit(f"no source starts on or before {date}")
    return started[-1]


def year_days(year: int, archive_end: datetime.date | None) -> int:
    jan1 = datetime.date(year, 1, 1)
    last = datetime.date(year, 12, 31)
    if archive_end is not None:
        if jan1 > archive_end:
            raise SystemExit(f"{year} starts after the archive ends {archive_end}")
        last = min(last, archive_end)
    return (last - jan1).days + 1


def roots_for(sources: list[Source], index: int, year: int, ndays: int) -> list[str]:
    """Monthly sources list one directory per product and month the year reads."""
    source = sources[index]
    if not source.monthly:
        return [source.pl, source.sfc]
    jan1 = datetime.date(year, 1, 1)
    months = sorted(
        {
            (jan1 + datetime.timedelta(days=day)).strftime("%Y%m")
            for day in range(ndays)
            if source_for(jan1 + datetime.timedelta(days=day), sources) == index
        }
    )
    return [
        os.path.join(root, month)
        for month in months
        for root in (source.pl, source.sfc)
    ]


def index_cache_path(cache_dir: str, source: Source, roots: list[str]) -> str:
    """A flat source is one listing for every year; a monthly source's listing covers
    only the months it was made for, so they are in its name."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", source.pl).strip("_")
    if source.monthly:
        months = sorted({os.path.basename(root) for root in roots})
        name = f"{name}_{months[0]}-{months[-1]}"
    return os.path.join(cache_dir, f"era5_index_{name}.csv")


_INDEX_CACHE: dict[int, Era5FileIndex] = {}


def file_index_for(index: int, config: dict) -> Era5FileIndex:
    if index not in _INDEX_CACHE:
        _INDEX_CACHE[index] = Era5FileIndex(
            config["roots"][index],
            profile=config["sources"][index].profile,
            index_cache_path=config["index_cache_paths"][index],
        )
    return _INDEX_CACHE[index]


def write_block(config: dict, group, day: int, data: dict) -> None:
    fd = os.open(config["output"], os.O_WRONLY)
    try:
        plan = write_plan(
            group,
            day,
            config["offset"],
            config["stride"],
            config["steps_per_day"],
        )
        for hour, offset, keys in plan:
            # pwrite copies these bytes verbatim, so spell out the dataset's dtype
            # rather than relying on the host being little endian.
            slab = np.stack([data[key][hour] for key in keys]).astype("<f4", copy=False)
            pwrite_all(
                fd, np.ascontiguousarray(slab).reshape(-1).view(np.uint8).data, offset
            )
    finally:
        os.close(fd)


def process_task(task_id: int, config: dict) -> None:
    group = config["groups"][task_id // config["ndays"]]
    day = task_id % config["ndays"]
    date = datetime.date(config["year"], 1, 1) + datetime.timedelta(days=day)
    index = source_for(date, config["sources"])
    source = config["sources"][index]
    times = pd.date_range(
        pd.Timestamp(date),
        periods=config["steps_per_day"],
        freq=f"{config['dhours']}h",
    )

    metadata = {
        "raw_name": group.raw_name,
        "date": str(date),
        "source": source.pl if group.product == "pl" else source.sfc,
        "channels": list(group.channel_indices),
    }
    with task_event_context(
        config["events_dir"], task_id=task_id, stage="era5_hourly_h5", metadata=metadata
    ):
        file_index = file_index_for(index, config)
        data = read_era5_source_group(
            file_index=file_index,
            raw_name=group.raw_name,
            times=times,
            levels=group.levels,
            cache_dir=config["cache_dir"] if source.remote else None,
            profile=source.profile,
        )
        write_block(config, group, day, data)

        # A daily pressure-level file serves exactly one task, so it is dead weight
        # afterwards. Monthly surface files are shared by the month's other tasks.
        if group.product == "pl" and source.remote and not config["keep_raw"]:
            AtomicFileCache(config["cache_dir"]).discard(
                file_index.resolve_file(group.raw_name, times[0])
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--year", type=int, required=True, help="calendar year to build"
    )
    parser.add_argument("--output", required=True, help="one year's .h5 file")
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="shared raw netCDF cache, safe across processes and nodes",
    )
    parser.add_argument(
        "--sources",
        default=None,
        metavar="TOML",
        help="raw-file sources (see the module docstring); default NCAR's public bucket",
    )
    parser.add_argument(
        "--archive-end",
        type=datetime.date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="last date to allocate and build; default the end of the year. Set it for"
        " a year ERA5 has not finished: every later day would fail for want of files",
    )
    parser.add_argument(
        "--channels",
        default="c75",
        help="preset (c73, c75, sst_sic), comma separated names (sst,sic,z500), or"
        " @file with one name per line",
    )
    parser.add_argument(
        "--dhours",
        type=int,
        choices=(1, 2, 3, 4, 6, 8, 12, 24),
        default=1,
        help="output cadence in hours; default 1",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="name recorded in metadata/data.json; defaults to the channel set",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="where metadata/data.json goes; defaults to the output directory. Pass the"
        " tree root when years sit in a split subdirectory, as in ARCHIVE/train/2018.h5",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="worker processes in this rank"
    )
    parser.add_argument(
        "--day-start", type=int, default=None, help="first day, inclusive; default 0"
    )
    parser.add_argument(
        "--day-stop", type=int, default=None, help="last day, exclusive; default all"
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="keep cached pressure-level netCDFs, which are otherwise discarded once"
        " their one task is done",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="allocate the output file, build the source index, write the sidecar, then"
        " exit; the writers require this to have run",
    )
    parser.add_argument(
        "--rank", type=int, default=None, help="rank id; set to launch outside Slurm"
    )
    parser.add_argument("--world-size", type=int, default=None, help="number of ranks")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    sources = read_sources(args.sources)
    ndays = year_days(args.year, args.archive_end)
    if ndays < (366 if calendar.isleap(args.year) else 365):
        logger.info(
            "%s is partial: %s days, archive ends %s",
            args.year,
            ndays,
            args.archive_end,
        )

    day_start = 0 if args.day_start is None else args.day_start
    day_stop = ndays if args.day_stop is None else args.day_stop
    # Workers pwrite by computed offset, so day >= ndays silently aliases onto the next
    # group's day 0 and overwrites another variable.
    if not 0 <= day_start < day_stop <= ndays:
        raise SystemExit(
            f"--day-start {day_start} --day-stop {day_stop} outside [0, {ndays}]"
        )

    # Setup creates the output, so a multi-rank setup would truncate it once per rank.
    if args.init_only and int(os.environ.get("SLURM_NPROCS", 1)) > 1:
        raise SystemExit("--init-only must run on one rank; SLURM_NPROCS is not 1")

    channels = resolve_channels(args.channels)
    groups = source_groups(channels)
    steps_per_day = HOURS_PER_BLOCK // args.dhours
    offset = create_output(
        args.output, ndays * steps_per_day, len(channels), create=args.init_only
    )
    used = sorted(
        {
            source_for(
                datetime.date(args.year, 1, 1) + datetime.timedelta(days=day), sources
            )
            for day in range(ndays)
        }
    )
    roots = {i: roots_for(sources, i, args.year, ndays) for i in used}
    root = args.dataset_root or str(Path(args.output).resolve().parent)
    config = {
        "year": args.year,
        "ndays": ndays,
        "output": args.output,
        "offset": offset,
        "stride": len(channels) * CHAN_BYTES,
        "dhours": args.dhours,
        "steps_per_day": steps_per_day,
        "groups": groups,
        "cache_dir": args.cache_dir,
        "keep_raw": args.keep_raw,
        "sources": sources,
        "roots": roots,
        "index_cache_paths": {
            i: index_cache_path(args.cache_dir, sources[i], roots[i]) for i in used
        },
        "events_dir": state_dir_for(root, args.output, "events"),
    }
    logger.info(
        "%s channels in %s groups: %s",
        len(channels),
        len(groups),
        ", ".join(g.raw_name for g in groups),
    )

    # Build each source index before the workers fork, otherwise they race to write
    # the same CSV on a shared filesystem.
    for index in used:
        file_index_for(index, config)

    # Checked on every invocation: comparing against the sidecar is the only thing that
    # catches a wave relaunched with the same channels in a different order.
    source_roots = [path for i in used for path in (sources[i].pl, sources[i].sfc)]
    sidecar = dict(
        name=args.dataset_name or f"era5-{cadence_name(args.dhours)}-{args.channels}",
        channels=channels,
        dhours=args.dhours,
        description=(
            f"ERA5 at 0.25 degree, every {args.dhours}h, first snapshot Jan 01 "
            "0000 UTC. Built by "
            "scripts/etl/era5/build_year_h5.py as a pure repack of the "
            "raw NCAR netCDFs. sst and sic, where present, are stored raw and "
            "are NaN over land. Channel order differs between sets: select by name."
        ),
        sources=", ".join(source_roots),
    )
    metadata_path = write_sidecar(root, **sidecar)

    if args.init_only:
        write_h5_attrs(args.output, sidecar_document(**sidecar))
        logger.info(
            "initialized %s (%s hours, offset %s), sources %s, metadata %s",
            args.output,
            ndays * steps_per_day,
            offset,
            source_roots,
            metadata_path,
        )
        return

    tasks = [
        group_index * ndays + day
        for group_index in range(len(groups))
        for day in range(day_start, day_stop)
    ]

    manager = WorkManager(
        path=state_dir_for(root, args.output, "jobs"),
        launch_mode=LaunchMode.MANUAL if args.rank is not None else LaunchMode.AUTO,
        rank=args.rank,
        world_size=args.world_size,
    )
    my_tasks = manager.pending(tasks)
    logger.info(
        "rank=%s/%s days=[%s,%s) %s tasks for me, %s of %s already done globally",
        manager.rank,
        manager.world_size,
        day_start,
        day_stop,
        len(my_tasks),
        len(manager.completed_tasks()),
        len(tasks),
    )

    failed = manager.run_parallel(
        process_task, my_tasks, config=config, workers=args.workers
    )
    if failed:
        raise SystemExit(
            f"{len(failed)} of {len(my_tasks)} tasks failed: {sorted(failed)}"
        )


if __name__ == "__main__":
    main()
