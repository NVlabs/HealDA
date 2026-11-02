#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a date range from a local ARCO ERA5 Zarr into an existing year HDF5.

ARCO can carry ERA5T, the preliminary release, so what this produces is a scoring
reference only. The training store comes from the NCAR netCDFs via build_year_h5.py.

See README.md.
"""

import argparse
import calendar
import datetime
import fcntl
import json
import logging
import os
import uuid
from pathlib import Path

import h5py
import numpy as np
import zarr

from scripts.etl.common.array_job import LaunchMode, WorkManager, state_dir_for
from scripts.etl.era5.channels import (
    ARCO_VARIABLE,
    CHAN_BYTES,
    HOURS_PER_BLOCK,
    hour_index,
    parse_channel,
    resolve_channels,
    source_groups,
    write_plan,
)
from scripts.etl.era5.year_h5 import (
    pwrite_all,
    read_sidecar,
    sidecar_path,
    write_h5_attrs,
)
from scripts.etl.common.tasks import task_event_context

logger = logging.getLogger(__name__)

_ARCO_GROUPS: dict[str, zarr.Group] = {}


def arco_group(path: str) -> zarr.Group:
    if path not in _ARCO_GROUPS:
        _ARCO_GROUPS[path] = zarr.open_group(path, mode="r")
    return _ARCO_GROUPS[path]


def read_group(config: dict, group, day: int) -> dict:
    store = arco_group(config["source"])
    variable = ARCO_VARIABLE[group.raw_name]
    array = store[variable]
    date = datetime.datetime(config["year"], 1, 1) + datetime.timedelta(days=day)
    level_indices = (
        [config["level_index"][level] for level in group.levels]
        if group.levels
        else None
    )

    result = {level: [] for level in group.levels or (None,)}
    for step in range(config["steps_per_day"]):
        timestamp = date + datetime.timedelta(hours=step * config["dhours"])
        index = hour_index(timestamp)
        if group.levels:
            slab = array[index, :, :, :]
            for level, level_index in zip(group.levels, level_indices, strict=True):
                result[level].append(slab[level_index])
        else:
            result[None].append(array[index, :, :])
    return {key: np.stack(values) for key, values in result.items()}


def process_task(task_id: int, config: dict) -> None:
    group_index, day = divmod(task_id, config["ndays"])
    group = config["groups"][group_index]
    metadata = {
        "raw_name": group.raw_name,
        "date": str(datetime.date(config["year"], 1, 1) + datetime.timedelta(days=day)),
        "channels": list(group.channel_indices),
    }
    with task_event_context(
        config["events_dir"], task_id=task_id, stage="era5_arco_h5", metadata=metadata
    ):
        data = read_group(config, group, day)
        fd = os.open(config["output"], os.O_WRONLY)
        try:
            for step, offset, keys in write_plan(
                group,
                day,
                config["offset"],
                config["stride"],
                config["steps_per_day"],
            ):
                slab = np.stack([data[key][step] for key in keys]).astype(
                    "<f4", copy=False
                )
                pwrite_all(
                    fd,
                    np.ascontiguousarray(slab).reshape(-1).view(np.uint8).data,
                    offset,
                )
        finally:
            os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="local sparse ARCO Zarr")
    parser.add_argument("--output", required=True, help="existing year HDF5")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--channels", required=True)
    parser.add_argument("--day-start", type=int, required=True)
    parser.add_argument("--day-stop", type=int, required=True)
    parser.add_argument("--dhours", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    return parser.parse_args()


def arco_label(
    start: datetime.date, stop: datetime.date, final_stop: datetime.date | None
) -> str:
    """Name the inclusive range ``start``..``stop``: ARCO serves final ERA5 through
    ``final_stop`` (its ``valid_time_stop``) and ERA5T after it."""
    if final_stop is None:
        return f"ARCO ERA5 or ERA5T {start} through {stop}"
    parts = []
    if start <= final_stop:
        parts.append(f"ARCO ERA5 {start} through {min(stop, final_stop)}")
    if stop > final_stop:
        first_era5t = max(start, final_stop + datetime.timedelta(days=1))
        parts.append(f"ARCO ERA5T {first_era5t} through {stop}")
    return "; ".join(parts)


def update_provenance(args: argparse.Namespace, root: str) -> None:
    path = sidecar_path(root)
    lock_path = path.with_suffix(".lock")
    final_stop = arco_group(args.source).attrs.asdict().get("valid_time_stop")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        document = json.loads(path.read_text())
        start = datetime.date(args.year, 1, 1) + datetime.timedelta(days=args.day_start)
        stop = datetime.date(args.year, 1, 1) + datetime.timedelta(
            days=args.day_stop - 1
        )
        arco_source = arco_label(
            start,
            stop,
            None
            if final_stop is None
            else datetime.date.fromisoformat(str(final_stop)[:10]),
        )
        sources = document["attrs"]["sources"]
        if arco_source not in sources:
            document["attrs"]["sources"] = f"{sources}; {arco_source}"
        document["attrs"]["description"] = (
            f"ERA5 at 0.25 degree, every {args.dhours}h, first snapshot Jan 01 "
            "0000 UTC. The ARCO ranges named in sources were written by "
            "scripts/etl/era5/write_arco_year_h5.py; every other step holds what "
            "the file held before. Channel order differs between sets: select by name."
        )
        scratch = path.with_name(f".data.json.{uuid.uuid4().hex}")
        scratch.write_text(json.dumps(document, indent=2))
        os.replace(scratch, path)
        write_h5_attrs(args.output, document)


def validate_source(
    source: zarr.Group, args: argparse.Namespace, groups
) -> dict[int, int]:
    missing = sorted(
        {ARCO_VARIABLE[group.raw_name] for group in groups} - set(source.array_keys())
    )
    if missing:
        raise SystemExit(f"ARCO source lacks {missing}")

    expected_latitude = np.linspace(90.0, -90.0, 721)
    expected_longitude = np.arange(1440) * 0.25
    if not np.array_equal(source["latitude"][:], expected_latitude):
        raise SystemExit("ARCO latitude does not match the HDF5 grid")
    if not np.array_equal(source["longitude"][:], expected_longitude):
        raise SystemExit("ARCO longitude does not match the HDF5 grid")

    time = source["time"]
    if time.attrs.asdict().get("units") != "hours since 1900-01-01 00:00:00":
        raise SystemExit(f"unsupported ARCO time units: {time.attrs.asdict()}")
    first = hour_index(
        datetime.datetime(args.year, 1, 1) + datetime.timedelta(days=args.day_start)
    )
    last = hour_index(
        datetime.datetime(args.year, 1, 1)
        + datetime.timedelta(days=args.day_stop, hours=-args.dhours)
    )
    if int(time[first]) != first or int(time[last]) != last:
        raise SystemExit("ARCO time coordinate does not cover the requested range")

    levels = source["level"][:]
    wanted_levels = sorted({level for group in groups for level in group.levels})
    missing_levels = sorted(set(wanted_levels) - set(levels.tolist()))
    if missing_levels:
        raise SystemExit(f"ARCO source lacks pressure levels {missing_levels}")
    return {int(level): index for index, level in enumerate(levels)}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    if HOURS_PER_BLOCK % args.dhours:
        raise SystemExit(f"--dhours {args.dhours} does not divide 24")

    channels = resolve_channels(args.channels)
    groups = source_groups(channels)
    unknown = sorted({group.raw_name for group in groups} - ARCO_VARIABLE.keys())
    if unknown:
        raise SystemExit(f"no ARCO mapping for {unknown}")

    ndays = 366 if calendar.isleap(args.year) else 365
    if not 0 <= args.day_start < args.day_stop <= ndays:
        raise SystemExit(
            f"days [{args.day_start},{args.day_stop}) outside year with {ndays} days"
        )
    steps_per_day = HOURS_PER_BLOCK // args.dhours
    root = args.dataset_root or str(Path(args.output).resolve().parent)
    sidecar_channels = read_sidecar(root)["coords"]["channel"]
    if [parse_channel(name) for name in sidecar_channels] != [
        parse_channel(name) for name in channels
    ]:
        raise SystemExit(
            f"{root} channel order {sidecar_channels} does not match {channels}"
        )
    with h5py.File(args.output, "r") as output:
        fields = output["fields"]
        if fields.shape[1] != len(channels):
            raise SystemExit(
                f"{args.output} has {fields.shape[1]} channels, expected {len(channels)}"
            )
        required_steps = args.day_stop * steps_per_day
        if fields.shape[0] < required_steps:
            raise SystemExit(
                f"{args.output} has {fields.shape[0]} steps, needs {required_steps}"
            )
        offset = fields.id.get_offset()
        if offset is None:
            raise SystemExit(f"{args.output}: fields is not contiguous")

    source = arco_group(args.source)
    level_index = validate_source(source, args, groups)

    config = {
        "source": args.source,
        "output": args.output,
        "year": args.year,
        "ndays": ndays,
        "dhours": args.dhours,
        "steps_per_day": steps_per_day,
        "groups": groups,
        "level_index": level_index,
        "offset": offset,
        "stride": len(channels) * CHAN_BYTES,
        "events_dir": state_dir_for(root, args.output, "arco-events"),
    }
    tasks = [
        group_index * ndays + day
        for group_index in range(len(groups))
        for day in range(args.day_start, args.day_stop)
    ]
    manager = WorkManager(
        path=state_dir_for(root, args.output, "arco-jobs"),
        launch_mode=LaunchMode.MANUAL if args.rank is not None else LaunchMode.AUTO,
        rank=args.rank,
        world_size=args.world_size,
    )
    my_tasks = manager.pending(tasks)
    logger.info(
        "rank=%s/%s days=[%s,%s) %s tasks, %s already done",
        manager.rank,
        manager.world_size,
        args.day_start,
        args.day_stop,
        len(my_tasks),
        len(tasks) - len(my_tasks),
    )
    failed = manager.run_parallel(
        process_task, my_tasks, config=config, workers=args.workers
    )
    if failed:
        raise SystemExit(
            f"{len(failed)} of {len(my_tasks)} tasks failed: {sorted(failed)}"
        )
    if set(tasks).issubset(manager.completed_tasks()):
        update_provenance(args, root)
    else:
        logger.info("skipping provenance update until every requested task completes")


if __name__ == "__main__":
    main()
