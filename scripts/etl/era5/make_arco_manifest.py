#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create an rclone --files-from manifest for an ARCO ERA5 subset at a given cadence."""

import argparse
import datetime
import itertools
import json
import math
from pathlib import Path

from scripts.etl.era5.channels import (
    ARCO_VARIABLE,
    HOURS_PER_BLOCK,
    hour_index,
    resolve_channels,
    source_groups,
)

COORDINATES = ("latitude", "longitude", "level", "time")


def chunk_keys(name: str, metadata: dict, hours: range) -> list[str]:
    array = metadata[f"{name}/.zarray"]
    attrs = metadata[f"{name}/.zattrs"]
    ranges = []
    for dimension, size, chunk in zip(
        attrs["_ARRAY_DIMENSIONS"], array["shape"], array["chunks"], strict=True
    ):
        if dimension == "time":
            ranges.append(sorted({hour // chunk for hour in hours}))
        else:
            ranges.append(range(math.ceil(size / chunk)))
    return [
        f"{name}/" + ".".join(str(index) for index in indices)
        for indices in itertools.product(*ranges)
    ]


def parse_timestamp(value: str) -> datetime.datetime:
    timestamp = datetime.datetime.fromisoformat(value)
    if timestamp.minute or timestamp.second or timestamp.microsecond:
        raise argparse.ArgumentTypeError(f"{value!r} is not hour-aligned")
    return timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="local copy of the source Zarr .zmetadata",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=parse_timestamp, required=True)
    parser.add_argument("--stop", type=parse_timestamp, required=True)
    parser.add_argument(
        "--dhours",
        type=int,
        default=1,
        help="cadence in hours: list the chunks of every --dhours-th hour from --start,"
        " as write_arco_year_h5 --dhours reads them. Default 1",
    )
    parser.add_argument(
        "--channels",
        action="append",
        required=True,
        help="channel preset/list; repeat to combine sets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop <= args.start:
        raise SystemExit("--stop must be later than --start")
    if HOURS_PER_BLOCK % args.dhours or args.start.hour % args.dhours:
        raise SystemExit(
            f"--dhours {args.dhours} must divide 24 and the hour of --start"
        )

    document = json.loads(args.metadata.read_text())
    metadata = document["metadata"]
    channel_names = [name for spec in args.channels for name in resolve_channels(spec)]
    raw_names = {group.raw_name for group in source_groups(channel_names)}
    missing = sorted(raw_names - ARCO_VARIABLE.keys())
    if missing:
        raise SystemExit(f"no ARCO mapping for {missing}")
    variables = sorted(ARCO_VARIABLE[name] for name in raw_names)

    hours = range(hour_index(args.start), hour_index(args.stop), args.dhours)
    files = [".zattrs", ".zgroup", ".zmetadata"]
    for name in (*COORDINATES, *variables):
        for suffix in (".zarray", ".zattrs"):
            key = f"{name}/{suffix}"
            if key not in metadata:
                raise SystemExit(f"{args.metadata} lacks {key}")
            files.append(key)
        files.extend(chunk_keys(name, metadata, hours))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(dict.fromkeys(files)) + "\n")
    print(f"{args.output}: {len(set(files))} files for {len(variables)} variables")


if __name__ == "__main__":
    main()
