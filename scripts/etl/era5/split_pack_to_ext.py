#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reslice a multi-channel ERA5 hourly pack into one self-contained file per variable.

Reads ``<pack_dir>/<year>.h5`` (channel axis interleaved) plus its
``metadata/data.json`` sidecar, and writes ``<ext_root>/<channel>/<year>.h5`` per
channel: a single-channel contiguous dataset carrying its own coordinate and
channel-name attributes (readable standalone, no sidecar needed), plus a
``<ext_root>/<channel>/metadata/data.json`` sidecar so the directory is also a
drop-in pack for ``healda.datasets.era5_hourly_h5.Era5HourlyH5``.

Idempotent and resumable: a channel-year already present with the right shape is
skipped, so a killed or re-run job redoes only what is missing. Verifies each
channel-year against the source with a sample of hours before moving on. Never
touches or deletes the source pack.

One process per (channel, year) task via ``--workers``; different tasks always write
different files, so it is safe to parallelize. Do not run two invocations against the
same ``--pack-dir``/``--ext-root`` pair at once — both would target the same scratch
file for a given (channel, year) and can corrupt each other's write.
"""

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path

import h5py
import numpy as np

from scripts.etl.era5.year_h5 import read_sidecar, write_sidecar

logger = logging.getLogger(__name__)

HOURS_CHUNK = 500
VERIFY_SAMPLE_HOURS = 20


def write_variable_sidecar(
    var_dir: Path, name: str, coords: dict, source_desc: str, dhours: int
) -> None:
    write_sidecar(
        var_dir,
        name=f"era5-{dhours}h-{name}",
        channels=[name],
        dhours=dhours,
        description=(
            f"ERA5 {name} at 0.25 degree, every {dhours}h, first snapshot Jan 01"
            " 0000 UTC. Split from a multi-channel pack by"
            " scripts/etl/era5/split_pack_to_ext.py."
        ),
        sources=source_desc,
        coords={
            "grid_type": coords["grid_type"],
            "lat": list(coords["lat"]),
            "lon": list(coords["lon"]),
            "channel": [name],
        },
    )


def write_variable_year(
    src: h5py.File,
    channel_index: int,
    channel_name: str,
    out_path: Path,
    coords: dict,
    source_desc: str,
    dhours: int,
) -> None:
    nhours, _, nlat, nlon = src["fields"].shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = out_path.with_name(f"{out_path.stem}.{os.getpid()}.tmp")
    with h5py.File(scratch, "w") as out:
        dataset = out.create_dataset(
            "fields", shape=(nhours, 1, nlat, nlon), dtype="<f4"
        )
        for start in range(0, nhours, HOURS_CHUNK):
            stop = min(start + HOURS_CHUNK, nhours)
            block = src["fields"][start:stop, channel_index : channel_index + 1, :, :]
            dataset[start:stop] = block
        out.attrs["dataset_name"] = f"era5-{dhours}h-{channel_name}"
        out.attrs["description"] = (
            f"ERA5 {channel_name} at 0.25 degree, every {dhours}h, first snapshot Jan 01 0000"
            " UTC. Self-contained single-variable file: no metadata/data.json sidecar"
            " needed. Split from a multi-channel pack by"
            " scripts/etl/era5/split_pack_to_ext.py."
        )
        out.attrs["sources"] = source_desc
        out.attrs["h5_path"] = "fields"
        out.attrs["dims"] = ["time", "channel", "lat", "lon"]
        out.attrs["dhours"] = dhours
        out.attrs["grid_type"] = coords["grid_type"]
        out.attrs["lat"] = np.asarray(coords["lat"], dtype="f8")
        out.attrs["lon"] = np.asarray(coords["lon"], dtype="f8")
        out.attrs["channel"] = [channel_name]
    scratch.rename(out_path)


def verify_variable_year(
    src: h5py.File, channel_index: int, out_path: Path, rng: np.random.Generator
) -> None:
    nhours = src["fields"].shape[0]
    sample = rng.choice(nhours, size=min(VERIFY_SAMPLE_HOURS, nhours), replace=False)
    sample.sort()
    with h5py.File(out_path, "r") as out:
        for hour in sample:
            expected = src["fields"][hour, channel_index]
            actual = out["fields"][hour, 0]
            if not np.array_equal(expected, actual, equal_nan=True):
                raise SystemExit(
                    f"{out_path} hour {hour} mismatches source channel {channel_index}"
                )


def already_split(
    out_path: Path, expected_shape: tuple[int, int, int, int], name: str, dhours: int
) -> bool:
    if not out_path.exists():
        return False
    with h5py.File(out_path, "r") as existing:
        dataset = existing["fields"]
        if dataset.shape != expected_shape or dataset.dtype != np.dtype("<f4"):
            return False
        if existing.attrs.get("dhours") != dhours:
            return False
        return list(existing.attrs.get("channel", [])) == [name]


def split_one(
    pack_dir: str,
    ext_root: str,
    year: int,
    channel_index: int,
    name: str,
    metadata: dict,
) -> str:
    src_path = Path(pack_dir) / f"{year}.h5"
    coords = metadata["coords"]
    source_desc = metadata["attrs"]["sources"]
    var_dir = Path(ext_root) / name
    out_path = var_dir / f"{year}.h5"

    dhours = metadata.get("dhours", 1)

    with h5py.File(src_path, "r") as src:
        write_variable_sidecar(var_dir, name, coords, source_desc, dhours)
        nhours, _, nlat, nlon = src["fields"].shape
        if already_split(out_path, (nhours, 1, nlat, nlon), name, dhours):
            return f"{out_path}: already split, skipping"
        write_variable_year(
            src, channel_index, name, out_path, coords, source_desc, dhours
        )
        verify_variable_year(
            src, channel_index, out_path, np.random.default_rng(seed=year)
        )
    return f"{out_path}: split and verified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pack-dir", required=True, help="source multi-channel pack directory"
    )
    parser.add_argument(
        "--ext-root", required=True, help="output root, e.g. .../era5_hourly_ext"
    )
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-stop", type=int, required=True, help="inclusive")
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel (channel, year) tasks"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    pack_dir = Path(args.pack_dir)
    metadata = read_sidecar(pack_dir)
    channels = metadata["coords"]["channel"]

    tasks = [
        (str(pack_dir), args.ext_root, year, index, name, metadata)
        for year in range(args.year_start, args.year_stop + 1)
        if (pack_dir / f"{year}.h5").exists()
        for index, name in enumerate(channels)
    ]
    logger.info(
        "%s: %s (channel, year) tasks, %s workers", pack_dir, len(tasks), args.workers
    )

    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(split_one, *task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            try:
                logger.info("[%s/%s] %s", i, len(tasks), future.result())
            except Exception:
                logger.exception(
                    "[%s/%s] failed: pack=%s year=%s channel=%s",
                    i,
                    len(tasks),
                    task[0],
                    task[2],
                    task[4],
                )
                failed += 1

    if failed:
        raise SystemExit(f"{failed} of {len(tasks)} tasks failed")
    logger.info(
        "done: %s [%s-%s] -> %s",
        pack_dir,
        args.year_start,
        args.year_stop,
        args.ext_root,
    )


if __name__ == "__main__":
    main()
