#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a year file built by ``build_year_h5.py``. See README.md.

The builder allocates the whole file up front, so its size says nothing about whether it
is finished and unwritten regions read back as ordinary zeros. Three modes, cheapest
first:

completeness (always, and the whole job with ``--quick``)  compare the (variable, day)
    tasks journalled by build_year_h5.py and write_arco_year_h5.py against the ones the
    checked days require.

audit (default)  read every value, catching a task that completed but wrote badly. An
    unwritten block is all zeros; NaN counts as nonzero, so ``sst`` and ``sic`` pass.

``--reference OTHER.h5``  bit-compare against an existing archive year, the only check
    that catches data written to the wrong channel. Compares the leading channels the
    two files share, which requires that they agree on their order.

Both expensive modes report every bad channel, not just the first.
"""

import argparse
import calendar
import concurrent.futures
import datetime
import glob
import json
import os
import time
from pathlib import Path

# HDF5 reads this when the library initializes, which importing h5py does.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from scripts.etl.common.array_job import state_dir_for  # noqa: E402
from scripts.etl.era5.channels import (  # noqa: E402
    HOURS_PER_BLOCK,
    resolve_channels,
    source_groups,
)

# These are NaN over land; they only need some finite ocean.
NAN_ALLOWED = {"sst", "sic"}


def find_sidecar(path: str) -> Path | None:
    # Years sit either directly under the dataset root or in a split subdirectory.
    year_dir = Path(path).resolve().parent
    for root in (year_dir, year_dir.parent):
        sidecar = root / "metadata" / "data.json"
        if sidecar.exists():
            return sidecar
    return None


def channel_names(path: str, spec: str | None, nchan: int) -> list[str]:
    """Names from --channels, else the sidecar the builder wrote.

    Every check keys off these names, so a set that disagrees with the file is an error.
    """
    if spec:
        names = resolve_channels(spec)
        source = f"--channels {spec}"
    else:
        sidecar = find_sidecar(path)
        if sidecar is None:
            year_dir = Path(path).resolve().parent
            raise SystemExit(
                f"no metadata/data.json under {year_dir} or {year_dir.parent};"
                " pass --channels"
            )
        names = json.loads(sidecar.read_text())["coords"]["channel"]
        source = str(sidecar)

    if len(names) != nchan:
        raise SystemExit(f"{source} names {len(names)} channels, file has {nchan}")
    return names


def check_embedded_channels(path: str, sidecar_names: list[str]) -> None:
    with h5py.File(path, "r") as f:
        if "channel" not in f.attrs:
            return
        embedded = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in f.attrs["channel"]
        ]
    if embedded != sidecar_names:
        raise SystemExit(
            f"{path} embedded channels differ from its sidecar: "
            f"{embedded[:3]}... vs {sidecar_names[:3]}..."
        )


def cadence(path: str) -> tuple[int, int]:
    sidecar = find_sidecar(path)
    dhours = 1 if sidecar is None else int(json.loads(sidecar.read_text())["dhours"])
    if HOURS_PER_BLOCK % dhours:
        raise SystemExit(f"{path} has dhours={dhours}, which does not divide 24")
    return dhours, HOURS_PER_BLOCK // dhours


def format_days(days) -> str:
    """Collapse to inclusive spans: ``0-11 15 20-23``."""
    spans = []
    for day in sorted(days):
        if spans and day == spans[-1][1] + 1:
            spans[-1][1] = day
        else:
            spans.append([day, day])
    return " ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def journals(path: str, ndays: int) -> tuple[str, list[tuple[str, int]]]:
    """The build jobs dir, and every jobs dir present with the day count its task ids
    are encoded with, ``group * days + day``: build_year_h5 encodes with the file's day
    count, write_arco_year_h5 with the calendar year's."""
    sidecar = find_sidecar(path)
    root = str(
        Path(path).resolve().parent if sidecar is None else sidecar.parent.parent
    )
    build = state_dir_for(root, path, "jobs")
    arco = state_dir_for(root, path, "arco-jobs")
    found = [(build, ndays)] if os.path.isdir(build) else []
    if os.path.isdir(arco):
        stem = Path(path).stem
        if not stem.isdigit():
            raise SystemExit(
                f"{arco} needs the year, but {path} is not named <year>.h5"
            )
        found.append((arco, 366 if calendar.isleap(int(stem)) else 365))
    return build, found


def check_completeness(
    path: str, ndays: int, days: range, channels: list[str], required: bool
) -> list[int]:
    """Days in ``days`` with an unfinished task, from the writers' jobs dirs."""
    build, found = journals(path, ndays)
    if not found:
        if required:
            raise SystemExit(
                f"no jobs dir at {build}; drop --quick to audit the data instead"
            )
        print(f"  no jobs dir at {build}, cannot check completeness")
        return []

    done: set[tuple[int, int]] = set()
    for jobs_dir, encoded_days in found:
        print(f"  jobs dir: {jobs_dir}")
        for log in glob.glob(os.path.join(jobs_dir, "*.log")):
            with open(log) as f:
                done.update(
                    divmod(int(line), encoded_days) for line in f if line.strip()
                )

    groups = source_groups(channels)
    total = len(groups) * len(days)
    missing = sorted(
        (group, day)
        for group in range(len(groups))
        for day in days
        if (group, day) not in done
    )
    if not missing:
        print(f"  all {total} tasks recorded complete")
        return []

    by_group: dict[str, list[int]] = {}
    for group, day in missing:
        by_group.setdefault(groups[group].raw_name, []).append(day)
    print(f"  {len(missing)} of {total} tasks never completed")
    for group, missing_days in sorted(by_group.items()):
        shown = format_days(missing_days[:40])
        suffix = " ..." if len(missing_days) > 40 else ""
        print(f"    {group}: {len(missing_days)} days -- {shown}{suffix}")
    return sorted({day for _, day in missing})


def audit_day(job):
    path, day, names, steps_per_day = job
    bad = []
    with h5py.File(path, "r") as f:
        fields = f["fields"]
        for hour in range(steps_per_day):
            t = day * steps_per_day + hour
            if t >= fields.shape[0]:
                break
            sample = fields[t]
            empty = ~np.any(sample, axis=(1, 2))
            finite = np.isfinite(sample)
            all_finite = finite.all(axis=(1, 2))
            any_finite = finite.any(axis=(1, 2))
            for channel in range(sample.shape[0]):
                name = names[channel]
                # All-zero first: an unwritten block is zeros, and zeros are finite.
                if empty[channel]:
                    bad.append((t, name, "all-zero"))
                elif name in NAN_ALLOWED:
                    if not any_finite[channel]:
                        bad.append((t, name, "no finite values"))
                elif not all_finite[channel]:
                    bad.append((t, name, "non-finite"))
    return day, bad


def compare_day(job):
    path, reference, day, nchan, names, steps_per_day = job
    bad = []
    with h5py.File(path, "r") as a, h5py.File(reference, "r") as b:
        for hour in range(steps_per_day):
            t = day * steps_per_day + hour
            if t >= a["fields"].shape[0]:
                break
            # Compare bit patterns: NaN != NaN would report every sst and sic as a diff.
            got = a["fields"][t, :nchan].view(np.int32)
            want = b["fields"][t, :nchan].view(np.int32)
            if np.array_equal(got, want):
                continue
            for channel in range(nchan):
                if np.array_equal(got[channel], want[channel]):
                    continue
                delta = np.abs(
                    got[channel].view(np.float32).astype(np.float64)
                    - want[channel].view(np.float32).astype(np.float64)
                )
                finite = delta[np.isfinite(delta)]
                detail = (
                    f"max|diff|={finite.max():g}" if finite.size else "differing NaNs"
                )
                bad.append((t, names[channel], detail))
    return day, bad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="one year file, as built by build_year_h5.py")
    parser.add_argument(
        "--reference",
        default=None,
        help="year file to bit-compare against, over the channels the two share",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="channel set, as build_year_h5.py takes it; defaults to the names in the"
        " sidecar metadata/data.json",
    )
    parser.add_argument("--workers", type=int, default=32, help="reader processes")
    parser.add_argument("--day-start", type=int, default=0, help="first day, inclusive")
    parser.add_argument(
        "--day-stop", type=int, default=None, help="last day, exclusive; default all"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="stop after the jobs-dir check, reading no data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.path, "r") as f:
        nsteps, nchan = f["fields"].shape[:2]
    dhours, steps_per_day = cadence(args.path)
    ndays = nsteps // steps_per_day
    stop = ndays if args.day_stop is None else args.day_stop
    if not 0 <= args.day_start < stop <= ndays:
        raise SystemExit(
            f"--day-start {args.day_start} --day-stop {stop} outside [0, {ndays}]"
        )
    days = range(args.day_start, stop)

    print(f"{args.path}: {nsteps} steps at {dhours}h, {nchan} channels", flush=True)
    names = channel_names(args.path, args.channels, nchan)
    check_embedded_channels(args.path, names)
    incomplete = check_completeness(args.path, ndays, days, names, required=args.quick)
    if args.quick:
        if incomplete:
            print(f"\nINCOMPLETE\nrebuild days: {format_days(incomplete)}")
            raise SystemExit(1)
        print("\nOK: every task recorded complete (no data read)")
        return

    if args.reference:
        with h5py.File(args.reference, "r") as f:
            ref_nchan = f["fields"].shape[1]
        shared = min(nchan, ref_nchan)
        # Comparison is positional, so the two must agree on the channels they share.
        ref_sidecar = find_sidecar(args.reference)
        if ref_sidecar is None:
            print(
                f"  WARNING: {args.reference} has no sidecar, channel order unchecked"
            )
        else:
            ref_names = json.loads(ref_sidecar.read_text())["coords"]["channel"]
            if ref_names[:shared] != names[:shared]:
                raise SystemExit(
                    f"{args.reference} orders its first {shared} channels differently:"
                    f" {ref_names[:shared]} vs {names[:shared]}"
                )
        fn = compare_day
        jobs = [
            (args.path, args.reference, d, shared, names, steps_per_day) for d in days
        ]
        what = (
            f"bit-comparing channels 0-{shared - 1} against {args.reference} "
            f"({nchan} vs {ref_nchan} channels)"
        )
    else:
        fn, jobs = audit_day, [(args.path, d, names, steps_per_day) for d in days]
        what = "auditing for unwritten and non-finite channels"

    print(what, flush=True)
    start = time.perf_counter()
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (day, bad) in enumerate(pool.map(fn, jobs, chunksize=1), 1):
            failures.extend(bad)
            for t, channel, why in bad:
                print(f"  BAD t={t} {channel}: {why}", flush=True)
            if done % 25 == 0 or done == len(jobs):
                print(
                    f"  {done}/{len(jobs)} days, {len(failures)} bad channel-hours "
                    f"({time.perf_counter() - start:.0f}s)",
                    flush=True,
                )

    elapsed = time.perf_counter() - start
    if failures:
        bad_days = sorted({t // steps_per_day for t, _, _ in failures})
        print(
            f"\nFAILED: {len(failures)} bad channel-hours across {len(bad_days)} day(s)"
        )
        print(f"rebuild days: {format_days(bad_days)}")
        raise SystemExit(1)
    print(
        f"\nOK: {len(jobs)} days clean in {elapsed / 60:.1f} min "
        f"({datetime.timedelta(seconds=int(elapsed))})"
    )


if __name__ == "__main__":
    main()
