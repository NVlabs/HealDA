#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin an existing ERA5 hourly year file to a coarser cadence. See README.md.

``build_year_h5.py`` builds a year from the raw NCAR netCDFs. This does the cheaper
thing when an hourly year already exists: copy every ``--dhours``-th timestep
straight out of it. A 6-hourly year moves 1/6 of the source bytes and decodes no
netCDF at all, so it costs a fraction of a rebuild and is bit-identical by
construction. Layout rules: ``scripts.etl.era5.year_h5``.

One task is a block of output timesteps. Run ``--init-only`` once to create the
file, then launch the writers.
"""

import argparse
import concurrent.futures
import logging
import os

import numpy as np

from scripts.etl.common.array_job import LaunchMode, WorkManager, state_dir_for
from scripts.etl.era5.channels import CHAN_BYTES, resolve_channels
from scripts.etl.era5.year_h5 import (
    RangeReader,
    cadence_name,
    copy_runs,
    create_output,
    index_runs,
    probe_source,
    read_sidecar,
    sidecar_document,
    write_h5_attrs,
    write_sidecar,
)
from scripts.etl.common.tasks import task_event_context

logger = logging.getLogger(__name__)


def step_bases(step: int, config: dict) -> tuple[int, int]:
    """Byte offsets of one output step in the source and in the output."""
    src = config["src_offset"] + step * config["stride_source"] * config["src_stride"]
    dst = config["out_offset"] + step * config["out_stride"]
    return src, dst


def process_task(task_id: int, config: dict) -> None:
    first = task_id * config["steps_per_task"]
    last = min(first + config["steps_per_task"], config["nsteps"])
    metadata = {"steps": [first, last], "source": config["source"]}

    with task_event_context(
        config["events_dir"], task_id=task_id, stage="era5_thin_h5", metadata=metadata
    ):
        fd = os.open(config["output"], os.O_WRONLY)
        try:
            with RangeReader(config["source"], config["source_profile"]) as reader:

                def copy(step: int) -> None:
                    src_base, dst_base = step_bases(step, config)
                    copy_runs(
                        reader,
                        fd,
                        runs=config["runs"],
                        src_base=src_base,
                        dst_base=dst_base,
                    )

                # Threads, not processes: every step is a blocking ranged GET, and
                # they pwrite disjoint ranges of the same descriptor.
                with concurrent.futures.ThreadPoolExecutor(
                    config["io_threads"]
                ) as pool:
                    # list() so an exception in any step propagates instead of being
                    # dropped with the unconsumed iterator.
                    list(pool.map(copy, range(first, last)))
        finally:
            os.close(fd)


def verify(config: dict, samples: int) -> None:
    """Check sampled output steps are byte-identical to the source steps they came from."""
    reader = RangeReader(config["source"], config["source_profile"])
    local = RangeReader(config["output"])
    rng = np.random.default_rng(0)
    steps = sorted(
        set(
            rng.integers(
                0, config["nsteps"], size=min(samples, config["nsteps"])
            ).tolist()
        )
        | {0, config["nsteps"] - 1}
    )

    bad = []
    for step in steps:
        src_base, dst_base = step_bases(step, config)
        for out_channel, src_channel, count in config["runs"]:
            length = count * CHAN_BYTES
            want = reader.read(src_base + src_channel * CHAN_BYTES, length)
            got = local.read(dst_base + out_channel * CHAN_BYTES, length)
            if want != got:
                bad.append(step)
                break
    if bad:
        raise SystemExit(
            f"verify FAILED: {len(bad)} of {len(steps)} steps differ: {bad}"
        )
    logger.info("verify OK: %s sampled steps byte-identical to source", len(steps))


def resolve_output_channels(
    args: argparse.Namespace, src_nchan: int
) -> tuple[list[int], list[str]]:
    """Source channel indices to keep, and their names (from the source sidecar)."""
    available = read_sidecar(args.source_root, args.source_profile)["coords"]["channel"]
    if len(available) != src_nchan:
        raise SystemExit(
            f"{args.source_root} sidecar lists {len(available)} channels but the"
            f" source array has {src_nchan}"
        )

    if not args.channels:
        return list(range(src_nchan)), list(available)

    wanted = resolve_channels(args.channels)
    missing = [name for name in wanted if name not in available]
    if missing:
        raise SystemExit(f"source lacks {missing}")
    return [available.index(name) for name in wanted], wanted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        required=True,
        help="hourly year file: a local path, or s3://bucket/key with --source-profile",
    )
    parser.add_argument(
        "--source-profile",
        default=None,
        help="storage profile (healda.utils.storage) of an s3:// source",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="source dataset root holding metadata/data.json; channel names come from"
        " it, so the output sidecar cannot be written without it",
    )
    parser.add_argument("--output", required=True, help="one year's thinned .h5 file")
    parser.add_argument(
        "--dhours",
        type=int,
        default=6,
        help="target cadence in hours: 6 keeps 00/06/12/18 UTC. Default 6",
    )
    parser.add_argument(
        "--source-dhours",
        type=int,
        default=1,
        help="cadence of the source; the stride taken is --dhours/--source-dhours",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="subset to keep, as a preset/comma list/@file. Default: every source channel",
    )
    parser.add_argument(
        "--dataset-name", default=None, help="name in metadata/data.json"
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="where metadata/data.json goes; defaults to the output directory. Pass the"
        " tree root when years sit in a split subdirectory, as in ARCHIVE/train/2019.h5",
    )
    parser.add_argument(
        "--steps-per-task", type=int, default=24, help="output timesteps per task"
    )
    parser.add_argument(
        "--output-steps",
        type=int,
        default=None,
        help="allocated output length; defaults to the copied source length",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="worker processes in this rank"
    )
    parser.add_argument(
        "--io-threads", type=int, default=8, help="concurrent ranged reads per worker"
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="allocate the output and write the sidecar, then exit; writers need this",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        help="after writing, compare N sampled steps byte for byte against the source",
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

    if args.dhours % args.source_dhours:
        raise SystemExit(
            f"--dhours {args.dhours} is not a multiple of --source-dhours"
            f" {args.source_dhours}; thinning can only drop whole timesteps"
        )
    stride_source = args.dhours // args.source_dhours

    info = probe_source(args.source, args.source_profile)
    src_ntime, src_nchan = info["shape"][0], info["shape"][1]
    nsteps = len(range(0, src_ntime, stride_source))
    output_steps = args.output_steps or nsteps
    if output_steps < nsteps:
        raise SystemExit(
            f"--output-steps {output_steps} is shorter than {nsteps} copied steps"
        )
    indices, channels = resolve_output_channels(args, src_nchan)
    source_metadata = read_sidecar(args.source_root, args.source_profile)
    source_provenance = source_metadata.get("attrs", {}).get("sources")

    # Setup creates the output, so a multi-rank setup would truncate it once per rank.
    if args.init_only and int(os.environ.get("SLURM_NPROCS", 1)) > 1:
        raise SystemExit("--init-only must run on one rank; SLURM_NPROCS is not 1")

    out_offset = create_output(
        args.output, output_steps, len(indices), create=args.init_only
    )
    root = args.dataset_root or str(os.path.dirname(os.path.abspath(args.output)))
    config = {
        "source": args.source,
        "source_profile": args.source_profile,
        "output": args.output,
        "src_offset": info["offset"],
        "out_offset": out_offset,
        "src_stride": src_nchan * CHAN_BYTES,
        "out_stride": len(indices) * CHAN_BYTES,
        "stride_source": stride_source,
        "runs": index_runs(indices),
        "nsteps": nsteps,
        "steps_per_task": args.steps_per_task,
        "io_threads": args.io_threads,
        "events_dir": state_dir_for(root, args.output, "thin-events"),
    }
    logger.info(
        "%s (%s steps x %s ch) -> %s (%s/%s steps x %s ch), stride %s, %s byte run(s)/step",
        args.source,
        src_ntime,
        src_nchan,
        args.output,
        nsteps,
        output_steps,
        len(indices),
        stride_source,
        len(config["runs"]),
    )

    sidecar = dict(
        name=args.dataset_name or f"era5-{cadence_name(args.dhours)}",
        channels=channels,
        dhours=args.dhours,
        description=(
            f"ERA5 at 0.25 degree, every {args.dhours}h, first snapshot Jan 01 0000 UTC."
            " Built by scripts/etl/era5/thin_year_h5.py by copying every"
            f" {stride_source}th timestep out of the source store, so every value is"
            " bit-identical to it. Channel order differs between sets: select by name."
        ),
        sources=source_provenance or f"thinned from {args.source}",
    )
    metadata_path = write_sidecar(root, **sidecar)

    if args.init_only:
        # Mirror the sidecar onto the file, as the source packs do, so a year stays
        # readable if it is moved away from its root.
        write_h5_attrs(args.output, sidecar_document(**sidecar))
        logger.info(
            "initialized %s (%s steps, offset %s), metadata %s",
            args.output,
            output_steps,
            out_offset,
            metadata_path,
        )
        return

    ntasks = -(-nsteps // args.steps_per_task)
    manager = WorkManager(
        path=state_dir_for(root, args.output, "thin-jobs"),
        launch_mode=LaunchMode.MANUAL if args.rank is not None else LaunchMode.AUTO,
        rank=args.rank,
        world_size=args.world_size,
    )
    my_tasks = manager.pending(list(range(ntasks)))
    logger.info(
        "rank=%s/%s %s tasks for me, %s of %s already done globally",
        manager.rank,
        manager.world_size,
        len(my_tasks),
        len(manager.completed_tasks()),
        ntasks,
    )

    failed = manager.run_parallel(
        process_task, my_tasks, config=config, workers=args.workers
    )
    if failed:
        raise SystemExit(
            f"{len(failed)} of {len(my_tasks)} tasks failed: {sorted(failed)}"
        )

    # Sampled steps span the whole output, so verifying before every rank has
    # journalled would read another rank's not-yet-written bytes.
    if args.verify and len(manager.completed_tasks()) == ntasks:
        verify(config, args.verify)
    elif args.verify:
        logger.info(
            "rank=%s done; skipping verify, other ranks still writing", manager.rank
        )


if __name__ == "__main__":
    main()
