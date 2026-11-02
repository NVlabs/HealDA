# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the channel-thinned derivative of the wide NNJA IR archive.

One Parquet file per sensor-day at `<archive root>/<preset>/<sensor>/YYYYMMDD.parquet`, with the
archive's own layout and column names, which is where `NNJAWideLoader._sensor_root` looks and
what lets it read a thinned day with no change. Rows are untouched: this narrows the channel
axis and drops columns.

Channels come from an `ir_spectral` preset, ranked in `ir_channel_ranking.csv`. At `ir48`:

    sensor  kept  published  channel ids
    CrIS      48        431  53 to 1570
    IASI      48        616  74 to 5403
    AIRS      48        281  7 to 1843

Each file records its preset and channel numbers in schema metadata. Which footprint columns
are kept is `footprint_columns`: the required footprint columns of the reader schema,
`schema.footprint_fields`, imported rather than restated so the two cannot drift, plus the QC
columns named below. Dropped are the nested source groups, the per-channel auxiliaries, and the
satellite and solar azimuth angles, whose two high-entropy float columns cost 4% of a thinned
day and which nothing reads yet. CrIS channels stay integer BUFR codes, with the
`value_encoding` metadata that reconstructs radiance. CrIS also gains `fov_quality_bad`, the
loader's provider flag, derived from the nested `band_calibration_quality.fov_quality`.

Physical layout is the archive's own, from `archive_writer.writer_options`: ZSTD-3,
DELTA_BINARY_PACKED on CrIS's integer codes, dictionary pages on the float32 channels, one
row group per DA window, statistics on `da_window` alone. A sweep over ten codecs found
nothing better, so a thinned column chunk is near a byte copy of the
source's. Stripping the rest of the schema metadata takes the footer from 2.44 MB to 85 KB.

At `ir48` the three sounders go from 40.4 TB to 4.69 TB: CrIS 17.7 to 1.97, IASI 15.0 to
1.28, AIRS 7.7 to 1.44.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from healda.config.environment import nnja_archive
from healda.observations import schema as reader_schema
from healda.observations.loaders.nnja_wide import (
    CRIS_FOV_QUALITY_BAD,
    IR_SOUNDERS,
    SENSOR_VALUE_PREFIX,
)
from healda.observations.preprocessing import ir_spectral

DEFAULT_PRESET = "ir48"

# Carried for QC though the loader reads none of them, and cheap because each is
# near-constant along a scan: 2.5% of a CrIS day, 0.2% of IASI, 0.6% of AIRS. The three
# source_* columns rejoin a thinned row to its published archive row. Not here:
# instrument_id, which is one value per sensor and so restates the directory, the varying
# identity being platform_id, which the loader requires anyway.
QC_COLUMNS = (
    "scan_line",
    "orbit_number",
    "source_file_ordinal",
    "source_message_index",
    "source_subset_index",
)

# Flags and scene descriptors that vary, per sensor. The counts differ because what each
# sensor publishes differs: CrIS alone publishes cloud and surface fields. Dropped as constant
# on every day sampled are IASI's system_quality_function, spectral_calibration_quality,
# radiometric_calibration_quality, instrument_noise_quality and avhrr_channel_combination_flags.
# AIRS's only per-value flag is the nested acquisition_quality, which is not sliced out here.
SENSOR_QC_COLUMNS = {
    "cris": (
        "scan_quality",
        "geolocation_quality",
        "quality_mark",
        "radiance_type_flags",
        "track_qualifier",
        "land_sea_qualifier",
        "land_fraction",
        "total_cloud_cover",
        "cloud_top_height",
        "surface_height",
    ),
    "iasi": ("summary_quality", "geometric_quality"),
    "airs": ("total_cloud_cover",),
}

CRIS_BANDS_COLUMN = "band_calibration_quality"
FOV_QUALITY_BAD = pa.field(
    "fov_quality_bad",
    pa.schema(reader_schema.footprint_fields("cris")).field("fov_quality_bad").type,
).with_metadata(
    {b"description": b"1 if bit 14 or 15 (LSB=0) of fov_quality is set in any band"}
)

# The one column whose statistics are load bearing: the loader prunes row groups by it.
WINDOW_COLUMN = "da_window"

VALUE_ENCODING_KEY = b"nnja.archive.value_encoding"
PRESET_KEY = b"nnja.thinned.preset"
CHANNELS_KEY = b"nnja.thinned.channels"


# --- write policy ----------------------------------------------------------


def writer_options(
    value_columns: list[str], other_columns: list[str], value_type: pa.DataType
) -> dict:
    options = {
        "compression": "zstd",
        "compression_level": 3,
        "data_page_version": "2.0",
        "write_statistics": [WINDOW_COLUMN],
    }
    if not pa.types.is_integer(value_type):
        options["use_dictionary"] = True
        return options
    # Delta suits CrIS's codes, a fixed scale over a 22-bit field keeping them in a narrow
    # correlated range. It is not right for integer channels in general: the archive measured
    # it on IASI's uint16 codes and rejected it at 1.08x, CHSF filling the field and leaving
    # the differences full entropy. Naming the footprint columns, which do want dictionary
    # pages, is how PyArrow allows a per-column encoding alongside them.
    options["use_dictionary"] = list(other_columns)
    options["column_encoding"] = {name: "DELTA_BINARY_PACKED" for name in value_columns}
    return options


# --- column selection ------------------------------------------------------


def channel_columns(sensor: str, published: set[str], preset: str) -> list[str]:
    prefix = f"{SENSOR_VALUE_PREFIX[sensor]}__ch_"
    wanted = sorted(ir_spectral.ir_channel_preset(preset)[sensor])
    columns = [f"{prefix}{channel:05d}" for channel in wanted]
    # Dropping a wanted channel in silence would narrow the model's input width later.
    missing = [name for name in columns if name not in published]
    if missing:
        raise ValueError(
            f"{sensor}: {len(missing)} of {len(columns)} {preset} channels are not "
            f"published, starting {missing[:4]}"
        )
    return columns


def footprint_columns(sensor: str, published: set[str]) -> list[str]:
    """The loader's columns, required, then whichever QC columns this day published.

    The QC extras are best effort because the source schema drifts over the 25 years the
    archive spans, and a day predating a flag should still be thinnable.
    """
    required = list(
        reader_schema.required_columns(
            pa.schema(reader_schema.footprint_fields(sensor))
        )
    )
    missing = [name for name in required if name not in published]
    if missing:
        raise ValueError(f"{sensor}: the loader's columns are not published: {missing}")
    extra = [
        name for name in (*QC_COLUMNS, *SENSOR_QC_COLUMNS[sensor]) if name in published
    ]
    return [*required, *extra]


def output_schema(
    source_schema: pa.Schema, columns: list[str], sensor: str, preset: str
) -> pa.Schema:
    """Standard Arrow metadata, plus what was kept and how CrIS decodes.

    The `nnja.archive.*` keys dropped here describe BUFR containers and source structures
    this derivative no longer publishes, and are most of the source footer. `value_encoding`
    stays because a CrIS integer code column cannot reconstruct to radiance without it.
    """
    fields = []
    for name in columns:
        field = source_schema.field(name)
        encoding = (field.metadata or {}).get(VALUE_ENCODING_KEY)
        fields.append(
            field.remove_metadata()
            if not encoding or encoding == b"null"
            else field.with_metadata({VALUE_ENCODING_KEY: encoding})
        )
    channels = sorted(ir_spectral.ir_channel_preset(preset)[sensor])
    return pa.schema(
        fields,
        metadata={
            PRESET_KEY: preset.encode(),
            CHANNELS_KEY: ",".join(str(channel) for channel in channels).encode(),
        },
    )


# --- thin one day ----------------------------------------------------------


@dataclass
class Projection:
    columns: list[str]  # read; CRIS_BANDS_COLUMN is replaced by FOV_QUALITY_BAD
    value_columns: list[str]
    other_columns: list[str]  # written
    value_type: pa.DataType
    schema: pa.Schema


def projection(source: pq.ParquetFile, sensor: str, preset: str) -> Projection:
    schema = source.schema_arrow
    published = set(schema.names)
    value_columns = channel_columns(sensor, published, preset)
    other_columns = footprint_columns(sensor, published)
    output = output_schema(schema, [*other_columns, *value_columns], sensor, preset)
    bands = (
        [CRIS_BANDS_COLUMN]
        if sensor == "cris" and CRIS_BANDS_COLUMN in published
        else []
    )
    if bands:
        output = output.insert(len(other_columns), FOV_QUALITY_BAD)
    return Projection(
        columns=[*other_columns, *bands, *value_columns],
        value_columns=value_columns,
        other_columns=[*other_columns, *(FOV_QUALITY_BAD.name for _ in bands)],
        value_type=schema.field(value_columns[0]).type,
        schema=output,
    )


def with_fov_quality_bad(window: pa.Table) -> pa.Table:
    bands = window[CRIS_BANDS_COLUMN].combine_chunks()
    fov = pc.struct_field(pc.list_flatten(bands), "fov_quality")
    bits = pc.fill_null(fov, 0).to_numpy(zero_copy_only=False).astype(np.uint32)
    bad = np.zeros(len(bands), np.uint8)
    bad[
        pc.list_parent_indices(bands).to_numpy()[(bits & CRIS_FOV_QUALITY_BAD) != 0]
    ] = 1
    index = window.schema.get_field_index(CRIS_BANDS_COLUMN)
    return window.set_column(index, FOV_QUALITY_BAD, pa.array(bad))


def prefetched_windows(
    source: pq.ParquetFile, columns: list[str], depth: int = 1
) -> Iterator[pa.Table]:
    """Projected row groups, fetching the next while the caller encodes the current one.

    `write_table` releases the GIL, so the overlap is real, worth 0.5 s of a 7 s CrIS day.
    One pool thread keeps reads off each other, and `depth + 1` windows are resident.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending: deque = deque()
        for group in range(source.num_row_groups):
            pending.append(pool.submit(source.read_row_group, group, columns=columns))
            if len(pending) > depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def write_windows(
    path: Path, windows: Iterable[pa.Table], schema: pa.Schema, options: dict
) -> None:
    """Publish by rename, so a reader never sees a half-written day."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with pq.ParquetWriter(temporary, schema, **options) as writer:
            for window in windows:
                writer.write_table(window, row_group_size=window.num_rows)
        # The rename is atomic against a crash only once the data is durable.
        handle = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def thin_day(
    sensor: str, day: str, preset: str, source_root: str, output_root: str
) -> dict:
    source_path = os.path.join(source_root, sensor, f"{day}.parquet")
    output_path = Path(output_root) / sensor / f"{day}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = pq.ParquetFile(source_path, pre_buffer=True)
    plan = projection(source, sensor, preset)
    options = writer_options(plan.value_columns, plan.other_columns, plan.value_type)
    windows = prefetched_windows(source, plan.columns)
    if CRIS_BANDS_COLUMN in plan.columns:
        windows = map(with_fov_quality_bad, windows)
    write_windows(output_path, windows, plan.schema, options)
    return {
        "sensor": sensor,
        "day": day,
        "source_bytes": os.path.getsize(source_path),
        "bytes": os.path.getsize(output_path),
        "seconds": time.perf_counter() - started,
    }


# --- entry point -----------------------------------------------------------


def published_days(sensor: str, source_root: str) -> list[str]:
    # A sensor that was never decoded, e.g. AIRS after its record ends, has nothing to thin.
    if not os.path.isdir(os.path.join(source_root, sensor)):
        return []
    with os.scandir(os.path.join(source_root, sensor)) as entries:
        return [
            entry.name[: -len(".parquet")]
            for entry in entries
            if entry.name.endswith(".parquet")
        ]


def day_argument(value: str) -> str:
    datetime.strptime(value, "%Y%m%d")
    return value


def _thin_day(work):
    return thin_day(*work)


def pending_work(arguments: argparse.Namespace) -> list[tuple]:
    work = []
    for sensor in arguments.sensors:
        for day in sorted(published_days(sensor, arguments.source_root)):
            if not arguments.start <= day <= arguments.end:
                continue
            if not arguments.overwrite and os.path.exists(
                os.path.join(arguments.output_root, sensor, f"{day}.parquet")
            ):
                continue
            work.append(
                (
                    sensor,
                    day,
                    arguments.preset,
                    arguments.source_root,
                    arguments.output_root,
                )
            )
    return work


def run(arguments: argparse.Namespace) -> None:
    work = pending_work(arguments)
    print(f"{len(work)} sensor-days to write into {arguments.output_root}", flush=True)
    written = source = 0
    started = time.perf_counter()
    with Pool(arguments.workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_thin_day, work), start=1):
            written += result["bytes"]
            source += result["source_bytes"]
            print(
                f"  [{done}/{len(work)}] {result['sensor']} {result['day']} "
                f"{result['bytes'] / 2**20:.0f} MiB "
                f"({result['bytes'] / result['source_bytes']:.1%}) "
                f"in {result['seconds']:.1f} s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(
        f"wrote {written / 2**30:.2f} GiB from {source / 2**30:.2f} GiB "
        f"({written / max(source, 1):.1%}) in {elapsed / 60:.1f} min"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensors", nargs="+", default=list(IR_SOUNDERS), choices=IR_SOUNDERS
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        choices=sorted(ir_spectral.IR_CHANNEL_PRESETS),
    )
    parser.add_argument("--source-root", default=nnja_archive("parquet"))
    parser.add_argument(
        "--output-root",
        help="Defaults to <source-root>/<preset>, where the loader discovers a store.",
    )
    parser.add_argument("--start", type=day_argument, default="19980101")
    parser.add_argument(
        "--end", type=day_argument, default=date.today().strftime("%Y%m%d")
    )
    # ParquetWriter encodes on the calling thread, so a day is a few seconds of one core and
    # throughput comes from days in parallel. Above about 64 workers a day is bound by
    # file-system write bandwidth rather than cores.
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--threads", type=int, default=2, help="Arrow CPU and IO pool size per worker"
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.output_root is None:
        arguments.output_root = os.path.join(arguments.source_root, arguments.preset)

    # Left alone, PyArrow sizes its pools from every core on the node in every worker.
    pa.set_cpu_count(arguments.threads)
    pa.set_io_thread_count(arguments.threads)

    run(arguments)


if __name__ == "__main__":
    main()
