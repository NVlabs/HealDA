# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Earth2Bufr decoding and batch conversion for the NNJA sounder archive."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import earth2bufr
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from healda.observations.preprocessing.sat_helpers import hpx_nest_pixels
from healda.utils.profiling import cpu_timing_range

from .archive_schema import (
    COMMON_FIELD_NAMES,
    ChannelManifest,
    FieldSpec,
    PrimaryLeaf,
    PrimaryValueEncoding,
    ATOVS_SENSORS,
    SENSOR_SPECS,
    SourceStructure,
    archive_field_spec,
    archive_schema,
    derive_cris_primary_encoding,
    primary_field_name,
)

_CHANNEL_AXIS_BLOCK_ROWS = 8_192
_HPX2048_LEVEL = 11
DECODER_NAME = "Earth2Bufr"
# Decoded Arrow bytes per batch; a batch also stops at MESSAGES_PER_BATCH.
TARGET_DECODED_BYTES = 64 * 2**20
MESSAGES_PER_BATCH = 8_192
# Batches earth2bufr decodes in the background while the current one converts.
PREFETCH_BATCHES = 1
# Sensors whose fixed replications decode as FixedSizeList<Struct> columns.
STRUCTURAL_SENSORS = frozenset({"cris"})
# earth2bufr does not mark structural batches; the ETL tags them so later steps can
# tell without scanning the schema.
_STRUCTURAL_KEY = b"earth2bufr.fixed_replication"


def _primary_raw_code_paths(sensor: str) -> tuple[str, ...] | None:
    spec = SENSOR_SPECS[sensor]
    if spec.primary_value_encoding is None:
        return None
    paths = tuple(
        f"{leaf.source_container}.{leaf.source_mnemonic}"
        for leaf in spec.primary_leaves
        if leaf.source_container is not None
    )
    if len(paths) != len(spec.primary_leaves):
        raise ValueError(f"{sensor} packed primary field has no source container")
    return paths


def _reference_time(message: earth2bufr.MessageInfo) -> datetime | None:
    hour, minute, second = message.hour, message.minute, message.second
    if (
        not 0 <= hour <= 24
        or not 0 <= minute <= 59
        or not 0 <= second < 61
        or (hour == 24 and (minute or second))
    ):
        return None
    try:
        day = datetime(message.year, message.month, message.day, tzinfo=timezone.utc)
    except ValueError:
        return None
    return day + timedelta(hours=hour, minutes=minute, seconds=second)


def _message_header(message: earth2bufr.MessageInfo) -> BufrMessageHeader:
    return BufrMessageHeader(
        index=message.source_message_index,
        edition=message.edition,
        master_table=message.master_table,
        center=message.center,
        sub_center=message.sub_center,
        update_sequence=message.update_sequence,
        optional_section_present=message.optional_section_present,
        reference_time=_reference_time(message),
        data_category=message.data_category,
        international_sub_category=message.international_sub_category,
        local_sub_category=message.local_sub_category,
        master_table_version=message.master_table_version,
        local_table_version=message.local_table_version,
        subset_count=message.subsets,
        compressed=message.compressed,
        section3_flags=message.section3_flags,
        descriptors=message.descriptors,
    )


@contextmanager
def open_source(path: Path) -> Iterator[Earth2BufrSource]:
    """Open and index one BUFR file for its headers, structures and batches."""

    with cpu_timing_range("source.open_index"):
        handle = earth2bufr.open_bufr(path, strict=True)
    with handle:
        yield Earth2BufrSource(handle)


class Earth2BufrSource:
    """One open BUFR file and its message index."""

    def __init__(self, handle: earth2bufr.BufrFile) -> None:
        self.handle = handle
        self._batches: earth2bufr.BatchIterator | None = None

    @cached_property
    def message_headers(self) -> tuple[BufrMessageHeader, ...]:
        headers = tuple(
            _message_header(message) for message in self.handle.info.data_messages
        )
        if not headers:
            raise ValueError("source contains no meaningful data messages")
        return headers

    def source_structure_bindings(self, sensor: str) -> dict[int, SourceStructure]:
        """Bind each data message to the structure of its table epoch and template."""

        structures: dict[tuple[int | None, tuple[str, ...]], SourceStructure] = {}
        bindings: dict[int, SourceStructure] = {}
        for message in self.handle.info.data_messages:
            index = message.source_message_index
            key = (message.table_epoch, message.descriptors)
            structure = structures.get(key)
            if structure is None:
                structure = SourceStructure.from_decoder(
                    sensor, self.handle.source_structure(index)
                )
                if structure.observation_descriptors != message.descriptors:
                    raise ValueError(
                        f"message {index} descriptors disagree with its "
                        "source structure"
                    )
                structures[key] = structure
            bindings[index] = structure
        return bindings

    def decode_failures(self) -> tuple[Mapping[str, Any], ...]:
        """Messages the last ``iter_batches`` call skipped as truncated."""

        if self._batches is None:
            return ()
        messages = self.handle.info.messages
        records = []
        with self.handle.path.open("rb") as stream:
            for failure in self._batches.decode_failures:
                message = messages[failure.source_message_index]
                stream.seek(message.offset)
                payload = stream.read(message.size_bytes)
                records.append(
                    {
                        "source_message_index": failure.source_message_index,
                        "subset_count": message.subsets,
                        "message_size_bytes": message.size_bytes,
                        "message_sha256": hashlib.sha256(payload).hexdigest(),
                        "reason": failure.reason,
                    }
                )
        return tuple(records)

    def iter_batches(self, *, sensor: str) -> Iterator[pa.RecordBatch]:
        structural = sensor in STRUCTURAL_SENSORS
        self._batches = self.handle.iter_batches(
            messages_per_batch=MESSAGES_PER_BATCH,
            raw_code_paths=_primary_raw_code_paths(sensor),
            prefetch_batches=PREFETCH_BATCHES,
            target_decoded_bytes=TARGET_DECODED_BYTES,
            fixed_replication="structural" if structural else "flat",
            skip_truncated_messages=True,
        )
        while True:
            with cpu_timing_range("decoder.next_batch"):
                batch = next(self._batches, None)
            if batch is None:
                return
            if structural:
                batch = batch.replace_schema_metadata(
                    {**(batch.schema.metadata or {}), _STRUCTURAL_KEY: b"structural"}
                )
            yield batch


# A NamedTuple: one is built per BUFR message (about 220,000 per CrIS file).
class BufrMessageHeader(NamedTuple):
    index: int
    edition: int
    master_table: int
    center: int
    sub_center: int
    update_sequence: int
    optional_section_present: bool
    reference_time: datetime | None
    data_category: int
    international_sub_category: int
    local_sub_category: int
    master_table_version: int
    local_table_version: int
    subset_count: int
    compressed: bool
    section3_flags: int
    descriptors: tuple[str, ...]


@dataclass(frozen=True)
class SounderPrimaryBatch:
    """One sensor's validated wide primary-channel block."""

    sensor: str
    table: pa.Table
    channel_manifest: ChannelManifest


@dataclass(frozen=True)
class CompiledArchiveLayout:
    """Immutable source-bound state shared by every converted decoder batch."""

    sensor: str
    schema: pa.Schema
    channel_manifest: ChannelManifest
    primary_encoding: PrimaryValueEncoding | None
    output_names: tuple[str, ...]
    source_structure: SourceStructure


@dataclass(frozen=True)
class SounderArchiveBatch:
    """One bounded, complete batch matching the logical archive schema."""

    sensor: str
    table: pa.Table
    channel_manifest: ChannelManifest
    layout: CompiledArchiveLayout


def _validate_decoder_raw_code_field(
    field: pa.Field, encoding: PrimaryValueEncoding
) -> None:
    metadata = field.metadata or {}
    expected = {
        b"representation": b"bufr_raw_code",
        b"descriptor": encoding.descriptor.encode(),
        b"scale": str(encoding.scale).encode(),
        b"reference": str(encoding.reference).encode(),
        b"bit_width": str(encoding.bit_width).encode(),
        b"missing_code": str(encoding.source_missing_code).encode(),
        b"source_units": encoding.source_units.encode(),
    }
    mismatches = {
        key.decode(): {
            "expected": expected_value.decode(),
            "actual": None if metadata.get(key) is None else metadata[key].decode(),
        }
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if b"physical_formula" not in metadata:
        mismatches["physical_formula"] = {"expected": "present", "actual": None}
    if mismatches:
        raise ValueError(
            f"CrIS decoder raw-code metadata does not match source structure: {mismatches}"
        )


def initialize_hpx2048() -> None:
    """Import and exercise the HEALPix pixel lookup once per worker process."""

    zero = np.zeros(1)
    hpx_nest_pixels(zero, zero, _HPX2048_LEVEL)


def compute_hpx2048_nest(longitude: pa.Array, latitude: pa.Array) -> pa.Array:
    """Return required Earth2Grid NESTED nside-2048 footprint pixels."""

    if len(longitude) != len(latitude):
        raise ValueError("longitude and latitude lengths differ")
    # Nulls become NaN, which hpx_nest_pixels reports as invalid.
    pixels, valid = hpx_nest_pixels(
        latitude.to_numpy(zero_copy_only=False),
        longitude.to_numpy(zero_copy_only=False),
        _HPX2048_LEVEL,
    )
    return pa.array(pixels, mask=~valid, type=pa.uint32())


def assign_da_windows(observation_times: pa.Array, cycle: datetime) -> pa.Array:
    """Assign nominal right-edge labels without clipping source observation time.

    Uses the source file's cycle, not one derived from time_utc, so
    out-of-interval tails stay in their file's day.
    """

    if observation_times.null_count:
        raise ValueError("observation time contains nulls")
    if cycle.tzinfo is None or cycle.utcoffset() != timedelta(0):
        raise ValueError("cycle must be timezone-aware UTC")
    cycle = cycle.astimezone(timezone.utc)
    cycle_scalar = pa.scalar(cycle, observation_times.type)
    return pc.if_else(
        pc.less_equal(observation_times, cycle_scalar),
        cycle_scalar,
        pa.scalar(cycle + timedelta(hours=3), observation_times.type),
    )


def _columns(batch: pa.RecordBatch, name: str) -> list[pa.Array]:
    return [
        batch.column(index)
        for index, field in enumerate(batch.schema)
        if field.name == name
    ]


def _flatten_structural_schema_fields(schema: pa.Schema) -> list[pa.Field]:
    fields: list[pa.Field] = []

    def append(field: pa.Field) -> None:
        data_type = field.type
        if not pa.types.is_fixed_size_list(data_type):
            fields.append(field)
            return
        if not pa.types.is_struct(data_type.value_type):
            raise ValueError(
                f"structural fixed field {field.name!r} is not FixedSizeList<Struct>"
            )
        for _ in range(data_type.list_size):
            for child in data_type.value_type:
                append(child)

    for field in schema:
        append(field)
    return fields


def _field_source_mnemonics(sensor: str, logical_name: str) -> tuple[str, ...]:
    return archive_field_spec(sensor, logical_name).source_mnemonics


def _single_field_source_mnemonic(sensor: str, logical_name: str) -> str:
    mnemonics = _field_source_mnemonics(sensor, logical_name)
    if len(mnemonics) != 1:
        raise ValueError(
            f"{sensor}/{logical_name} has {len(mnemonics)} source mnemonics"
        )
    return mnemonics[0]


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    columns = _columns(batch, name)
    if len(columns) != 1:
        raise ValueError(f"expected one {name!r} column, found {len(columns)}")
    return columns[0]


def _first_occurrence(batch: pa.RecordBatch, name: str) -> pa.Array:
    columns = _columns(batch, name)
    if not columns:
        raise ValueError(f"missing {name!r}")
    return columns[0]


def _checked_cast(values: pa.Array, data_type: pa.DataType, name: str) -> pa.Array:
    try:
        return pc.cast(values, data_type, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ValueError(f"{name} cannot be represented as {data_type}") from exc


def _field_source_present(batch: pa.RecordBatch, field_spec: FieldSpec) -> bool:
    names = set(batch.schema.names)
    if field_spec.source_container is not None:
        return field_spec.source_container in names
    return any(mnemonic in names for mnemonic in field_spec.source_mnemonics)


def _null_array(length: int, data_type: pa.DataType) -> pa.Array:
    """Build nested nulls with physical children required by Parquet.

    The parent is null as well as its children, so a missing source field reads
    as an Arrow null rather than as a present list or struct of nulls.
    """

    if pa.types.is_fixed_size_list(data_type):
        values = _null_array(length * data_type.list_size, data_type.value_type)
        return pa.FixedSizeListArray.from_arrays(
            values,
            data_type.list_size,
            mask=pa.array(np.ones(length, dtype=bool)),
        )
    if pa.types.is_struct(data_type):
        return pa.StructArray.from_arrays(
            [_null_array(length, field.type) for field in data_type],
            fields=list(data_type),
            mask=pa.array(np.ones(length, dtype=bool)),
        )
    return pa.nulls(length, type=data_type)


def _common_scalar_array(
    batch: pa.RecordBatch,
    *,
    sensor: str,
    logical_name: str,
    unsigned: bool = False,
) -> pa.Array:
    field_spec = archive_field_spec(sensor, logical_name)
    mnemonic = _single_field_source_mnemonic(sensor, logical_name)
    columns = _columns(batch, mnemonic)
    if not columns:
        if not field_spec.source_optional:
            raise ValueError(f"required source field {mnemonic!r} is absent")
        return _null_array(batch.num_rows, field_spec.data_type)
    indices = field_spec.source_occurrence_indices.get(mnemonic)
    if indices:
        if len(indices) != 1 or indices[0] >= len(columns):
            raise ValueError(
                f"{logical_name} binding {indices} exceeds "
                f"{len(columns)} {mnemonic!r} columns"
            )
        column = columns[indices[0]]
    else:
        if len(columns) != 1:
            raise ValueError(f"expected one {mnemonic!r} column, found {len(columns)}")
        column = columns[0]
    if unsigned:
        return _checked_cast(column, field_spec.data_type, logical_name)
    return _checked_cast(column, field_spec.data_type, logical_name)


def _time_arrays(
    batch: pa.RecordBatch, *, sensor: str
) -> tuple[pa.Array, pa.StructArray]:
    names = _field_source_mnemonics(sensor, "time_utc")
    source = [_first_occurrence(batch, name) for name in names]
    if any(array.null_count for array in source):
        raise ValueError("source observation time contains null components")
    values = [
        np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
        for array in source
    ]
    if any(np.any(~np.isfinite(value)) for value in values):
        raise ValueError("source observation time contains non-finite components")

    year, month, day, hour, minute = [
        np.rint(value).astype(np.int64) for value in values[:5]
    ]
    for name, original, integer in zip(
        names[:5], values[:5], (year, month, day, hour, minute)
    ):
        if not np.array_equal(original, integer.astype(np.float64)):
            raise ValueError(f"{name} contains non-integer values")
    second = values[5]
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    month_lengths = np.array(
        [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64
    )
    valid_month = (month >= 1) & (month <= 12)
    maximum_day = np.zeros(len(month), dtype=np.int64)
    maximum_day[valid_month] = month_lengths[month[valid_month]]
    maximum_day[(month == 2) & leap] = 29
    if np.any(
        (year < 1678)
        | (year > 2261)
        | ~valid_month
        | (day < 1)
        | (day > maximum_day)
        | (hour < 0)
        | (hour > 24)
        | ((hour == 24) & ((minute != 0) | (second != 0.0)))
        | (minute < 0)
        | (minute > 59)
        | (second < 0.0)
        | (second >= 61.0)
    ):
        raise ValueError("source observation time is outside supported UTC ranges")

    adjusted_year = year - (month <= 2)
    era = np.floor_divide(adjusted_year, 400)
    year_of_era = adjusted_year - era * 400
    shifted_month = month + np.where(month > 2, -3, 9)
    day_of_year = (153 * shifted_month + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    days_since_epoch = era * 146_097 + day_of_era - 719_468
    nanoseconds = (
        days_since_epoch * 86_400_000_000_000
        + hour * 3_600_000_000_000
        + minute * 60_000_000_000
        + np.rint(second * 1_000_000_000).astype(np.int64)
    )
    time_utc = pa.array(nanoseconds, type=pa.timestamp("ns", tz="UTC"))
    source_time = pa.StructArray.from_arrays(
        [
            pa.array(year, type=pa.uint16()),
            pa.array(month, type=pa.uint8()),
            pa.array(day, type=pa.uint8()),
            pa.array(hour, type=pa.uint8()),
            pa.array(minute, type=pa.uint8()),
            pa.array(second, type=pa.float64()),
        ],
        names=["year", "month", "day", "hour", "minute", "second"],
    )
    return time_utc, source_time


def _message_time_array(
    source_message_index: pa.Array,
    message_reference_times: Mapping[int, datetime],
) -> pa.Array:
    indices = np.asarray(source_message_index.to_numpy(zero_copy_only=True))
    unique, inverse = np.unique(indices, return_inverse=True)
    times: list[datetime] = []
    for index in unique:
        reference_time = message_reference_times.get(int(index))
        if reference_time is None:
            raise ValueError(f"source message {int(index)} has no reference time")
        if reference_time.tzinfo is None or reference_time.utcoffset() != timedelta(0):
            raise ValueError(f"source message {int(index)} time is not UTC")
        times.append(reference_time.astimezone(timezone.utc))
    unique_values = pa.array(times, type=pa.timestamp("ms", tz="UTC"))
    return pc.take(unique_values, pa.array(inverse, type=pa.int64()))


def convert_common_footprint_batch(
    batch: pa.RecordBatch,
    *,
    sensor: str,
    cycle: datetime,
    source_file_ordinal: int,
    message_reference_times: Mapping[int, datetime],
) -> pa.Table:
    """Build the common archive prefix without dropping source rows."""

    if sensor not in SENSOR_SPECS:
        raise ValueError(f"unsupported sensor {sensor!r}")
    if batch.num_rows == 0:
        raise ValueError("decoder batch contains no source rows")
    file_ordinal_type = archive_field_spec(sensor, "source_file_ordinal").data_type
    file_ordinal_dtype = np.dtype(file_ordinal_type.to_pandas_dtype())
    if (
        source_file_ordinal < 0
        or source_file_ordinal > np.iinfo(file_ordinal_dtype).max
    ):
        raise ValueError("source_file_ordinal does not fit uint16")
    latitude_name = _single_field_source_mnemonic(sensor, "latitude")
    longitude_name = _single_field_source_mnemonic(sensor, "longitude")
    time_utc, source_time = _time_arrays(batch, sensor=sensor)
    latitude = _checked_cast(
        _first_occurrence(batch, latitude_name),
        archive_field_spec(sensor, "latitude").data_type,
        "latitude",
    )
    longitude = _checked_cast(
        _first_occurrence(batch, longitude_name),
        archive_field_spec(sensor, "longitude").data_type,
        "longitude",
    )
    source_message_index = _checked_cast(
        _column(batch, "_source_message_index"),
        archive_field_spec(sensor, "source_message_index").data_type,
        "source_message_index",
    )
    with cpu_timing_range("convert.hpx2048"):
        hpx2048_nest = compute_hpx2048_nest(longitude, latitude)
    arrays = {
        "da_window": assign_da_windows(time_utc, cycle),
        "time_utc": time_utc,
        "source_time": source_time,
        "platform_id": _common_scalar_array(
            batch,
            sensor=sensor,
            logical_name="platform_id",
            unsigned=True,
        ),
        "instrument_id": _common_scalar_array(
            batch,
            sensor=sensor,
            logical_name="instrument_id",
            unsigned=True,
        ),
        "latitude": latitude,
        "longitude": longitude,
        "satellite_zenith_angle": _common_scalar_array(
            batch,
            sensor=sensor,
            logical_name="satellite_zenith_angle",
        ),
        "satellite_azimuth_angle": _common_scalar_array(
            batch,
            sensor=sensor,
            logical_name="satellite_azimuth_angle",
        ),
        "solar_zenith_angle": _common_scalar_array(
            batch,
            sensor=sensor,
            logical_name="solar_zenith_angle",
        ),
        "solar_azimuth_angle": _common_scalar_array(
            batch,
            sensor=sensor,
            logical_name="solar_azimuth_angle",
        ),
        "field_of_view": _common_scalar_array(
            batch,
            sensor=sensor,
            logical_name="field_of_view",
            unsigned=True,
        ),
        "hpx2048_nest": hpx2048_nest,
        "source_file_ordinal": pa.array(
            np.full(batch.num_rows, source_file_ordinal, dtype=file_ordinal_dtype),
            type=file_ordinal_type,
        ),
        "source_message_index": source_message_index,
        "source_subset_index": _checked_cast(
            _column(batch, "_source_subset_index"),
            archive_field_spec(sensor, "source_subset_index").data_type,
            "source_subset_index",
        ),
        "source_message_time": _message_time_array(
            source_message_index, message_reference_times
        ),
        "source_data_category": _checked_cast(
            _column(batch, "_data_category"),
            archive_field_spec(sensor, "source_data_category").data_type,
            "source_data_category",
        ),
    }
    return pa.Table.from_arrays(
        [arrays[name] for name in COMMON_FIELD_NAMES], names=COMMON_FIELD_NAMES
    )


def _manifest_from_channel_matrix(
    channel_matrix: np.ndarray,
    *,
    sensor: str,
    release: str,
    expected_manifest: ChannelManifest | None,
) -> ChannelManifest:
    spec = SENSOR_SPECS[sensor]
    if channel_matrix.shape != (channel_matrix.shape[0], spec.channel_count):
        raise ValueError(
            f"{sensor} channel matrix has shape {channel_matrix.shape}, expected "
            f"(*, {spec.channel_count})"
        )
    if channel_matrix.shape[0] == 0:
        raise ValueError(f"{sensor} channel matrix contains no source rows")
    if np.any(~np.isfinite(channel_matrix)):
        raise ValueError(f"{sensor} primary channel identifiers contain nulls")
    first_axis = channel_matrix[0]
    rounded_first_axis = np.rint(first_axis)
    if not np.array_equal(first_axis, rounded_first_axis):
        raise ValueError(f"{sensor} primary channel identifiers are not integers")
    for start in range(1, channel_matrix.shape[0], _CHANNEL_AXIS_BLOCK_ROWS):
        stop = min(start + _CHANNEL_AXIS_BLOCK_ROWS, channel_matrix.shape[0])
        if not np.all(channel_matrix[start:stop] == first_axis):
            raise ValueError(
                f"{sensor} primary channel axis varies between source rows"
            )
    channel_ids = tuple(rounded_first_axis.astype(np.int64).tolist())
    if expected_manifest is not None:
        expected_axis = tuple(
            channel.source_channel_id for channel in expected_manifest.channels
        )
        if expected_manifest.release != release or channel_ids != expected_axis:
            raise ValueError(
                f"{sensor} channel manifest changed between decoder batches"
            )
        return expected_manifest
    manifest = ChannelManifest.from_ids(sensor, channel_ids, release=release)
    manifest.validate(spec.channel_count)
    return manifest


def _list_primary_parts(
    batch: pa.RecordBatch, *, sensor: str
) -> tuple[pa.StructArray, np.ndarray]:
    spec = SENSOR_SPECS[sensor]
    containers = {leaf.source_container for leaf in spec.primary_leaves}
    if len(containers) != 1 or None in containers:
        raise ValueError(f"{sensor} primary source container is not unambiguous")
    repeated = _column(batch, containers.pop())
    lengths = np.asarray(pc.list_value_length(repeated).fill_null(-1))
    if not np.all(lengths == spec.channel_count):
        raise ValueError(
            f"{sensor} primary channel count changed within batch: "
            f"{np.unique(lengths).tolist()}"
        )
    flattened = pc.list_flatten(repeated)
    if len(spec.manifest_mnemonics) != 1:
        raise ValueError(f"{sensor} channel manifest source is not unambiguous")
    channel_values = flattened.field(spec.manifest_mnemonics[0])
    if channel_values.null_count:
        raise ValueError(f"{sensor} primary channel identifiers contain nulls")
    channel_matrix = np.asarray(
        channel_values.to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(batch.num_rows, spec.channel_count)
    return flattened, channel_matrix


def _wide_physical_columns(
    values: pa.Array,
    *,
    row_count: int,
    channel_ids: Sequence[int],
    leaf: PrimaryLeaf,
) -> pa.Table:
    channel_count = len(channel_ids)
    if len(values) != row_count * channel_count:
        raise ValueError(
            f"{leaf.prefix} contains {len(values)} values, expected "
            f"{row_count * channel_count}"
        )
    nulls = np.asarray(values.is_null()).reshape(row_count, channel_count)
    physical = np.asarray(
        values.to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(row_count, channel_count)
    return _wide_columns(physical, nulls, channel_ids=channel_ids, leaf=leaf)


def _wide_columns(
    physical: np.ndarray,
    nulls: np.ndarray,
    *,
    channel_ids: Sequence[int],
    leaf: PrimaryLeaf,
) -> pa.Table:
    """One primitive column per channel from row-major (rows, channels) matrices."""

    channel_count = len(channel_ids)
    channel_major = np.ascontiguousarray(
        physical.T, dtype=leaf.data_type.to_pandas_dtype()
    )
    # A column without nulls is wrapped zero-copy, with no validity bitmap to build.
    has_nulls = nulls.any(axis=0)
    channel_major_nulls = np.ascontiguousarray(nulls.T) if has_nulls.any() else None
    arrays = [
        pa.array(
            channel_major[position],
            mask=channel_major_nulls[position] if has_nulls[position] else None,
            type=leaf.data_type,
        )
        for position in range(channel_count)
    ]
    names = [primary_field_name(leaf, channel_id) for channel_id in channel_ids]
    return pa.Table.from_arrays(arrays, names=names)


def _convert_atovs_primary_batch(
    batch: pa.RecordBatch,
    *,
    sensor: str,
    release: str,
    expected_manifest: ChannelManifest | None,
) -> SounderPrimaryBatch:
    spec = SENSOR_SPECS[sensor]
    leaf = spec.primary_leaves[0]
    channel_columns = _columns(batch, spec.manifest_mnemonics[0])
    value_columns = _columns(batch, leaf.source_mnemonic)
    expected_count = spec.channel_count
    if len(channel_columns) != expected_count or len(value_columns) != expected_count:
        raise ValueError(
            f"{sensor} expected {expected_count} channel/value occurrences, found "
            f"{len(channel_columns)}/{len(value_columns)}"
        )
    channel_matrix = np.column_stack(
        [
            np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
            for column in channel_columns
        ]
    )
    manifest = _manifest_from_channel_matrix(
        channel_matrix,
        sensor=sensor,
        release=release,
        expected_manifest=expected_manifest,
    )
    channel_ids = [channel.source_channel_id for channel in manifest.channels]
    arrays = [
        _checked_cast(
            column,
            leaf.data_type,
            f"{sensor} {leaf.source_mnemonic} channel {channel_id}",
        )
        for channel_id, column in zip(channel_ids, value_columns, strict=True)
    ]
    names = [primary_field_name(leaf, channel_id) for channel_id in channel_ids]
    return SounderPrimaryBatch(
        sensor=sensor,
        table=pa.Table.from_arrays(arrays, names=names),
        channel_manifest=manifest,
    )


def _convert_list_float_primary_batch(
    batch: pa.RecordBatch,
    *,
    sensor: str,
    release: str,
    expected_manifest: ChannelManifest | None,
) -> SounderPrimaryBatch:
    with cpu_timing_range("primary.flatten_axis"):
        flattened, channel_matrix = _list_primary_parts(batch, sensor=sensor)
    with cpu_timing_range("primary.validate_manifest"):
        manifest = _manifest_from_channel_matrix(
            channel_matrix,
            sensor=sensor,
            release=release,
            expected_manifest=expected_manifest,
        )
    channel_ids = [channel.source_channel_id for channel in manifest.channels]
    with cpu_timing_range("primary.wide_columns"):
        tables = [
            _wide_physical_columns(
                flattened.field(leaf.source_mnemonic),
                row_count=batch.num_rows,
                channel_ids=channel_ids,
                leaf=leaf,
            )
            for leaf in SENSOR_SPECS[sensor].primary_leaves
        ]
    return SounderPrimaryBatch(
        sensor=sensor,
        table=pa.Table.from_arrays(
            [column for table in tables for column in table.columns],
            names=[name for table in tables for name in table.column_names],
        ),
        channel_manifest=manifest,
    )


def _convert_iasi_primary_batch(
    batch: pa.RecordBatch,
    *,
    release: str,
    expected_manifest: ChannelManifest | None,
) -> SounderPrimaryBatch:
    sensor = "iasi"
    flattened, channel_matrix = _list_primary_parts(batch, sensor=sensor)
    manifest = _manifest_from_channel_matrix(
        channel_matrix,
        sensor=sensor,
        release=release,
        expected_manifest=expected_manifest,
    )
    scale_groups = archive_field_spec(sensor, "main_scale_groups")
    starts = _bound_columns(batch, scale_groups, "STCH")
    stops = _bound_columns(batch, scale_groups, "ENCH")
    scales = _bound_columns(batch, scale_groups, "CHSF")
    if any(array.null_count for array in (*starts, *stops)):
        raise ValueError("IASI main-spectrum channel bounds contain nulls")
    start_matrix = np.column_stack(
        [np.asarray(value.to_numpy(), dtype=np.float64) for value in starts]
    )
    stop_matrix = np.column_stack(
        [np.asarray(value.to_numpy(), dtype=np.float64) for value in stops]
    )
    scale_matrix = np.column_stack(
        [
            np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float64)
            for value in scales
        ]
    )
    scale_nulls = np.column_stack([np.asarray(value.is_null()) for value in scales])
    if np.any(~np.isfinite(start_matrix)) or np.any(~np.isfinite(stop_matrix)):
        raise ValueError("IASI main-spectrum channel bounds contain non-finite values")
    if np.any(~np.isfinite(scale_matrix) & ~scale_nulls):
        raise ValueError("IASI main-spectrum scale groups contain non-finite values")
    channel_ids = np.asarray(
        [channel.source_channel_id for channel in manifest.channels], dtype=np.float64
    )
    # Rows share few distinct (start, stop) layouts; resolve each layout's group per
    # channel once.
    bounds = np.concatenate([start_matrix, stop_matrix], axis=1)
    if np.all(bounds == bounds[0]):
        layouts, layout_of_row = bounds[:1], np.zeros(len(bounds), dtype=np.intp)
    else:
        layouts, layout_of_row = np.unique(bounds, axis=0, return_inverse=True)
    group_count = start_matrix.shape[1]
    group_of_channel = np.empty((len(layouts), len(channel_ids)), dtype=np.intp)
    for position, layout in enumerate(layouts):
        membership = (channel_ids[None, :] >= layout[:group_count, None]) & (
            channel_ids[None, :] <= layout[group_count:, None]
        )
        if not np.all(membership.sum(axis=0) == 1):
            raise ValueError(
                "IASI scale groups do not cover every primary channel once"
            )
        group_of_channel[position] = membership.argmax(axis=0)
    group_index = group_of_channel[layout_of_row.reshape(-1)]
    rows = np.arange(batch.num_rows)[:, None]

    def per_channel(values: np.ndarray) -> np.ndarray:
        if len(layouts) == 1:
            return values[:, group_of_channel[0]]
        return values[rows, group_index]

    source_values = flattened.field("SCRA")
    source_nulls = np.asarray(source_values.is_null()).reshape(
        batch.num_rows, len(channel_ids)
    )
    source_matrix = np.asarray(
        source_values.to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(batch.num_rows, len(channel_ids))
    nulls = source_nulls
    if scale_nulls.any():
        missing_channel_scales = per_channel(scale_nulls)
        invalid_unknown_scale = (
            missing_channel_scales & ~source_nulls & (source_matrix != 0.0)
        )
        if np.any(invalid_unknown_scale):
            row, channel = np.argwhere(invalid_unknown_scale)[0]
            raise ValueError(
                "IASI main-spectrum scale is null for a nonzero SCRA value "
                f"at row {row}, channel {int(channel_ids[channel])}"
            )
        nulls = source_nulls | missing_channel_scales
    # 10**-CHSF is the descriptor's own scale, in W m^-2 sr^-1 (m^-1)^-1; the further
    # 10**5 is 10**3 for milliwatts and 10**2 for per cm^-1, the CRTM scale.
    group_factor = np.power(10.0, 5.0 - np.where(scale_nulls, 0.0, scale_matrix))
    physical = source_matrix * per_channel(group_factor)
    table = _wide_columns(
        physical,
        nulls,
        channel_ids=channel_ids.astype(np.int64).tolist(),
        leaf=SENSOR_SPECS[sensor].primary_leaves[0],
    )
    return SounderPrimaryBatch(
        sensor=sensor,
        table=table,
        channel_manifest=manifest,
    )


def _structural_top_level_occurrence_indices(
    batch: pa.RecordBatch,
    mnemonic: str,
    indices: Sequence[int],
) -> tuple[int, ...]:
    if (batch.schema.metadata or {}).get(_STRUCTURAL_KEY) != b"structural":
        return tuple(indices)

    mapping = dict(_structural_top_level_occurrence_map(batch.schema, mnemonic))
    try:
        return tuple(mapping[index] for index in indices)
    except KeyError as exc:
        raise ValueError(
            f"{mnemonic} binding {tuple(indices)} targets a structural fixed group"
        ) from exc


def _structural_top_level_occurrence_map(
    schema: pa.Schema,
    mnemonic: str,
) -> tuple[tuple[int, int], ...]:
    global_to_top_level: dict[int, int] = {}
    global_occurrence = 0
    top_level_occurrence = 0

    def nested_count(field: pa.Field) -> int:
        data_type = field.type
        if pa.types.is_fixed_size_list(data_type):
            return data_type.list_size * sum(
                nested_count(child) for child in data_type.value_type
            )
        if pa.types.is_struct(data_type):
            return sum(nested_count(child) for child in data_type)
        return int(field.name == mnemonic)

    for field in schema:
        if pa.types.is_fixed_size_list(field.type):
            global_occurrence += nested_count(field)
        elif field.name == mnemonic:
            global_to_top_level[global_occurrence] = top_level_occurrence
            global_occurrence += 1
            top_level_occurrence += 1
    return tuple(global_to_top_level.items())


def _bound_columns(
    batch: pa.RecordBatch, field_spec: FieldSpec, mnemonic: str
) -> list[pa.Array]:
    columns = _columns(batch, mnemonic)
    indices = field_spec.source_occurrence_indices.get(mnemonic)
    if indices is None:
        return columns
    indices = _structural_top_level_occurrence_indices(batch, mnemonic, indices)
    if not indices or max(indices) >= len(columns):
        raise ValueError(
            f"{field_spec.name}/{mnemonic} binding {indices} exceeds "
            f"{len(columns)} decoded occurrences"
        )
    return [columns[index] for index in indices]


def _interleave_columns(
    columns: Sequence[pa.Array], data_type: pa.DataType, name: str
) -> pa.Array:
    if not columns:
        raise ValueError(f"{name} has no source occurrences")
    cast = [_checked_cast(column, data_type, name) for column in columns]
    values = np.column_stack([column.to_numpy(zero_copy_only=False) for column in cast])
    nulls = np.column_stack([np.asarray(column.is_null()) for column in cast])
    return pa.array(values.reshape(-1), mask=nulls.reshape(-1), type=data_type)


def _fixed_list_from_columns(
    columns: Sequence[pa.Array], data_type: pa.DataType, size: int, name: str
) -> pa.FixedSizeListArray:
    if len(columns) != size:
        raise ValueError(f"{name} needs {size} occurrences, found {len(columns)}")
    return pa.FixedSizeListArray.from_arrays(
        _interleave_columns(columns, data_type, name), size
    )


def _fixed_struct_list(
    children: Sequence[tuple[str, Sequence[pa.Array], pa.DataType]],
    *,
    size: int,
    name: str,
) -> pa.FixedSizeListArray:
    arrays = [
        _interleave_columns(columns, data_type, f"{name}.{child_name}")
        for child_name, columns, data_type in children
    ]
    struct = pa.StructArray.from_arrays(
        arrays, names=[child_name for child_name, _, _ in children]
    )
    return pa.FixedSizeListArray.from_arrays(struct, size)


def _flatten_container(
    batch: pa.RecordBatch,
    field_spec: FieldSpec,
    *,
    expected_size: int,
) -> pa.StructArray:
    if field_spec.source_container is None:
        raise ValueError(f"{field_spec.name} has no source container binding")
    repeated = _column(batch, field_spec.source_container)
    lengths = np.asarray(pc.list_value_length(repeated).fill_null(-1))
    if not np.all(lengths == expected_size):
        raise ValueError(
            f"{field_spec.name} needs {expected_size} entries per row, found "
            f"{np.unique(lengths).tolist()}"
        )
    return pc.list_flatten(repeated)


@dataclass(frozen=True)
class _StructuralGroup:
    size: int
    child_names: tuple[str, ...]
    nested_size: int | None = None
    nested_child_names: tuple[str, ...] = ()


_CRIS_BAND_QUALITY_GROUP = _StructuralGroup(
    size=3, child_names=("TOBD", "WVNM", "WVNM", "STCH", "ENCH", "NCQF", "NFQF")
)
_CRIS_IMAGER_GROUP = _StructuralGroup(
    size=7,
    child_names=("YAPCG", "ZAPCG", "FCPH", "__fixed__"),
    nested_size=16,
    nested_child_names=("CHNM", "FOST", "SCHRAD", "FOST", "SCHRAD"),
)


def _structural_fixed_group(
    batch: pa.RecordBatch, group: _StructuralGroup
) -> tuple[int, pa.FixedSizeListArray]:
    matches = []
    for position, (field, array) in enumerate(
        zip(batch.schema, batch.columns, strict=True)
    ):
        data_type = field.type
        if (
            not field.name.startswith("fixed_")
            or not pa.types.is_fixed_size_list(data_type)
            or data_type.list_size != group.size
            or not pa.types.is_struct(data_type.value_type)
        ):
            continue
        actual_names = tuple(
            "__fixed__" if pa.types.is_fixed_size_list(child.type) else child.name
            for child in data_type.value_type
        )
        if actual_names != group.child_names:
            continue
        if group.nested_size is not None:
            nested = data_type.value_type[-1].type
            if (
                not pa.types.is_fixed_size_list(nested)
                or nested.list_size != group.nested_size
                or not pa.types.is_struct(nested.value_type)
                or tuple(child.name for child in nested.value_type)
                != group.nested_child_names
            ):
                continue
        matches.append((position, array))
    if len(matches) != 1:
        raise ValueError(
            f"expected one structural fixed group {group}, found {len(matches)}"
        )
    return matches[0]


def _simple_secondary_array(batch: pa.RecordBatch, field_spec: FieldSpec) -> pa.Array:
    if len(field_spec.source_mnemonics) != 1:
        raise ValueError(f"{field_spec.name} is not a simple source field")
    columns = _bound_columns(batch, field_spec, field_spec.source_mnemonics[0])
    if len(columns) != 1:
        raise ValueError(
            f"{field_spec.name} needs one source occurrence, found {len(columns)}"
        )
    return _checked_cast(columns[0], field_spec.data_type, field_spec.name)


def _atovs_secondary_array(batch: pa.RecordBatch, field_spec: FieldSpec) -> pa.Array:
    if field_spec.name != "cold_space_temperature_correction":
        return _simple_secondary_array(batch, field_spec)
    size = field_spec.data_type.list_size
    columns = _bound_columns(batch, field_spec, field_spec.source_mnemonics[0])
    return _fixed_list_from_columns(
        columns, field_spec.data_type.value_type, size, field_spec.name
    )


def _atms_secondary_array(batch: pa.RecordBatch, field_spec: FieldSpec) -> pa.Array:
    if field_spec.name != "channel_auxiliary":
        return _simple_secondary_array(batch, field_spec)
    value_type = field_spec.data_type.value_type
    flattened = _flatten_container(batch, field_spec, expected_size=22)
    if len(field_spec.source_mnemonics) != len(value_type):
        raise ValueError("ATMS channel-auxiliary source/target arity changed")
    children = [
        _checked_cast(
            flattened.field(source),
            child.type,
            f"{field_spec.name}.{child.name}",
        )
        for source, child in zip(field_spec.source_mnemonics, value_type, strict=True)
    ]
    return pa.FixedSizeListArray.from_arrays(
        pa.StructArray.from_arrays(children, fields=list(value_type)), 22
    )


def _cris_structural_band_quality(
    batch: pa.RecordBatch, field_spec: FieldSpec
) -> pa.FixedSizeListArray:
    _, source = _structural_fixed_group(batch, _CRIS_BAND_QUALITY_GROUP)
    source_values = source.values
    value_type = field_spec.data_type.value_type
    arrays = [
        _checked_cast(
            source_values.field(index),
            child.type,
            f"{field_spec.name}.{child.name}",
        )
        for index, child in enumerate(value_type)
    ]
    values = pa.StructArray.from_arrays(arrays, fields=list(value_type))
    return pa.FixedSizeListArray.from_arrays(values, 3)


def _cris_structural_imager_scenes(
    batch: pa.RecordBatch, field_spec: FieldSpec
) -> pa.FixedSizeListArray:
    # Children follow _CRIS_IMAGER_GROUP: scene YAPCG, ZAPCG, FCPH, channels;
    # channel CHNM, FOST, SCHRAD, FOST, SCHRAD.
    _, source = _structural_fixed_group(batch, _CRIS_IMAGER_GROUP)
    source_scenes = source.values
    source_channels = source_scenes.field(3)
    source_channel_values = source_channels.values

    scene_type = field_spec.data_type.value_type
    channel_type = scene_type.field("channels").type.value_type
    statistic_type = channel_type.field("statistics").type.value_type
    statistics = pa.StructArray.from_arrays(
        [
            _interleave_columns(
                [source_channel_values.field(1), source_channel_values.field(3)],
                statistic_type.field("statistic_type").type,
                "collocated_imager_scenes.statistic_type",
            ),
            _interleave_columns(
                [source_channel_values.field(2), source_channel_values.field(4)],
                statistic_type.field("radiance").type,
                "collocated_imager_scenes.radiance",
            ),
        ],
        fields=list(statistic_type),
    )
    statistic_lists = pa.FixedSizeListArray.from_arrays(statistics, 2)
    channels = pa.StructArray.from_arrays(
        [
            _checked_cast(
                source_channel_values.field(0),
                channel_type.field("channel_number").type,
                "collocated_imager_scenes.channel_number",
            ),
            statistic_lists,
        ],
        fields=list(channel_type),
    )
    channel_lists = pa.FixedSizeListArray.from_arrays(channels, 16)
    scenes = pa.StructArray.from_arrays(
        [
            _checked_cast(
                source_scenes.field(index),
                scene_type.field(name).type,
                f"collocated_imager_scenes.{name}",
            )
            for index, name in enumerate(("y_angle", "z_angle", "clear_fraction"))
        ]
        + [channel_lists],
        fields=list(scene_type),
    )
    return pa.FixedSizeListArray.from_arrays(scenes, 7)


def _cris_secondary_array(batch: pa.RecordBatch, field_spec: FieldSpec) -> pa.Array:
    if field_spec.name == "band_calibration_quality":
        return _cris_structural_band_quality(batch, field_spec)
    if field_spec.name == "collocated_imager_scenes":
        return _cris_structural_imager_scenes(batch, field_spec)
    if field_spec.name == "spacecraft_position_m":
        return _fixed_list_from_columns(
            [_column(batch, mnemonic) for mnemonic in field_spec.source_mnemonics],
            field_spec.data_type.value_type,
            3,
            field_spec.name,
        )
    if field_spec.name == "guard_spectrum":
        value_type = field_spec.data_type.value_type
        flattened = _flatten_container(batch, field_spec, expected_size=12)
        if len(field_spec.source_mnemonics) != len(value_type):
            raise ValueError("CrIS guard-spectrum source/target arity changed")
        struct = pa.StructArray.from_arrays(
            [
                _checked_cast(
                    flattened.field(source), child.type, f"guard_spectrum.{child.name}"
                )
                for source, child in zip(
                    field_spec.source_mnemonics, value_type, strict=True
                )
            ],
            fields=list(value_type),
        )
        return pa.FixedSizeListArray.from_arrays(struct, 12)
    return _simple_secondary_array(batch, field_spec)


def _iasi_avhrr_scenes(
    batch: pa.RecordBatch, field_spec: FieldSpec
) -> pa.FixedSizeListArray:
    scene_type = field_spec.data_type.value_type
    channel_type = scene_type.field("channels").type.value_type
    scales = _bound_columns(batch, field_spec, "CHSF")
    channel_children = (
        ("channel_number", _bound_columns(batch, field_spec, "CHNM")),
        ("mean_scale_exponent", scales[0::2]),
        ("std_scale_exponent", scales[1::2]),
        ("mean_radiance", _bound_columns(batch, field_spec, "SMRA")),
        ("std_radiance", _bound_columns(batch, field_spec, "SSDR")),
    )
    channel_struct = pa.StructArray.from_arrays(
        [
            _interleave_columns(
                columns,
                channel_type.field(name).type,
                f"collocated_avhrr_scenes.{name}",
            )
            for name, columns in channel_children
        ],
        fields=list(channel_type),
    )
    channel_lists = pa.FixedSizeListArray.from_arrays(channel_struct, 6)
    scene_struct = pa.StructArray.from_arrays(
        [
            _interleave_columns(
                _bound_columns(batch, field_spec, mnemonic),
                scene_type.field(target).type,
                f"collocated_avhrr_scenes.{target}",
            )
            for mnemonic, target in (
                ("YAPCG", "y_angle"),
                ("ZAPCG", "z_angle"),
                ("FCPH", "clear_fraction"),
            )
        ]
        + [channel_lists],
        fields=list(scene_type),
    )
    return pa.FixedSizeListArray.from_arrays(scene_struct, 7)


def _iasi_secondary_array(batch: pa.RecordBatch, field_spec: FieldSpec) -> pa.Array:
    if field_spec.name == "main_scale_groups":
        value_type = field_spec.data_type.value_type
        return _fixed_struct_list(
            [
                (
                    target,
                    _bound_columns(batch, field_spec, mnemonic),
                    value_type.field(target).type,
                )
                for mnemonic, target in (
                    ("STCH", "start_channel"),
                    ("ENCH", "end_channel"),
                    ("CHSF", "scale_exponent"),
                )
            ],
            size=10,
            name=field_spec.name,
        )
    if field_spec.name == "collocated_avhrr_scenes":
        return _iasi_avhrr_scenes(batch, field_spec)
    return _simple_secondary_array(batch, field_spec)


def _airs_header(batch: pa.RecordBatch, field_spec: FieldSpec) -> pa.StructArray:
    header_type = field_spec.data_type
    time_type = header_type.field("source_time").type
    leaf_types = [
        header_type.field("instrument_id").type,
        *(child.type for child in time_type),
        *(child.type for child in list(header_type)[2:]),
    ]
    values = []
    for mnemonic, data_type in zip(
        field_spec.source_mnemonics, leaf_types, strict=True
    ):
        columns = _bound_columns(batch, field_spec, mnemonic)
        if len(columns) != 1:
            raise ValueError(f"{field_spec.name}/{mnemonic} needs one occurrence")
        values.append(
            _checked_cast(columns[0], data_type, f"{field_spec.name}.{mnemonic}")
        )
    source_time = pa.StructArray.from_arrays(values[1:7], fields=list(time_type))
    return pa.StructArray.from_arrays(
        [values[0], source_time, *values[7:]], fields=list(header_type)
    )


def _airs_secondary_array(batch: pa.RecordBatch, field_spec: FieldSpec) -> pa.Array:
    if field_spec.name == "instrument_temperatures":
        value_type = field_spec.data_type.value_type
        return _fixed_struct_list(
            [
                (
                    "radiometer_id",
                    _bound_columns(batch, field_spec, "RAID"),
                    value_type.field("radiometer_id").type,
                ),
                (
                    "temperature_k",
                    _bound_columns(batch, field_spec, "TMINST"),
                    value_type.field("temperature_k").type,
                ),
            ],
            size=9,
            name=field_spec.name,
        )
    if field_spec.name == "channel_auxiliary":
        value_type = field_spec.data_type.value_type
        flattened = _flatten_container(batch, field_spec, expected_size=281)
        if len(field_spec.source_mnemonics) != len(value_type):
            raise ValueError("AIRS channel-auxiliary source/target arity changed")
        struct = pa.StructArray.from_arrays(
            [
                _checked_cast(
                    flattened.field(source),
                    child.type,
                    f"{field_spec.name}.{child.name}",
                )
                for source, child in zip(
                    field_spec.source_mnemonics, value_type, strict=True
                )
            ],
            fields=list(value_type),
        )
        return pa.FixedSizeListArray.from_arrays(struct, 281)
    if field_spec.name == "visible_statistics":
        value_type = field_spec.data_type.value_type
        statistic_types = _bound_columns(batch, field_spec, "FOST")
        albedos = _bound_columns(batch, field_spec, "ALBD")
        children = (
            ("channel_number", _bound_columns(batch, field_spec, "CHNM")),
            ("log10_wavenumber", _bound_columns(batch, field_spec, "LOGRCW")),
            ("acquisition_quality", _bound_columns(batch, field_spec, "ACQF")),
            ("statistic_type_1", statistic_types[0::3]),
            ("albedo_1", albedos[0::2]),
            ("statistic_type_2", statistic_types[1::3]),
            ("albedo_2", albedos[1::2]),
            ("statistic_type_3", statistic_types[2::3]),
        )
        return _fixed_struct_list(
            [
                (name, columns, value_type.field(name).type)
                for name, columns in children
            ],
            size=4,
            name=field_spec.name,
        )
    if field_spec.name in {"companion_amsua_header", "companion_hsb_header"}:
        return _airs_header(batch, field_spec)
    if field_spec.name in {"companion_amsua_channels", "companion_hsb_channels"}:
        size = field_spec.data_type.list_size
        value_type = field_spec.data_type.value_type
        children = (
            ("channel_number", "CHNM"),
            ("log10_wavenumber", "LOGRCW"),
            ("acquisition_quality", "ACQF"),
            ("brightness_temperature_k", "TMBR"),
        )
        return _fixed_struct_list(
            [
                (
                    target,
                    _bound_columns(batch, field_spec, source),
                    value_type.field(target).type,
                )
                for target, source in children
            ],
            size=size,
            name=field_spec.name,
        )
    return _simple_secondary_array(batch, field_spec)


def convert_sounder_secondary_batch(batch: pa.RecordBatch, *, sensor: str) -> pa.Table:
    """Convert every non-common, non-primary field in the sensor contract."""

    converter = {
        "amsua": _atovs_secondary_array,
        "amsub": _atovs_secondary_array,
        "mhs": _atovs_secondary_array,
        "atms": _atms_secondary_array,
        "cris": _cris_secondary_array,
        "iasi": _iasi_secondary_array,
        "airs": _airs_secondary_array,
    }[sensor]
    fields = [
        field
        for field in SENSOR_SPECS[sensor].fields
        if field.name not in COMMON_FIELD_NAMES
    ]
    arrays = []
    for field in fields:
        with cpu_timing_range(f"secondary.{field.name}"):
            direct_structural = sensor == "cris" and field.name in {
                "band_calibration_quality",
                "collocated_imager_scenes",
            }
            if not direct_structural and not _field_source_present(batch, field):
                if not field.source_optional:
                    raise ValueError(f"required archive field {field.name!r} is absent")
                arrays.append(_null_array(batch.num_rows, field.data_type))
            else:
                arrays.append(converter(batch, field))
    for field, array in zip(fields, arrays, strict=True):
        if array.type != field.data_type:
            raise TypeError(
                f"{sensor}/{field.name} produced {array.type}, expected "
                f"{field.data_type}"
            )
    return pa.Table.from_arrays(arrays, names=[field.name for field in fields])


def validate_decoder_field_coverage(batch: pa.RecordBatch, *, sensor: str) -> None:
    """Fail when a decoded source occurrence is outside the archive contract."""

    spec = SENSOR_SPECS[sensor]
    positions: dict[str, list[int]] = {}
    for position, field in enumerate(batch.schema):
        positions.setdefault(field.name, []).append(position)
    claimed: set[int] = {
        position
        for name in ("_source_message_index", "_source_subset_index", "_data_category")
        for position in positions.get(name, ())
    }
    if sensor == "cris":
        for group in (_CRIS_BAND_QUALITY_GROUP, _CRIS_IMAGER_GROUP):
            claimed.add(_structural_fixed_group(batch, group)[0])
    for name in spec.ignored_source_fields:
        for position in positions.get(name, ()):
            array = batch.column(position)
            if name == "BYTCNT":
                values = np.asarray(
                    array.to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )
                if (
                    array.null_count
                    or np.any(~np.isfinite(values))
                    or np.any(values <= 0)
                ):
                    raise ValueError("BYTCNT framing values are invalid")
            elif name == "DRF8BIT":
                if not (
                    pa.types.is_list(array.type) or pa.types.is_large_list(array.type)
                ) or not pa.types.is_struct(array.type.value_type):
                    raise ValueError("DRF8BIT padding has an unexpected type")
                children = [field.name for field in array.type.value_type]
                if children != ["LOCAL_0-63-255"]:
                    raise ValueError(f"DRF8BIT padding children changed: {children}")
                padding = pc.list_flatten(array).field("LOCAL_0-63-255")
                values = np.asarray(
                    padding.to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )
                if (
                    padding.null_count
                    or np.any(~np.isfinite(values))
                    or np.any(values != 0)
                ):
                    raise ValueError("DRF8BIT local padding is not all zero")
            claimed.add(position)

    def claim(mnemonic: str, indices: Sequence[int] | None) -> None:
        available = positions.get(mnemonic, [])
        selected = (
            range(len(available))
            if indices is None
            else _structural_top_level_occurrence_indices(batch, mnemonic, indices)
        )
        for occurrence in selected:
            if occurrence < 0 or occurrence >= len(available):
                raise ValueError(
                    f"{sensor}/{mnemonic} occurrence {occurrence} is absent; "
                    f"decoder produced {len(available)}"
                )
            claimed.add(available[occurrence])

    containers: dict[str, set[str]] = {}
    for field_spec in spec.fields:
        if not field_spec.source_mnemonics and field_spec.source_container is None:
            continue
        if sensor == "cris" and field_spec.name in {
            "band_calibration_quality",
            "collocated_imager_scenes",
        }:
            continue
        if not _field_source_present(batch, field_spec):
            if field_spec.source_optional:
                continue
            raise ValueError(f"required archive field {field_spec.name!r} is absent")
        if field_spec.source_container is not None:
            claim(field_spec.source_container, None)
            containers.setdefault(field_spec.source_container, set()).update(
                field_spec.source_mnemonics
            )
            continue
        for mnemonic in field_spec.source_mnemonics:
            indices = field_spec.source_occurrence_indices.get(mnemonic)
            if indices is not None:
                claim(mnemonic, indices)
            elif pa.types.is_nested(field_spec.data_type):
                claim(mnemonic, None)
            else:
                claim(mnemonic, (0,))
    for leaf in spec.primary_leaves:
        if leaf.source_container is None:
            for mnemonic in (*spec.manifest_mnemonics, leaf.source_mnemonic):
                claim(mnemonic, None)
        else:
            claim(leaf.source_container, None)
            containers.setdefault(leaf.source_container, set()).update(
                (*spec.manifest_mnemonics, leaf.source_mnemonic)
            )

    for container, expected_children in containers.items():
        arrays = _columns(batch, container)
        if len(arrays) != 1:
            raise ValueError(
                f"{sensor}/{container} needs one decoded container, found {len(arrays)}"
            )
        actual_children = {field.name for field in arrays[0].type.value_type}
        if actual_children != expected_children:
            raise ValueError(
                f"{sensor}/{container} child fields changed: produced "
                f"{sorted(actual_children)}, contract has {sorted(expected_children)}"
            )

    unclaimed = [
        (position, field.name)
        for position, field in enumerate(batch.schema)
        if position not in claimed
    ]
    if unclaimed:
        raise ValueError(
            f"{sensor} decoder fields are not represented by archive_schema: {unclaimed}"
        )


def _decoder_arrow_shape(
    field: pa.Field,
    raw_code_bindings: Mapping[str, PrimaryValueEncoding],
    *,
    path: str | None = None,
) -> tuple[Any, ...]:
    path = field.name if path is None else path
    data_type = field.type
    if path in raw_code_bindings:
        if data_type != pa.int32():
            raise TypeError(f"raw-code path {path} arrived as {data_type}, not int32")
        _validate_decoder_raw_code_field(field, raw_code_bindings[path])
        return ("float64",)
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return ("utf8",)
    if pa.types.is_float64(data_type):
        return ("float64",)
    if (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    ):
        value_type = data_type.value_type
        if not pa.types.is_struct(value_type):
            value_field = pa.field("item", value_type)
            return (
                "list",
                _decoder_arrow_shape(
                    value_field,
                    raw_code_bindings,
                    path=f"{path}.item",
                ),
            )
        return (
            "list<struct>",
            tuple(
                (
                    child.name,
                    _decoder_arrow_shape(
                        child,
                        raw_code_bindings,
                        path=f"{path}.{child.name}",
                    ),
                )
                for child in value_type
            ),
        )
    return (str(data_type),)


def _source_arrow_shape(record: Mapping[str, Any]) -> tuple[Any, ...]:
    arrow_type = {"double": "float64", "string": "utf8"}.get(
        str(record["arrow_type"]), str(record["arrow_type"])
    )
    if arrow_type != "list<struct>":
        return (arrow_type,)
    return (
        arrow_type,
        tuple(
            (str(child["name"]), _source_arrow_shape(child))
            for child in record.get("children", ())
        ),
    )


def validate_source_structure_binding(
    batch: pa.RecordBatch,
    *,
    sensor: str,
    source_structure: SourceStructure,
) -> None:
    """Bind decoder columns and static occurrence mappings to the actual template."""

    source_structure.validate()
    if source_structure.sensor != sensor:
        raise ValueError(
            f"{source_structure.sensor} source structure cannot bind {sensor}"
        )
    source_fields = sorted(
        source_structure.decoded_physical_schema,
        key=lambda record: int(record["position"]),
    )
    decoder_fields = [
        field
        for field in _flatten_structural_schema_fields(batch.schema)
        if field.name
        not in {"_data_category", "_source_message_index", "_source_subset_index"}
    ]
    if len(source_fields) != len(decoder_fields):
        raise ValueError(
            f"{sensor} actual source schema has {len(source_fields)} fields but "
            f"the decoder batch has {len(decoder_fields)}"
        )
    raw_code_paths = _primary_raw_code_paths(sensor)
    if raw_code_paths is None:
        raw_code_bindings: dict[str, PrimaryValueEncoding] = {}
    else:
        encoding = derive_cris_primary_encoding(source_structure)
        raw_code_bindings = dict.fromkeys(raw_code_paths, encoding)
    for position, (source_field, decoder_field) in enumerate(
        zip(source_fields, decoder_fields, strict=True)
    ):
        if int(source_field["position"]) != position:
            raise ValueError(f"{sensor} source schema positions are not contiguous")
        expected = (str(source_field["name"]), _source_arrow_shape(source_field))
        actual = (
            decoder_field.name,
            _decoder_arrow_shape(decoder_field, raw_code_bindings),
        )
        if actual != expected:
            raise ValueError(
                f"{sensor} decoder field {position} disagrees with actual source "
                f"structure: expected {expected}, got {actual}"
            )

    spec = SENSOR_SPECS[sensor]
    required_mnemonics = {
        mnemonic
        for field_spec in spec.fields
        if not field_spec.source_optional
        for mnemonic in (
            *field_spec.source_mnemonics,
            *(
                ()
                if field_spec.source_container is None
                else (field_spec.source_container,)
            ),
        )
    }
    for leaf in spec.primary_leaves:
        required_mnemonics.add(leaf.source_mnemonic)
        if leaf.source_container is not None:
            required_mnemonics.add(leaf.source_container)
    required_mnemonics.update(spec.manifest_mnemonics)
    actual_mnemonics = {
        str(record["mnemonic"])
        for record in source_structure.effective_occurrences
        if record.get("mnemonic") is not None
    }
    missing = required_mnemonics - actual_mnemonics
    if missing:
        raise ValueError(
            f"{sensor} actual source structure is missing contract mnemonics "
            f"{sorted(missing)}"
        )


def convert_cris_primary_batch(
    batch: pa.RecordBatch,
    *,
    source_structure: SourceStructure,
    release: str,
    expected_manifest: ChannelManifest | None = None,
    primary_encoding: PrimaryValueEncoding | None = None,
) -> SounderPrimaryBatch:
    """Convert one bounded raw decoder batch to packed wide CrIS primary columns."""

    if batch.num_rows == 0:
        raise ValueError("CrIS decoder batch contains no source rows")

    spec = SENSOR_SPECS["cris"]
    encoding = primary_encoding or derive_cris_primary_encoding(source_structure)

    with cpu_timing_range("primary.flatten_axis"):
        flattened, channel_matrix = _list_primary_parts(batch, sensor="cris")
    with cpu_timing_range("primary.validate_manifest"):
        manifest = _manifest_from_channel_matrix(
            channel_matrix,
            sensor="cris",
            release=release,
            expected_manifest=expected_manifest,
        )
    channel_ids = [channel.source_channel_id for channel in manifest.channels]

    leaf = spec.primary_leaves[0]
    packed_codes = flattened.field(leaf.source_mnemonic)
    if packed_codes.type != pa.int32():
        raise TypeError(
            f"CrIS {leaf.source_mnemonic} must arrive as int32 raw codes, "
            f"got {packed_codes.type}"
        )
    _validate_decoder_raw_code_field(
        flattened.type.field(leaf.source_mnemonic), encoding
    )
    with cpu_timing_range("primary.transpose"):
        if packed_codes.null_count:
            flat_codes = np.asarray(
                pc.fill_null(packed_codes, 0).to_numpy(zero_copy_only=False),
                dtype=np.int32,
            )
            flat_nulls: np.ndarray | None = np.asarray(packed_codes.is_null())
        else:
            flat_codes = packed_codes.to_numpy(zero_copy_only=True)
            flat_nulls = None
        channel_major_codes = np.ascontiguousarray(
            flat_codes.reshape(batch.num_rows, spec.channel_count).T
        )
        channel_major_nulls = (
            None
            if flat_nulls is None
            else np.ascontiguousarray(
                flat_nulls.reshape(batch.num_rows, spec.channel_count).T
            )
        )

    arrays: list[pa.Array] = []
    names: list[str] = []
    with cpu_timing_range("primary.arrow_columns"):
        for position, channel_id in enumerate(channel_ids):
            arrays.append(
                pa.array(
                    channel_major_codes[position],
                    mask=(
                        None
                        if channel_major_nulls is None
                        else channel_major_nulls[position]
                    ),
                    type=pa.int32(),
                )
            )
            names.append(primary_field_name(leaf, channel_id))
    return SounderPrimaryBatch(
        sensor="cris",
        table=pa.Table.from_arrays(arrays, names=names),
        channel_manifest=manifest,
    )


def convert_sounder_primary_batch(
    batch: pa.RecordBatch,
    *,
    sensor: str,
    release: str,
    source_structure: SourceStructure,
    expected_manifest: ChannelManifest | None = None,
    primary_encoding: PrimaryValueEncoding | None = None,
) -> SounderPrimaryBatch:
    """Convert the hot primary channel block for any archive sounder."""

    if sensor in ATOVS_SENSORS:
        return _convert_atovs_primary_batch(
            batch,
            sensor=sensor,
            release=release,
            expected_manifest=expected_manifest,
        )
    if sensor in {"atms", "airs"}:
        return _convert_list_float_primary_batch(
            batch,
            sensor=sensor,
            release=release,
            expected_manifest=expected_manifest,
        )
    if sensor == "iasi":
        return _convert_iasi_primary_batch(
            batch,
            release=release,
            expected_manifest=expected_manifest,
        )
    if sensor == "cris":
        return convert_cris_primary_batch(
            batch,
            source_structure=source_structure,
            release=release,
            expected_manifest=expected_manifest,
            primary_encoding=primary_encoding,
        )
    raise ValueError(f"unsupported sensor {sensor!r}")


def compile_archive_layout(
    *,
    sensor: str,
    channel_manifest: ChannelManifest,
    source_structure: SourceStructure,
) -> CompiledArchiveLayout:
    """Compile immutable schema state once for one resolved source release."""

    schema = archive_schema(
        sensor,
        channel_manifest,
        source_structure=source_structure,
    )
    primary_encoding = (
        derive_cris_primary_encoding(source_structure) if sensor == "cris" else None
    )
    return CompiledArchiveLayout(
        sensor=sensor,
        schema=schema,
        channel_manifest=channel_manifest,
        primary_encoding=primary_encoding,
        output_names=tuple(schema.names),
        source_structure=source_structure,
    )


def validate_archive_layout_compatibility(
    reference: CompiledArchiveLayout,
    candidate: CompiledArchiveLayout,
) -> None:
    """Require two source layouts to share one writer contract."""

    reference_fields = tuple(
        (field.name, field.type, field.nullable) for field in reference.schema
    )
    candidate_fields = tuple(
        (field.name, field.type, field.nullable) for field in candidate.schema
    )
    if candidate_fields != reference_fields:
        raise ValueError("source structures produce different logical archive fields")
    if (
        candidate.channel_manifest.as_payload()
        != reference.channel_manifest.as_payload()
    ):
        raise ValueError("source structures produce different channel manifests")
    if candidate.primary_encoding != reference.primary_encoding:
        raise ValueError("source structures produce different primary encodings")


def _record_batch_arrays(table: pa.Table) -> list[pa.Array]:
    arrays: list[pa.Array] = []
    for column in table.columns:
        if column.num_chunks != 1:
            raise ValueError(
                "batch-local conversion unexpectedly produced chunked data"
            )
        arrays.append(column.chunk(0))
    return arrays


def convert_sounder_archive_batch(
    batch: pa.RecordBatch,
    *,
    sensor: str,
    cycle: datetime,
    source_file_ordinal: int,
    message_reference_times: Mapping[int, datetime],
    release: str,
    source_structure: SourceStructure,
    expected_manifest: ChannelManifest | None = None,
    compiled_layout: CompiledArchiveLayout | None = None,
) -> SounderArchiveBatch:
    """Convert one bounded decoder batch to the complete logical archive schema."""

    if compiled_layout is not None:
        if compiled_layout.sensor != sensor:
            raise ValueError(
                f"{compiled_layout.sensor} layout cannot convert {sensor} batches"
            )
        if (
            source_structure is not compiled_layout.source_structure
            and source_structure != compiled_layout.source_structure
        ):
            raise ValueError(
                "source structure changed after archive layout compilation"
            )
        if (
            expected_manifest is not None
            and expected_manifest.as_payload()
            != compiled_layout.channel_manifest.as_payload()
        ):
            raise ValueError("expected manifest disagrees with compiled archive layout")
        expected_manifest = compiled_layout.channel_manifest

    with cpu_timing_range("convert.coverage"):
        validate_decoder_field_coverage(batch, sensor=sensor)
    with cpu_timing_range("convert.common"):
        common = convert_common_footprint_batch(
            batch,
            sensor=sensor,
            cycle=cycle,
            source_file_ordinal=source_file_ordinal,
            message_reference_times=message_reference_times,
        )
    with cpu_timing_range("convert.secondary"):
        secondary = convert_sounder_secondary_batch(batch, sensor=sensor)
    with cpu_timing_range("convert.primary"):
        primary = convert_sounder_primary_batch(
            batch,
            sensor=sensor,
            release=release,
            source_structure=source_structure,
            expected_manifest=expected_manifest,
            primary_encoding=(
                None if compiled_layout is None else compiled_layout.primary_encoding
            ),
        )
    with cpu_timing_range("convert.layout"):
        layout = compiled_layout or compile_archive_layout(
            sensor=sensor,
            channel_manifest=primary.channel_manifest,
            source_structure=source_structure,
        )
    with cpu_timing_range("convert.assemble"):
        arrays = [
            *_record_batch_arrays(common),
            *_record_batch_arrays(secondary),
            *_record_batch_arrays(primary.table),
        ]
        names = (
            *common.column_names,
            *secondary.column_names,
            *primary.table.column_names,
        )
    if names != layout.output_names:
        raise ValueError(
            f"{sensor} ETL field order does not match archive_schema: "
            f"produced {names}, expected {layout.output_names}"
        )
    with cpu_timing_range("convert.assemble"):
        record_batch = pa.record_batch(arrays, schema=layout.schema)
        table = pa.Table.from_batches([record_batch], schema=layout.schema)
    return SounderArchiveBatch(
        sensor=sensor,
        table=table,
        channel_manifest=primary.channel_manifest,
        layout=layout,
    )


def iter_open_sounder_archive_batches(
    source: Earth2BufrSource,
    *,
    sensor: str,
    cycle: datetime,
    source_file_ordinal: int,
    release: str,
    structures: Mapping[int, SourceStructure],
) -> Iterator[SounderArchiveBatch]:
    """Convert one opened source into complete archive batches.

    ``structures`` maps each data message index to its template's structure, as
    returned by ``Earth2BufrSource.source_structure_bindings``.
    """

    if any(structure.sensor != sensor for structure in structures.values()):
        raise ValueError(f"source structure cannot decode {sensor}")
    message_reference_times = {
        header.index: header.reference_time
        for header in source.message_headers
        if header.reference_time is not None
    }
    layouts: dict[int, CompiledArchiveLayout] = {}
    for raw_batch in source.iter_batches(sensor=sensor):
        message_indices = pc.unique(raw_batch.column("_source_message_index"))
        try:
            batch_structures = {
                id(structures[index]): structures[index]
                for index in message_indices.to_pylist()
            }
        except KeyError as exc:
            raise ValueError(
                f"decoder batch references unbound source message {exc.args[0]}"
            ) from exc
        if len(batch_structures) != 1:
            raise ValueError("bounded decoder batch spans source structures")
        [(key, batch_structure)] = batch_structures.items()
        layout = layouts.get(key)
        if layout is None:
            with cpu_timing_range("convert.bind_structure"):
                validate_source_structure_binding(
                    raw_batch,
                    sensor=sensor,
                    source_structure=batch_structure,
                )
        converted = convert_sounder_archive_batch(
            raw_batch,
            sensor=sensor,
            cycle=cycle,
            source_structure=batch_structure,
            source_file_ordinal=source_file_ordinal,
            release=release,
            message_reference_times=message_reference_times,
            compiled_layout=layout,
        )
        layouts.setdefault(key, converted.layout)
        yield converted
