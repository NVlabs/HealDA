#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical contract for the source-faithful NNJA sounder Parquet archive.

The archive is application-neutral: it is neither an Earth2Studio public query
schema nor a model-facing long table. Sensor evidence gates that remain open
are recorded by ``complete_template_evidence`` rather than hidden by the API.
"""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
from healda.observations import schema as reader_schema

from ..archive_common.descriptor_catalog import Descriptor, load_descriptor_catalog

CONTRACT_NAME = "ncep-nnja-native-sounder-archive"
CONTRACT_VERSION = "0.3.0-rc5"
LEGACY_CONTRACT_VERSION = "0.3.0-rc3"
FOOTPRINT_ID_DOMAIN = b"ncep-nnja-footprint-v1\x00"
CATALOG_PATH = Path(__file__).resolve().parent / "descriptor_catalog.csv"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _encoded_metadata(value: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {key.encode(): _canonical_json(item).encode() for key, item in value.items()}


# Base Table-B packing recorded per source descriptor in field metadata.
_DESCRIPTOR_RECORD_FIELDS = (
    "descriptor",
    "mnemonic",
    "meaning",
    "units",
    "base_scale",
    "base_reference",
    "base_bits",
    "value_kind",
    "interpretation_table_key",
)


def _descriptor_record(descriptor: Descriptor) -> dict[str, Any]:
    return {name: getattr(descriptor, name) for name in _DESCRIPTOR_RECORD_FIELDS}


DESCRIPTORS = load_descriptor_catalog(CATALOG_PATH)


@dataclass(frozen=True)
class Channel:
    """Stable identity and optional interpretation metadata for one channel."""

    source_channel_id: int
    source_position: int
    # Never set, so always null in the footer and field metadata. Fix with the next contract version: schema_id covers this text.
    frequency_hz: float | None = None
    wavenumber_cm_inverse: float | None = None


@dataclass(frozen=True)
class ChannelManifest:
    """Ordered channel axis stored in every file footer."""

    sensor: str
    channels: tuple[Channel, ...]
    release: str

    @classmethod
    def from_ids(
        cls, sensor: str, channel_ids: Sequence[int], *, release: str
    ) -> ChannelManifest:
        return cls(
            sensor=sensor,
            channels=tuple(
                Channel(source_channel_id=int(channel), source_position=position)
                for position, channel in enumerate(channel_ids)
            ),
            release=release,
        )

    def validate(self, expected_count: int) -> None:
        if self.sensor not in SENSOR_SPECS:
            raise ValueError(f"unknown channel-manifest sensor: {self.sensor}")
        if len(self.channels) != expected_count:
            raise ValueError(
                f"{self.sensor} needs {expected_count} channels, got "
                f"{len(self.channels)}"
            )
        positions = [channel.source_position for channel in self.channels]
        if positions != list(range(expected_count)):
            raise ValueError("source_position must be contiguous and ordered")
        ids = [channel.source_channel_id for channel in self.channels]
        if len(set(ids)) != len(ids):
            raise ValueError("source_channel_id values must be unique")

    def as_payload(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "release": self.release,
            "channels": [asdict(channel) for channel in self.channels],
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_payload()).encode()).hexdigest()


@dataclass(frozen=True)
class FieldSpec:
    """Logical Arrow field and its ownership/source binding."""

    name: str
    data_type: pa.DataType
    source_mnemonics: tuple[str, ...]
    role: str
    description: str
    nullable: bool = True
    representation: str = "physical"
    occurrence_scope: str = "all occurrences represented by this logical field"
    data_class: str | None = None
    axes: tuple[Axis, ...] = ()
    source_occurrence_indices: Mapping[str, tuple[int, ...]] = field(
        default_factory=dict
    )
    source_container: str | None = None
    source_optional: bool = False

    def arrow_field(self, sensor: str) -> pa.Field:
        descriptor_rows = []
        for mnemonic in self.source_mnemonics:
            try:
                descriptor_rows.append(
                    _descriptor_record(DESCRIPTORS[sensor][mnemonic])
                )
            except KeyError as exc:
                raise KeyError(f"{sensor}/{self.name}: unknown {mnemonic}") from exc
        data_class = self.data_class or ROLE_DATA_CLASS[self.role]
        if data_class not in DATA_CLASSES:
            raise ValueError(f"{sensor}/{self.name}: invalid data class {data_class}")
        metadata = _encoded_metadata(
            {
                "nnja.archive.role": self.role,
                "nnja.archive.data_class": data_class,
                "nnja.archive.description": self.description,
                "nnja.archive.representation": self.representation,
                "nnja.archive.source_descriptors": descriptor_rows,
                "nnja.archive.source_occurrences": self.occurrence_scope,
                "nnja.archive.source_occurrence_indices": dict(
                    self.source_occurrence_indices
                ),
                "nnja.archive.source_container": self.source_container,
                "nnja.archive.axes": [asdict(axis) for axis in self.axes],
            }
        )
        return pa.field(
            self.name, self.data_type, nullable=self.nullable, metadata=metadata
        )


# A field the healda reader declares: its Arrow type and nullability are the reader's.
def _base_field(sensor: str, name: str, *args: Any, **kwargs: Any) -> FieldSpec:
    base = pa.schema(reader_schema.footprint_fields(sensor)).field(name)
    return FieldSpec(name, base.type, *args, nullable=base.nullable, **kwargs)


def _reader_value_type(sensor: str) -> pa.DataType:
    return reader_schema.channel_field(sensor, 0, kelvin=False).type


@dataclass(frozen=True)
class Axis:
    """Named fixed axis within one grouped footprint field."""

    name: str
    length: int
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("axis name must not be empty")
        if self.length <= 0:
            raise ValueError("axis length must be positive")


DATA_CLASSES = frozenset(
    {
        "primary_data",
        "primary_descriptor",
        "secondary_data",
        "secondary_descriptor",
        "provenance",
        "routing",
    }
)

ROLE_DATA_CLASS = {
    "band_definition": "secondary_descriptor",
    "band_quality": "secondary_descriptor",
    "channel_auxiliary": "secondary_descriptor",
    "collocated_imager": "secondary_data",
    "companion_instrument": "secondary_data",
    "footprint_identity": "primary_descriptor",
    "geometry": "primary_descriptor",
    "guard_spectrum": "secondary_data",
    "housekeeping": "secondary_data",
    "platform_state": "secondary_descriptor",
    "processing_provenance": "provenance",
    "provenance": "provenance",
    "quality": "primary_descriptor",
    "routing": "routing",
    "scan_identity": "primary_descriptor",
    "scene_context": "primary_descriptor",
    "source_fidelity": "provenance",
    "spectral_scaling": "secondary_descriptor",
    "surface_context": "primary_descriptor",
    "visible_payload": "secondary_data",
}


@dataclass(frozen=True)
class PrimaryLeaf:
    """One hot per-channel value widened into primitive Parquet columns."""

    prefix: str
    source_mnemonic: str
    data_type: pa.DataType
    units: str
    description: str
    representation: str = "physical"
    source_container: str | None = None


@dataclass(frozen=True)
class PrimaryValueEncoding:
    """Machine-readable transform from stored primary codes to physical values."""

    id: str
    storage_dtype: str
    descriptor: str
    scale: int
    reference: int
    bit_width: int
    source_missing_code: int
    null_representation: str
    physical_dtype: str
    source_units: str
    physical_units: str
    formula: str


@dataclass(frozen=True)
class Exclusion:
    """A source construct intentionally not represented as a value column."""

    source: str
    reason: str
    disposition: str


@dataclass(frozen=True)
class SensorArchiveSpec:
    """Versioned logical layout for one active NNJA sounder template."""

    sensor: str
    display_name: str
    active_template: str
    channel_count: int
    complete_template_evidence: str
    fields: tuple[FieldSpec, ...]
    primary_leaves: tuple[PrimaryLeaf, ...]
    exclusions: tuple[Exclusion, ...]
    manifest_mnemonics: tuple[str, ...] = ()
    primary_value_encoding: PrimaryValueEncoding | None = None
    ignored_source_fields: tuple[str, ...] = ()

    @property
    def structurally_represented_mnemonics(self) -> set[str]:
        return {
            exclusion.source
            for exclusion in self.exclusions
            if exclusion.source in DESCRIPTORS[self.sensor]
        }


ATOVS_SENSORS = frozenset({"amsua", "amsub", "mhs"})

# One companion footprint header in AIRS, in FOOTPRINT_HEADER leaf order.
AIRS_COMPANION_HEADER_MNEMONICS = (
    "SIID",
    "YEAR",
    "MNTH",
    "DAYS",
    "HOUR",
    "MINU",
    "SECO",
    "CLATH",
    "CLONH",
    "SAZA",
    "BEARAZ",
    "FOVN",
)


SOURCE_TIME = pa.struct(
    [
        pa.field("year", pa.uint16()),
        pa.field("month", pa.uint8()),
        pa.field("day", pa.uint8()),
        pa.field("hour", pa.uint8()),
        pa.field("minute", pa.uint8()),
        pa.field("second", pa.float64()),
    ]
)

FOOTPRINT_HEADER = pa.struct(
    [
        pa.field("instrument_id", pa.uint16()),
        pa.field("source_time", SOURCE_TIME),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("satellite_zenith_angle", pa.float32()),
        pa.field("satellite_azimuth_angle", pa.float32()),
        pa.field("field_of_view", pa.uint16()),
    ]
)


def _common_fields(sensor: str) -> tuple[FieldSpec, ...]:
    latitude = "CLAT" if sensor in ATOVS_SENSORS else "CLATH"
    longitude = "CLON" if sensor in ATOVS_SENSORS else "CLONH"
    return (
        _base_field(
            sensor,
            "da_window",
            (),
            "routing",
            "Three-hour UTC routing window; row groups never cross it.",
            representation="derived",
        ),
        _base_field(
            sensor,
            "time_utc",
            ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO"),
            "footprint_identity",
            "Canonical UTC observation time.",
            occurrence_scope="primary instrument occurrence",
        ),
        FieldSpec(
            "source_time",
            SOURCE_TIME,
            ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO"),
            "source_fidelity",
            "Decoded source components retained independently of time_utc.",
            occurrence_scope="primary instrument occurrence",
        ),
        _base_field(
            sensor,
            "platform_id",
            ("SAID",),
            "footprint_identity",
            "NCEP BUFR satellite identifier.",
        ),
        FieldSpec(
            "instrument_id",
            pa.uint16(),
            ("SIID",),
            "footprint_identity",
            "Primary instrument identifier.",
            occurrence_scope="primary instrument occurrence",
        ),
        _base_field(
            sensor,
            "latitude",
            (latitude,),
            "geometry",
            "Decoded footprint latitude in degrees north.",
        ),
        _base_field(
            sensor,
            "longitude",
            (longitude,),
            "geometry",
            "Decoded footprint longitude in degrees east.",
        ),
        _base_field(
            sensor,
            "satellite_zenith_angle",
            ("SAZA",),
            "geometry",
            "Satellite zenith angle in degrees.",
        ),
        FieldSpec(
            "satellite_azimuth_angle",
            pa.float32(),
            ("BEARAZ",),
            "geometry",
            "Satellite bearing/azimuth in degrees true.",
            source_optional=sensor in ATOVS_SENSORS,
        ),
        _base_field(
            sensor,
            "solar_zenith_angle",
            ("SOZA",),
            "geometry",
            "Solar zenith angle in degrees.",
        ),
        FieldSpec(
            "solar_azimuth_angle",
            pa.float32(),
            ("SOLAZI",),
            "geometry",
            "Solar azimuth angle in degrees true.",
            source_optional=sensor in ATOVS_SENSORS,
        ),
        _base_field(
            sensor,
            "field_of_view",
            ("FOVN",),
            "scan_identity",
            "Source field-of-view number.",
        ),
        _base_field(
            sensor,
            "hpx2048_nest",
            (),
            "routing",
            # Stale output text: the field is nullable (null for invalid geolocation).
            # Fix with the next contract version: schema_id covers this text.
            "Required Earth2Grid HEALPix NESTED pixel at nside=2048.",
            representation="derived",
        ),
        FieldSpec(
            "source_file_ordinal",
            pa.uint16(),
            (),
            "provenance",
            "Index into the footer source_files array.",
            nullable=False,
            representation="derived",
        ),
        FieldSpec(
            "source_message_index",
            pa.uint32(),
            (),
            "provenance",
            "Zero-based index among all complete BUFR messages in source byte order.",
            nullable=False,
            representation="decoder provenance",
        ),
        FieldSpec(
            "source_subset_index",
            pa.uint16(),
            (),
            "provenance",
            "Zero-based subset index in the source message.",
            nullable=False,
            representation="decoder provenance",
        ),
        FieldSpec(
            "source_message_time",
            pa.timestamp("ms", tz="UTC"),
            (),
            "provenance",
            "BUFR message reference time from Section 1.",
            nullable=False,
            representation="decoder provenance",
        ),
        FieldSpec(
            "source_data_category",
            pa.uint8(),
            (),
            "provenance",
            "BUFR data category supplied by the decoder.",
            nullable=False,
            representation="decoder provenance",
        ),
    )


COMMON_FIELD_NAMES = tuple(field.name for field in _common_fields("amsua"))


def _atovs_fields(sensor: str, channel_count: int) -> tuple[FieldSpec, ...]:
    return (
        FieldSpec(
            "land_sea_qualifier",
            pa.uint8(),
            ("LSQL",),
            "surface_context",
            "Land/sea qualifier code.",
        ),
        FieldSpec(
            "surface_height",
            pa.float32(),
            ("HOLS",),
            "surface_context",
            "Height of land surface in metres.",
        ),
        FieldSpec(
            "orbit_height",
            pa.float32(),
            ("HMSL",),
            "platform_state",
            "Platform height above mean sea level in metres.",
        ),
        FieldSpec(
            "cold_space_temperature_correction",
            pa.list_(pa.float32(), channel_count),
            ("CSTC",),
            "channel_auxiliary",
            "Channel-aligned cold-space temperature corrections in kelvin.",
            occurrence_scope=f"fixed replicated {channel_count}-channel block",
            axes=(Axis("channel", channel_count, "Primary source-channel order"),),
            source_optional=sensor in ATOVS_SENSORS,
        ),
    )


CRIS_GUARD = pa.list_(
    pa.struct(
        [
            pa.field("channel_number", pa.uint16()),
            pa.field("spectral_radiance", pa.float32()),
        ]
    ),
    12,
)

CRIS_BAND_CALIBRATION_QUALITY = pa.list_(
    pa.struct(
        [
            pa.field("band_type", pa.uint8()),
            pa.field("start_wavenumber_m_inverse", pa.float64()),
            pa.field("end_wavenumber_m_inverse", pa.float64()),
            pa.field("start_channel", pa.uint16()),
            pa.field("end_channel", pa.uint16()),
            pa.field("calibration_quality", pa.uint16()),
            pa.field("fov_quality", pa.uint32()),
        ]
    ),
    3,
)

CRIS_IMAGER_STATISTIC = pa.struct(
    [
        pa.field("statistic_type", pa.uint8()),
        pa.field("radiance", pa.float32()),
    ]
)
CRIS_IMAGER_CHANNEL = pa.struct(
    [
        pa.field("channel_number", pa.uint16()),
        pa.field("statistics", pa.list_(CRIS_IMAGER_STATISTIC, 2)),
    ]
)
CRIS_IMAGER_SCENES = pa.list_(
    pa.struct(
        [
            pa.field("y_angle", pa.float32()),
            pa.field("z_angle", pa.float32()),
            pa.field("clear_fraction", pa.uint8()),
            pa.field("channels", pa.list_(CRIS_IMAGER_CHANNEL, 16)),
        ]
    ),
    7,
)

IASI_SCALE_GROUPS = pa.list_(
    pa.struct(
        [
            pa.field("start_channel", pa.uint16()),
            pa.field("end_channel", pa.uint16()),
            pa.field("scale_exponent", pa.uint8()),
        ]
    ),
    10,
)

IASI_AVHRR_CHANNEL = pa.struct(
    [
        pa.field("channel_number", pa.uint16()),
        pa.field("mean_scale_exponent", pa.uint8()),
        pa.field("std_scale_exponent", pa.uint8()),
        pa.field("mean_radiance", pa.float32()),
        pa.field("std_radiance", pa.float32()),
    ]
)
IASI_AVHRR_SCENES = pa.list_(
    pa.struct(
        [
            pa.field("y_angle", pa.float32()),
            pa.field("z_angle", pa.float32()),
            pa.field("clear_fraction", pa.uint8()),
            pa.field("channels", pa.list_(IASI_AVHRR_CHANNEL, 6)),
        ]
    ),
    7,
)

AIRS_INSTRUMENT_TEMPERATURES = pa.list_(
    pa.struct(
        [
            pa.field("radiometer_id", pa.uint16()),
            pa.field("temperature_k", pa.float32()),
        ]
    ),
    9,
)

AIRS_CHANNEL_AUXILIARY = pa.list_(
    pa.struct(
        [
            pa.field("channel_number", pa.uint16()),
            pa.field("log10_wavenumber", pa.float64()),
            pa.field("acquisition_quality", pa.uint32()),
        ]
    ),
    281,
)

AIRS_VISIBLE = pa.list_(
    pa.struct(
        [
            pa.field("channel_number", pa.uint16()),
            pa.field("log10_wavenumber", pa.float64()),
            pa.field("acquisition_quality", pa.uint32()),
            pa.field("statistic_type_1", pa.uint8()),
            pa.field("albedo_1", pa.float32()),
            pa.field("statistic_type_2", pa.uint8()),
            pa.field("albedo_2", pa.float32()),
            pa.field("statistic_type_3", pa.uint8()),
        ]
    ),
    4,
)

AIRS_COMPANION_CHANNEL = pa.struct(
    [
        pa.field("channel_number", pa.uint16()),
        pa.field("log10_wavenumber", pa.float64()),
        pa.field("acquisition_quality", pa.uint32()),
        pa.field("brightness_temperature_k", pa.float32()),
    ]
)


def _structural_exclusions(*replication_mnemonics: str) -> tuple[Exclusion, ...]:
    exclusions = [
        Exclusion(
            source=mnemonic,
            reason="BUFR delayed-replication factor, not an independent observation",
            disposition="captured by fixed-size Arrow shape and validated per batch",
        )
        for mnemonic in replication_mnemonics
    ]
    exclusions.extend(
        [
            Exclusion(
                source="BUFR message framing and compressed differences",
                reason="Parquet is a semantic archive, not a byte-identical BUFR container",
                # Stale output text: describes a migration step, not the archive.
                # Fix with the next contract version: schema_id covers this text.
                disposition="retain original BUFR checksum/object until replacement gates pass",
            ),
            Exclusion(
                source="embedded Table B/D text and operator tokens",
                reason="these are interpretation metadata rather than footprint values",
                disposition="embed the active source structure and its content hash in file metadata",
            ),
        ]
    )
    return tuple(exclusions)


def _sensor_specs() -> dict[str, SensorArchiveSpec]:
    result = {
        sensor: SensorArchiveSpec(
            sensor=sensor,
            display_name=display,
            active_template=template,
            channel_count=count,
            complete_template_evidence=evidence,
            fields=_common_fields(sensor) + _atovs_fields(sensor, count),
            primary_leaves=(
                PrimaryLeaf(
                    "brightness_temperature",
                    "TMBR",
                    _reader_value_type(sensor),
                    "K",
                    "Source brightness temperature.",
                ),
            ),
            exclusions=_structural_exclusions()
            + (
                (
                    Exclusion(
                        source="BYTCNT",
                        reason="NCEP local message byte-count framing",
                        disposition="validated and retained in source structure only",
                    ),
                    Exclusion(
                        source="DRF8BIT.LOCAL_0-63-255",
                        reason="NCEP local zero-padding block",
                        disposition="validated as zero and omitted from row columns",
                    ),
                )
                if sensor in {"amsua", "amsub"}
                else ()
            ),
            manifest_mnemonics=("CHNM",),
            ignored_source_fields=(
                ("BYTCNT", "DRF8BIT") if sensor in {"amsua", "amsub"} else ()
            ),
        )
        for sensor, display, template, count, evidence in (
            (
                "amsua",
                "AMSU-A",
                "3-61-223",
                15,
                "complete four-cycle source-population and logical Parquet round-trip gate measured",
            ),
            (
                "amsub",
                "AMSU-B",
                "3-61-224",
                5,
                "complete four-cycle source-population and logical Parquet round-trip gate measured",
            ),
            (
                "mhs",
                "MHS",
                "3-61-234",
                5,
                "complete four-cycle source-population and logical Parquet round-trip gate measured",
            ),
        )
    }
    result["atms"] = SensorArchiveSpec(
        sensor="atms",
        display_name="ATMS",
        active_template="3-10-061",
        channel_count=22,
        complete_template_evidence=(
            "complete four-cycle source-population and logical Parquet round-trip gate measured"
        ),
        fields=_common_fields("atms")
        + (
            FieldSpec(
                "originating_center",
                pa.uint8(),
                ("OGCE",),
                "provenance",
                "Originating center code.",
            ),
            FieldSpec(
                "originating_subcenter",
                pa.uint8(),
                ("GSES",),
                "provenance",
                "Originating subcenter code.",
            ),
            FieldSpec(
                "satellite_classification",
                pa.uint16(),
                ("SCLF",),
                "provenance",
                "Satellite classification code.",
            ),
            FieldSpec(
                "orbit_number", pa.uint32(), ("ORBN",), "scan_identity", "Orbit number."
            ),
            FieldSpec(
                "scan_line",
                pa.uint16(),
                ("SLNM",),
                "scan_identity",
                "Scan-line number.",
            ),
            _base_field(
                "atms",
                "granule_quality",
                ("ATMSGQ",),
                "quality",
                "ATMS granule-level quality flags.",
            ),
            _base_field(
                "atms",
                "scan_quality",
                ("ATMSSQ",),
                "quality",
                "ATMS scan-level quality flags.",
            ),
            FieldSpec(
                "geolocation_quality",
                pa.uint8(),
                ("NGQI",),
                "quality",
                "Geolocation quality indicator.",
            ),
            FieldSpec(
                "orbit_height",
                pa.float32(),
                ("HMSL",),
                "platform_state",
                "Platform height above mean sea level in metres.",
            ),
            FieldSpec(
                "antenna_correction_version",
                pa.uint8(),
                ("SACV",),
                "processing_provenance",
                "Antenna-correction version number.",
            ),
            _base_field(
                "atms",
                "channel_auxiliary",
                ("CHNM", "SCCF", "SCBW", "ANPO", "NEDTCO", "NEDTWA", "ATMSCHQ"),
                "channel_auxiliary",
                "Aligned channel definition, calibration diagnostics, and quality.",
                occurrence_scope="22-entry delayed-replication block",
                axes=(Axis("channel", 22, "Primary ATMS source-channel order"),),
            ),
        ),
        primary_leaves=(
            PrimaryLeaf(
                "antenna_temperature",
                "TMANT",
                _reader_value_type("atms"),
                "K",
                "Source antenna temperature.",
                source_container="DRF16BIT",
            ),
            PrimaryLeaf(
                "brightness_temperature",
                "TMBR",
                pa.float32(),
                "K",
                "Source brightness temperature.",
                source_container="DRF16BIT",
            ),
        ),
        exclusions=_structural_exclusions("DRF16BIT"),
        manifest_mnemonics=("CHNM",),
    )
    result["cris"] = SensorArchiveSpec(
        sensor="cris",
        display_name="CrIS FSR",
        active_template="3-10-198",
        channel_count=431,
        complete_template_evidence=(
            "complete four-cycle source-population and logical Parquet round-trip gate measured"
        ),
        fields=_common_fields("cris")
        + (
            FieldSpec(
                "originating_center",
                pa.uint8(),
                ("OGCE",),
                "provenance",
                "Originating center code.",
            ),
            FieldSpec(
                "secondary_instrument_id",
                pa.uint16(),
                ("SIID",),
                "provenance",
                "Secondary SIID occurrence.",
                occurrence_scope="secondary instrument occurrence",
            ),
            FieldSpec(
                "satellite_classification",
                pa.uint16(),
                ("SCLF",),
                "provenance",
                "Satellite classification code.",
            ),
            FieldSpec(
                "spacecraft_position_m",
                pa.list_(pa.float64(), 3),
                ("PD00", "PD90", "PDNP"),
                "platform_state",
                "Cartesian spacecraft position toward 0E, 90E, and North Pole.",
                axes=(
                    Axis(
                        "cartesian_component",
                        3,
                        "Directions toward 0E, 90E, and the North Pole",
                    ),
                ),
            ),
            FieldSpec(
                "track_qualifier",
                pa.uint8(),
                ("STKO",),
                "scan_identity",
                "Satellite track/orbit qualifier.",
            ),
            FieldSpec(
                "scan_line",
                pa.uint16(),
                ("SLNM",),
                "scan_identity",
                "Scan-line number.",
            ),
            _base_field(
                "cris",
                "field_of_regard",
                ("FORN",),
                "scan_identity",
                "Field-of-regard number.",
            ),
            FieldSpec(
                "orbit_number", pa.uint32(), ("ORBN",), "scan_identity", "Orbit number."
            ),
            FieldSpec(
                "surface_height",
                pa.float32(),
                ("HOLS",),
                "surface_context",
                "Height of land surface in metres.",
            ),
            FieldSpec(
                "orbit_height",
                pa.float32(),
                ("HMSL",),
                "platform_state",
                "Platform height above mean sea level in metres.",
            ),
            FieldSpec(
                "land_fraction",
                pa.float32(),
                ("ALFR",),
                "surface_context",
                "Land fraction.",
            ),
            FieldSpec(
                "land_sea_qualifier",
                pa.uint8(),
                ("LSQL",),
                "surface_context",
                "Land/sea qualifier.",
            ),
            FieldSpec(
                "total_cloud_cover",
                pa.uint8(),
                ("TOCC",),
                "scene_context",
                "Total cloud cover percent.",
            ),
            FieldSpec(
                "cloud_top_height",
                pa.float32(),
                ("HOCT",),
                "scene_context",
                "Cloud-top height.",
            ),
            FieldSpec(
                "radiance_type_flags",
                pa.uint16(),
                ("RDTF",),
                "quality",
                "Radiance-type flags.",
            ),
            _base_field(
                "cris",
                "scan_quality",
                ("NSQF",),
                "quality",
                "CrIS scan-level quality flags.",
            ),
            FieldSpec(
                "geolocation_quality",
                pa.uint8(),
                ("NGQI",),
                "quality",
                "Geolocation quality indicator.",
            ),
            FieldSpec(
                "quality_mark", pa.uint8(), ("QMRKH",), "quality", "BUFR quality mark."
            ),
            FieldSpec(
                "message_type",
                pa.string(),
                ("MTYP",),
                "provenance",
                "Source message/measurement type.",
            ),
            FieldSpec(
                "guard_spectrum",
                CRIS_GUARD,
                ("CHNM", "SRAD"),
                "guard_spectrum",
                "Independent 12-channel guard spectrum, decoded to radiance in "
                "W m^-2 sr^-1 (cm^-1)^-1 rather than kept as the main spectrum's codes.",
                occurrence_scope="12-entry DRF8BIT block",
                axes=(Axis("guard_channel", 12, "Guard-spectrum source order"),),
            ),
            FieldSpec(
                "band_calibration_quality",
                CRIS_BAND_CALIBRATION_QUALITY,
                ("TOBD", "WVNM", "STCH", "ENCH", "NCQF", "NFQF"),
                "band_quality",
                "Three BCFQFSQ records retaining paired band bounds and quality.",
                occurrence_scope="three fixed BCFQFSQ occurrences before the main spectrum",
                axes=(Axis("band", 3, "Source BCFQFSQ band order"),),
            ),
            FieldSpec(
                "primary_spectrum_band_type",
                pa.uint8(),
                ("TOBD",),
                "band_definition",
                "TOBD occurrence immediately before the main CrIS spectrum.",
                occurrence_scope="TOBD occurrence 3 in expanded template order",
            ),
            FieldSpec(
                "guard_spectrum_band_type",
                pa.uint8(),
                ("TOBD",),
                "band_definition",
                "TOBD occurrence immediately before the guard spectrum.",
                occurrence_scope="TOBD occurrence 4 in expanded template order",
            ),
            FieldSpec(
                "collocated_imager_band_type",
                pa.uint8(),
                ("TOBD",),
                "band_definition",
                "TOBD occurrence immediately before the collocated imager block.",
                occurrence_scope="TOBD occurrence 5 in expanded template order",
            ),
            FieldSpec(
                "collocated_imager_scenes",
                CRIS_IMAGER_SCENES,
                ("YAPCG", "ZAPCG", "FCPH", "CHNM", "FOST", "SCHRAD"),
                "collocated_imager",
                "Seven scenes, each with 16 channels and two statistics per channel. "
                "Radiance is per micron, W m^-2 sr^-1 um^-1, unlike the main spectrum's "
                "per cm^-1; clear_fraction is a percentage.",
                occurrence_scope="seven scenes x 16 channels x two statistics",
                axes=(
                    Axis("scene", 7, "Collocated imager scene-cluster order"),
                    Axis("channel", 16, "Imager channel order within each scene"),
                    Axis("statistic", 2, "Statistic order within each channel"),
                ),
            ),
        ),
        primary_leaves=(
            PrimaryLeaf(
                "spectral_radiance_code",
                "SRAD",
                _reader_value_type("cris"),
                # The stored value is the source code itself; value_encoding carries the
                # formula that turns it into radiance and the units that result.
                "dimensionless",
                "Exact CrIS source integer code; value_encoding gives the physical value.",
                representation="source integer code; DELTA_BINARY_PACKED + ZSTD-3",
                source_container="DRF16BIT",
            ),
        ),
        exclusions=_structural_exclusions("DRF16BIT", "DRF8BIT"),
        manifest_mnemonics=("CHNM",),
        primary_value_encoding=PrimaryValueEncoding(
            id="cris_srad_014044_s7_r-100000_w22_v1",
            storage_dtype="int32",
            descriptor="0-14-044",
            scale=7,
            reference=-100000,
            bit_width=22,
            source_missing_code=4194303,
            null_representation="arrow_null",
            physical_dtype="float32",
            source_units="W M**-2 SR**-1 CM",
            physical_units="W m^-2 sr^-1 (cm^-1)^-1",
            formula="(code + reference) * 10**(-scale)",
        ),
    )
    result["iasi"] = SensorArchiveSpec(
        sensor="iasi",
        display_name="IASI C0616",
        active_template="3-61-207",
        channel_count=616,
        complete_template_evidence=(
            "complete four-cycle source-population and logical Parquet round-trip gate measured"
        ),
        fields=_common_fields("iasi")
        + (
            FieldSpec(
                "originating_center",
                pa.uint16(),
                ("GCLONG",),
                "provenance",
                "Originating/generating center code.",
            ),
            FieldSpec(
                "secondary_instrument_id",
                pa.uint16(),
                ("SIID",),
                "provenance",
                "Secondary SIID occurrence.",
                occurrence_scope="secondary instrument occurrence",
            ),
            FieldSpec(
                "satellite_classification",
                pa.uint16(),
                ("SCLF",),
                "provenance",
                "Satellite classification code.",
            ),
            FieldSpec(
                "orbit_number", pa.uint32(), ("ORBN",), "scan_identity", "Orbit number."
            ),
            FieldSpec(
                "scan_line",
                pa.uint16(),
                ("SLNM",),
                "scan_identity",
                "Scan-line number.",
            ),
            FieldSpec(
                "major_frame_count",
                pa.uint8(),
                ("MJFC",),
                "scan_identity",
                "Major-frame count.",
            ),
            FieldSpec(
                "station_elevation",
                pa.float32(),
                ("SELV",),
                # Stale output text: SELV here is platform altitude, so the role
                # belongs with orbit_height, not surface_context. Fix with the next contract version: schema_id covers this text.
                "surface_context",
                "Station/platform elevation in metres, from this sequence.",
            ),
            _base_field(
                "iasi",
                "summary_quality",
                ("QGFQ",),
                "quality",
                "IASI system summary quality flag.",
            ),
            FieldSpec(
                "instrument_noise_quality",
                pa.uint8(),
                ("QGQI",),
                "quality",
                "Instrument-noise quality index, percent.",
            ),
            FieldSpec(
                "geometric_quality",
                pa.uint8(),
                ("QGQIL",),
                "quality",
                "Geometric quality index, percent.",
            ),
            FieldSpec(
                "radiometric_calibration_quality",
                pa.uint8(),
                ("QGQIR",),
                "quality",
                "Radiometric-calibration quality index, percent.",
            ),
            FieldSpec(
                "spectral_calibration_quality",
                pa.uint8(),
                ("QGQIS",),
                "quality",
                "Spectral-calibration quality index, percent.",
            ),
            FieldSpec(
                "system_quality_function",
                pa.uint32(),
                ("QGSSQ",),
                "quality",
                "TEC system-quality function output.",
            ),
            FieldSpec(
                "main_scale_groups",
                IASI_SCALE_GROUPS,
                ("STCH", "ENCH", "CHSF"),
                "spectral_scaling",
                "Ten scale groups used to interpret SCRA, as SCRA * 10**(5 - CHSF) on the "
                "CRTM scale. A null CHSF group is null radiance only when every aligned "
                "SCRA source value is zero or null; nonzero unknown-scale values fail.",
                axes=(Axis("scale_group", 10, "IASI source scale-group order"),),
                source_occurrence_indices={
                    "STCH": tuple(range(10)),
                    "ENCH": tuple(range(10)),
                    "CHSF": tuple(range(10)),
                },
            ),
            FieldSpec(
                "avhrr_channel_combination_flags",
                pa.uint8(),
                ("AVHCST",),
                "collocated_imager",
                "AVHRR channel-combination flags.",
                data_class="secondary_descriptor",
            ),
            FieldSpec(
                "collocated_avhrr_scenes",
                IASI_AVHRR_SCENES,
                ("YAPCG", "ZAPCG", "FCPH", "CHNM", "CHSF", "SMRA", "SSDR"),
                "collocated_imager",
                "Seven AVHRR scenes, each with six mean/std channel records. The "
                "radiances are source-scaled in W m^-2 sr^-1 m and their paired CHSF "
                "exponents are stored unapplied; clear_fraction is a percentage.",
                occurrence_scope="seven scenes x six AVHRR channels",
                axes=(
                    Axis("scene", 7, "Collocated AVHRR scene-cluster order"),
                    Axis("channel", 6, "AVHRR channel order within each scene"),
                ),
            ),
        ),
        primary_leaves=(
            PrimaryLeaf(
                "spectral_radiance",
                "SCRA",
                _reader_value_type("iasi"),
                "mW m^-2 sr^-1 (cm^-1)^-1",
                "Radiance as SCRA * 10**(5 - CHSF), the CRTM scale.",
                source_container="DRF16BIT",
            ),
        ),
        exclusions=_structural_exclusions("DRF16BIT"),
        manifest_mnemonics=("CHNM",),
    )
    result["airs"] = SensorArchiveSpec(
        sensor="airs",
        display_name="AIRS",
        active_template="3-50-243",
        channel_count=281,
        complete_template_evidence=(
            "complete four-cycle source-population and logical Parquet round-trip gate measured"
        ),
        fields=_common_fields("airs")
        + (
            FieldSpec(
                "orbit_number", pa.uint32(), ("ORBN",), "scan_identity", "Orbit number."
            ),
            FieldSpec(
                "scan_line",
                pa.uint16(),
                ("SLNM",),
                "scan_identity",
                "Scan-line number.",
            ),
            FieldSpec(
                "major_frame_count",
                pa.uint8(),
                ("MJFC",),
                "scan_identity",
                "Major-frame count.",
            ),
            FieldSpec(
                "station_elevation",
                pa.float32(),
                ("SELV",),
                # Stale output text: SELV here is platform altitude, so the role
                # belongs with orbit_height, not surface_context. Fix with the next contract version: schema_id covers this text.
                "surface_context",
                "Station/platform elevation in metres.",
            ),
            FieldSpec(
                "instrument_temperatures",
                AIRS_INSTRUMENT_TEMPERATURES,
                ("RAID", "TMINST"),
                "housekeeping",
                "Nine radiometer-ID/instrument-temperature records.",
                axes=(Axis("radiometer", 9, "Source radiometer order"),),
            ),
            FieldSpec(
                "channel_auxiliary",
                AIRS_CHANNEL_AUXILIARY,
                ("CHNM", "LOGRCW", "ACQF"),
                "channel_auxiliary",
                "AIRS channel identity, central wavenumber encoding, and acquisition "
                "flags. log10_wavenumber is log10 of the wavenumber in m^-1.",
                occurrence_scope="281-entry delayed-replication block",
                axes=(Axis("channel", 281, "Primary AIRS source-channel order"),),
            ),
            FieldSpec(
                "visible_statistics",
                AIRS_VISIBLE,
                ("CHNM", "LOGRCW", "ACQF", "FOST", "ALBD"),
                "visible_payload",
                "Four visible-channel statistic records; albedo is a percentage and "
                "log10_wavenumber is log10 of the wavenumber in m^-1.",
                axes=(Axis("visible_channel", 4, "Visible-channel source order"),),
            ),
            FieldSpec(
                "total_cloud_cover",
                pa.uint8(),
                ("TOCC",),
                "scene_context",
                "Total cloud cover percent.",
            ),
            FieldSpec(
                "companion_amsua_header",
                FOOTPRINT_HEADER,
                AIRS_COMPANION_HEADER_MNEMONICS,
                "companion_instrument",
                "Collocated companion AMSU-A footprint header.",
                occurrence_scope="AMSU-A instrument occurrence",
            ),
            FieldSpec(
                "companion_amsua_channels",
                pa.list_(AIRS_COMPANION_CHANNEL, 15),
                ("CHNM", "LOGRCW", "ACQF", "TMBR"),
                "companion_instrument",
                "Fifteen collocated AMSU-A channel records, brightness temperature in "
                "kelvin and log10_wavenumber as log10 of the wavenumber in m^-1.",
                occurrence_scope="companion AMSU-A 15-channel block",
                axes=(Axis("channel", 15, "Companion AMSU-A source-channel order"),),
            ),
            FieldSpec(
                "companion_hsb_header",
                FOOTPRINT_HEADER,
                AIRS_COMPANION_HEADER_MNEMONICS,
                "companion_instrument",
                "Collocated companion HSB footprint header.",
                occurrence_scope="HSB instrument occurrence",
            ),
            FieldSpec(
                "companion_hsb_channels",
                pa.list_(AIRS_COMPANION_CHANNEL, 5),
                ("CHNM", "LOGRCW", "ACQF", "TMBR"),
                "companion_instrument",
                "Five collocated HSB channel records, brightness temperature in kelvin "
                "and log10_wavenumber as log10 of the wavenumber in m^-1.",
                occurrence_scope="companion HSB five-channel block",
                axes=(Axis("channel", 5, "Companion HSB source-channel order"),),
            ),
        ),
        primary_leaves=(
            PrimaryLeaf(
                "brightness_temperature",
                "TMBR",
                _reader_value_type("airs"),
                "K",
                "AIRS main-channel brightness temperature.",
                source_container="DRF16BIT",
            ),
        ),
        exclusions=_structural_exclusions("DRF16BIT"),
        manifest_mnemonics=("CHNM",),
    )
    occurrence_bindings: dict[
        tuple[str, str], tuple[Mapping[str, tuple[int, ...]], str | None]
    ] = {
        ("atms", "channel_auxiliary"): ({}, "DRF16BIT"),
        ("cris", "instrument_id"): ({"SIID": (0,)}, None),
        ("cris", "secondary_instrument_id"): ({"SIID": (1,)}, None),
        (
            "cris",
            "band_calibration_quality",
        ): (
            {
                "TOBD": (0, 1, 2),
                "WVNM": tuple(range(6)),
                "STCH": tuple(range(3)),
                "ENCH": tuple(range(3)),
                "NCQF": tuple(range(3)),
                "NFQF": tuple(range(3)),
            },
            None,
        ),
        ("cris", "primary_spectrum_band_type"): ({"TOBD": (3,)}, None),
        ("cris", "guard_spectrum"): ({}, "DRF8BIT"),
        ("cris", "guard_spectrum_band_type"): ({"TOBD": (4,)}, None),
        ("cris", "collocated_imager_band_type"): ({"TOBD": (5,)}, None),
        (
            "cris",
            "collocated_imager_scenes",
        ): (
            {
                "YAPCG": tuple(range(7)),
                "ZAPCG": tuple(range(7)),
                "FCPH": tuple(range(7)),
                "CHNM": tuple(range(112)),
                "FOST": tuple(range(224)),
                "SCHRAD": tuple(range(224)),
            },
            None,
        ),
        ("iasi", "instrument_id"): ({"SIID": (0,)}, None),
        ("iasi", "secondary_instrument_id"): ({"SIID": (1,)}, None),
        (
            "iasi",
            "collocated_avhrr_scenes",
        ): (
            {
                "YAPCG": tuple(range(7)),
                "ZAPCG": tuple(range(7)),
                "FCPH": tuple(range(7)),
                "CHNM": tuple(range(42)),
                "CHSF": tuple(range(10, 94)),
                "SMRA": tuple(range(42)),
                "SSDR": tuple(range(42)),
            },
            None,
        ),
        ("airs", "instrument_id"): ({"SIID": (0,)}, None),
        ("airs", "satellite_zenith_angle"): ({"SAZA": (0,)}, None),
        ("airs", "satellite_azimuth_angle"): ({"BEARAZ": (0,)}, None),
        ("airs", "field_of_view"): ({"FOVN": (0,)}, None),
        ("airs", "instrument_temperatures"): (
            {"RAID": tuple(range(9)), "TMINST": tuple(range(9))},
            None,
        ),
        ("airs", "channel_auxiliary"): ({}, "DRF16BIT"),
        (
            "airs",
            "visible_statistics",
        ): (
            {
                "CHNM": tuple(range(4)),
                "LOGRCW": tuple(range(4)),
                "ACQF": tuple(range(4)),
                "FOST": tuple(range(12)),
                "ALBD": tuple(range(8)),
            },
            None,
        ),
        (
            "airs",
            "companion_amsua_header",
        ): (
            {mnemonic: (1,) for mnemonic in AIRS_COMPANION_HEADER_MNEMONICS},
            None,
        ),
        (
            "airs",
            "companion_amsua_channels",
        ): (
            {
                "CHNM": tuple(range(4, 19)),
                "LOGRCW": tuple(range(4, 19)),
                "ACQF": tuple(range(4, 19)),
                "TMBR": tuple(range(15)),
            },
            None,
        ),
        (
            "airs",
            "companion_hsb_header",
        ): (
            {mnemonic: (2,) for mnemonic in AIRS_COMPANION_HEADER_MNEMONICS},
            None,
        ),
        (
            "airs",
            "companion_hsb_channels",
        ): (
            {
                "CHNM": tuple(range(19, 24)),
                "LOGRCW": tuple(range(19, 24)),
                "ACQF": tuple(range(19, 24)),
                "TMBR": tuple(range(15, 20)),
            },
            None,
        ),
    }
    for sensor, spec in tuple(result.items()):
        fields = []
        for field_spec in spec.fields:
            binding = occurrence_bindings.get((sensor, field_spec.name))
            fields.append(
                field_spec
                if binding is None
                else replace(
                    field_spec,
                    source_occurrence_indices=binding[0],
                    source_container=binding[1],
                )
            )
        result[sensor] = replace(spec, fields=tuple(fields))
    return result


SENSOR_SPECS = _sensor_specs()


def archive_field_spec(sensor: str, logical_name: str) -> FieldSpec:
    try:
        spec = SENSOR_SPECS[sensor]
    except KeyError as exc:
        raise ValueError(f"unsupported sensor {sensor!r}") from exc
    matches = [field for field in spec.fields if field.name == logical_name]
    if len(matches) != 1:
        raise ValueError(
            f"{sensor} archive contract has {len(matches)} fields named "
            f"{logical_name!r}"
        )
    return matches[0]


def primary_field_name(leaf: PrimaryLeaf, source_channel_id: int) -> str:
    return f"{leaf.prefix}__ch_{source_channel_id:05d}"


@dataclass(frozen=True)
class SourceStructure:
    """Exact active BUFR structure needed to interpret one source release."""

    sensor: str
    embedded_table_sha256: str
    observation_descriptors: tuple[str, ...]
    active_sequences: tuple[Mapping[str, Any], ...]
    active_elements: tuple[Mapping[str, Any], ...]
    effective_occurrences: tuple[Mapping[str, Any], ...]
    decoded_physical_schema: tuple[Mapping[str, Any], ...]
    format_version: int = 1

    @classmethod
    def from_decoder(cls, sensor: str, payload: Mapping[str, Any]) -> SourceStructure:
        """Build the structure from one Earth2Bufr source-structure payload."""

        embedded_table_sha256 = payload.get("embedded_table_sha256")
        if embedded_table_sha256 is None:
            raise ValueError(f"{sensor} source structure has no embedded DX table")
        structure = cls(
            sensor=sensor,
            embedded_table_sha256=str(embedded_table_sha256),
            observation_descriptors=tuple(payload["observation_descriptors"]),
            active_sequences=tuple(payload["active_sequences"]),
            active_elements=tuple(payload["active_elements"]),
            effective_occurrences=tuple(payload["effective_occurrences"]),
            decoded_physical_schema=tuple(payload["decoded_physical_schema"]),
            format_version=int(payload["format_version"]),
        )
        structure.validate()
        return structure

    def as_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "sensor": self.sensor,
            "embedded_table_sha256": self.embedded_table_sha256,
            "observation_descriptors": list(self.observation_descriptors),
            "active_sequences": list(self.active_sequences),
            "active_elements": list(self.active_elements),
            "effective_occurrences": list(self.effective_occurrences),
            "decoded_physical_schema": list(self.decoded_physical_schema),
        }

    @cached_property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_payload()).encode()).hexdigest()

    def as_record(self) -> dict[str, Any]:
        return {"sha256": self.fingerprint, "structure": self.as_payload()}

    def validate(self) -> None:
        if self.sensor not in SENSOR_SPECS:
            raise ValueError(f"unknown source-structure sensor: {self.sensor}")
        if len(self.embedded_table_sha256) != 64:
            raise ValueError("invalid embedded-table SHA-256")
        if not self.observation_descriptors:
            raise ValueError("source structure has no observation descriptors")
        if not self.active_elements:
            raise ValueError("source structure has no active element descriptors")
        if not self.effective_occurrences:
            raise ValueError("source structure has no effective occurrence ledger")
        if not self.decoded_physical_schema:
            raise ValueError("source structure has no decoded physical schema")


_CRIS_SOURCE_RADIANCE_UNITS = "W M**-2 SR**-1 CM"


def derive_cris_primary_encoding(
    source_structure: SourceStructure,
) -> PrimaryValueEncoding:
    """Derive one common main-spectrum encoding from the effective ledger."""

    source_structure.validate()
    if source_structure.sensor != "cris":
        raise ValueError(
            f"CrIS conversion received {source_structure.sensor!r} source structure"
        )
    # The main spectrum is one SRAD member inside DRF16BIT, repeated per channel.
    occurrences = [
        occurrence
        for occurrence in source_structure.effective_occurrences
        if occurrence.get("decoder_path") == "DRF16BIT.SRAD"
        and occurrence.get("mnemonic") == "SRAD"
    ]
    if len(occurrences) != 1:
        raise ValueError(
            "CrIS source structure must contain one DRF16BIT.SRAD member, found "
            f"{len(occurrences)}"
        )
    packing = occurrences[0]

    descriptor = packing.get("descriptor")
    source_units = packing.get("units")
    if not isinstance(descriptor, str) or not descriptor:
        raise ValueError("CrIS main SRAD descriptor is missing")
    if source_units != _CRIS_SOURCE_RADIANCE_UNITS:
        raise ValueError(
            f"CrIS main SRAD units {source_units!r} are not "
            f"{_CRIS_SOURCE_RADIANCE_UNITS!r}"
        )
    try:
        scale = int(packing["effective_scale"])
        reference = int(packing["effective_reference"])
        bit_width = int(packing["effective_width"])
        missing_code = int(packing["source_missing_code"])
    except (TypeError, ValueError) as exc:
        raise ValueError("CrIS main SRAD effective packing is incomplete") from exc
    if bit_width <= 0 or bit_width > 31:
        raise ValueError(
            f"CrIS main SRAD width {bit_width} cannot be stored as signed int32"
        )
    expected_missing_code = (1 << bit_width) - 1
    if missing_code != expected_missing_code:
        raise ValueError(
            "CrIS main SRAD missing code is inconsistent with its width: "
            f"expected {expected_missing_code}, got {missing_code}"
        )
    compact_descriptor = descriptor.replace("-", "")
    return PrimaryValueEncoding(
        id=(f"cris_srad_{compact_descriptor}_s{scale}_r{reference}_w{bit_width}_v1"),
        storage_dtype="int32",
        descriptor=descriptor,
        scale=scale,
        reference=reference,
        bit_width=bit_width,
        source_missing_code=missing_code,
        null_representation="arrow_null",
        physical_dtype="float32",
        source_units=source_units,
        physical_units="W m^-2 sr^-1 (cm^-1)^-1",
        formula="(code + reference) * 10**(-scale)",
    )


@dataclass(frozen=True)
class SourceFile:
    """One BUFR input represented in a daily Parquet file."""

    object_key: str
    sha256: str
    size_bytes: int
    cycle: str
    message_type: str
    message_header_manifest: Mapping[str, Any]
    source_structure_sha256: str | None = None
    source_structure_sha256s: tuple[str, ...] = ()

    @property
    def source_structure_hashes(self) -> tuple[str, ...]:
        return source_file_structure_hashes(self)

    def as_record(self) -> dict[str, Any]:
        record = {
            "object_key": self.object_key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "cycle": self.cycle,
            "message_type": self.message_type,
        }
        hashes = self.source_structure_hashes
        if len(hashes) == 1:
            record["source_structure_sha256"] = hashes[0]
        else:
            record["source_structure_sha256s"] = list(hashes)
        record["message_header_manifest"] = self.message_header_manifest
        return record


def source_file_structure_hashes(
    source: SourceFile | Mapping[str, Any],
) -> tuple[str, ...]:
    """Return one source object's normalized structure references."""

    if isinstance(source, SourceFile):
        singular = source.source_structure_sha256
        plural = source.source_structure_sha256s
    else:
        singular = source.get("source_structure_sha256")
        plural_value = source.get("source_structure_sha256s", ())
        if isinstance(plural_value, (str, bytes)) or not isinstance(
            plural_value, Sequence
        ):
            raise ValueError("source_structure_sha256s must be an array")
        plural = tuple(str(value) for value in plural_value)
    if singular is not None and plural:
        raise ValueError(
            "source object cannot use singular and plural structure hashes"
        )
    if singular is not None:
        hashes = (str(singular),)
    else:
        hashes = tuple(plural)
        if len(hashes) < 2:
            raise ValueError("plural source-structure references require two hashes")
    if len(hashes) != len(set(hashes)):
        raise ValueError("source-structure references must be unique")
    if any(len(value) != 64 for value in hashes):
        raise ValueError("invalid source-structure SHA-256")
    return hashes


# Section 1/3 message-header fields stored per source object, in manifest order.
MESSAGE_HEADER_FIELDS = (
    "index",
    "edition",
    "master_table",
    "center",
    "sub_center",
    "update_sequence",
    "optional_section_present",
    "reference_time",
    "data_category",
    "international_sub_category",
    "local_sub_category",
    "master_table_version",
    "local_table_version",
    "subset_count",
    "compressed",
    "section3_flags",
    "descriptors",
)


def decode_message_header_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, list[Any]]:
    """Expand one compact, columnar Section 1/3 message manifest."""

    if manifest.get("format_version") != 1:
        raise ValueError("unsupported message-header manifest format")
    count = int(manifest.get("count", -1))
    if count < 1:
        raise ValueError("message-header manifest count must be positive")
    columns = manifest.get("columns")
    if not isinstance(columns, Mapping):
        raise ValueError("message-header manifest columns are missing")
    decoded: dict[str, list[Any]] = {}
    for name, encoded in columns.items():
        if not isinstance(encoded, Mapping):
            raise ValueError(f"message-header column {name!r} is invalid")
        encoding = encoded.get("encoding")
        if encoding == "constant":
            values = [encoded.get("value")] * count
        elif encoding == "linear":
            start = int(encoded["start"])
            step = int(encoded["step"])
            values = [start + step * index for index in range(count)]
        elif encoding == "rle":
            values = []
            for run_count, value in encoded.get("runs", ()):
                values.extend([value] * int(run_count))
        elif encoding == "plain":
            values = list(encoded.get("values", ()))
        else:
            raise ValueError(
                f"message-header column {name!r} has unknown encoding {encoding!r}"
            )
        if len(values) != count:
            raise ValueError(
                f"message-header column {name!r} has {len(values)} values, "
                f"expected {count}"
            )
        decoded[str(name)] = values
    missing = set(MESSAGE_HEADER_FIELDS) - decoded.keys()
    if missing:
        raise ValueError(f"message-header manifest is missing {sorted(missing)}")
    indices = [int(value) for value in decoded["index"]]
    if indices != sorted(set(indices)):
        raise ValueError("message-header indices must be unique and ordered")
    if any(int(value) < 0 for value in decoded["subset_count"]):
        raise ValueError("message-header subset count must not be negative")
    return decoded


def derive_footprint_id(
    source_sha256: str, source_message_index: int, source_subset_index: int
) -> bytes:
    """Return the virtual 128-bit identity for one source BUFR subset.

    The canonical hash payload is the domain separator, the 32 raw checksum
    bytes, an unsigned 32-bit big-endian complete-message index, and an unsigned
    16-bit big-endian subset index. The identifier is never stored in the
    archive because all inputs are already retained.
    """

    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must contain 64 hexadecimal characters")
    try:
        checksum = bytes.fromhex(source_sha256)
    except ValueError as exc:
        raise ValueError("source_sha256 is not hexadecimal") from exc
    if not 0 <= source_message_index < 1 << 32:
        raise ValueError("source_message_index is outside uint32")
    if not 0 <= source_subset_index < 1 << 16:
        raise ValueError("source_subset_index is outside uint16")
    payload = b"".join(
        (
            FOOTPRINT_ID_DOMAIN,
            checksum,
            source_message_index.to_bytes(4, "big"),
            source_subset_index.to_bytes(2, "big"),
        )
    )
    return hashlib.sha256(payload).digest()[:16]


@dataclass(frozen=True)
class FileProvenance:
    """Required dynamic footer metadata for one daily archive file."""

    utc_day: date
    source_files: tuple[SourceFile, ...]
    source_structures: tuple[SourceStructure, ...]
    decoder_name: str
    decoder_version: str
    decoder_revision: str
    etl_revision: str
    effective_packing_ledger_sha256: str
    code_flag_catalog_sha256: str
    source_subsets_encountered: int
    quarantined_subsets: int = 0
    quarantine_manifest_sha256: str | None = None
    quarantined_source_messages: tuple[Mapping[str, Any], ...] = ()
    missing_cycles: tuple[str, ...] = ()
    empty_windows: tuple[str, ...] = ()
    unusable_source_objects: tuple[Mapping[str, Any], ...] = ()
    decoder_mode: str = "complete_active_template"
    created_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self) -> None:
        if not self.source_files:
            raise ValueError("source_files must not be empty")
        if not self.source_structures:
            raise ValueError("source_structures must not be empty")
        structures: dict[str, SourceStructure] = {}
        for structure in self.source_structures:
            structure.validate()
            if structure.fingerprint in structures:
                raise ValueError(f"duplicate source structure {structure.fingerprint}")
            structures[structure.fingerprint] = structure
        structure_sensors = {structure.sensor for structure in self.source_structures}
        if len(structure_sensors) != 1:
            raise ValueError("one archive file cannot mix sensor structures")
        for source in self.source_files:
            try:
                valid_sha256 = len(bytes.fromhex(source.sha256)) == 32
            except ValueError:
                valid_sha256 = False
            if not valid_sha256:
                raise ValueError(f"invalid SHA-256 for {source.object_key}")
            if source.size_bytes <= 0:
                raise ValueError(f"invalid source size for {source.object_key}")
            if not source.message_type:
                raise ValueError(f"missing message type for {source.object_key}")
            hashes = source.source_structure_hashes
            unknown = set(hashes) - structures.keys()
            if unknown:
                raise ValueError(
                    f"{source.object_key} references unknown source structures "
                    f"{sorted(unknown)}"
                )
            headers = decode_message_header_manifest(source.message_header_manifest)
            templates = {
                tuple(str(value) for value in descriptors)
                for descriptors in headers["descriptors"]
            }
            referenced_templates = {
                structures[value].observation_descriptors for value in hashes
            }
            if len(referenced_templates) != len(hashes):
                raise ValueError(
                    f"{source.object_key} references duplicate descriptor templates"
                )
            if templates != referenced_templates:
                raise ValueError(
                    f"{source.object_key} structure references do not match "
                    "message-header templates"
                )
        referenced = {
            value
            for source in self.source_files
            for value in source.source_structure_hashes
        }
        if referenced != structures.keys():
            raise ValueError("unreferenced source structure in file provenance")
        if len(self.effective_packing_ledger_sha256) != 64:
            raise ValueError("invalid effective-packing-ledger SHA-256")
        if len(self.code_flag_catalog_sha256) != 64:
            raise ValueError("invalid code/flag-catalog SHA-256")
        if self.source_subsets_encountered <= 0:
            raise ValueError("source_subsets_encountered must be positive")
        manifest_subsets = sum(
            int(value)
            for source in self.source_files
            for value in decode_message_header_manifest(source.message_header_manifest)[
                "subset_count"
            ]
        )
        if manifest_subsets != self.source_subsets_encountered:
            raise ValueError(
                "message-header manifests disagree with source subset population"
            )
        if not 0 <= self.quarantined_subsets <= self.source_subsets_encountered:
            raise ValueError("quarantined_subsets is outside the source population")
        cycles = [
            datetime.fromisoformat(source.cycle).astimezone(timezone.utc)
            for source in self.source_files
        ]
        for source, cycle in zip(self.source_files, cycles, strict=True):
            if cycle.date() != self.utc_day:
                raise ValueError(
                    f"{source.object_key} cycle {source.cycle} is not on {self.utc_day}"
                )
        present_hours = {cycle.hour for cycle in cycles}
        expected_missing = tuple(
            f"{hour:02d}Z" for hour in (0, 6, 12, 18) if hour not in present_hours
        )
        if self.missing_cycles != expected_missing:
            raise ValueError(
                f"missing_cycles must be {expected_missing}, got {self.missing_cycles}"
            )
        expected_windows = {
            (cycle + offset).isoformat()
            for cycle in cycles
            for offset in (timedelta(0), timedelta(hours=3))
        }
        if len(self.empty_windows) != len(set(self.empty_windows)) or not set(
            self.empty_windows
        ).issubset(expected_windows):
            raise ValueError("empty_windows are not unique available-cycle windows")
        for source in self.unusable_source_objects:
            if int(source["size_bytes"]) < 0 or not source.get("reason"):
                raise ValueError("invalid unusable source object provenance")
        if self.quarantined_subsets and not self.quarantine_manifest_sha256:
            raise ValueError("quarantined subsets require a quarantine manifest")
        quarantined_subsets = sum(
            int(record["subset_count"]) for record in self.quarantined_source_messages
        )
        if quarantined_subsets != self.quarantined_subsets:
            raise ValueError(
                "quarantined source-message population disagrees with provenance"
            )
        if self.quarantined_source_messages:
            expected_manifest_sha256 = hashlib.sha256(
                _canonical_json(list(self.quarantined_source_messages)).encode()
            ).hexdigest()
            if self.quarantine_manifest_sha256 != expected_manifest_sha256:
                raise ValueError("quarantine-manifest SHA-256 disagrees with records")
        if (
            self.quarantine_manifest_sha256 is not None
            and len(self.quarantine_manifest_sha256) != 64
        ):
            raise ValueError("invalid quarantine-manifest SHA-256")
        if self.created_utc.tzinfo is None:
            raise ValueError("created_utc must be timezone-aware")


def _primary_field(
    sensor: str,
    leaf: PrimaryLeaf,
    channel: Channel,
    primary_value_encoding: PrimaryValueEncoding | None,
) -> pa.Field:
    if primary_value_encoding is None:
        descriptor = _descriptor_record(DESCRIPTORS[sensor][leaf.source_mnemonic])
    else:
        catalog = DESCRIPTORS[sensor][leaf.source_mnemonic]
        descriptor = {
            "descriptor": primary_value_encoding.descriptor,
            "mnemonic": leaf.source_mnemonic,
            "meaning": catalog.meaning,
            "units": primary_value_encoding.source_units,
            "value_kind": catalog.value_kind,
            "interpretation_table_key": catalog.interpretation_table_key,
            "effective_scale": primary_value_encoding.scale,
            "effective_reference": primary_value_encoding.reference,
            "effective_bits": primary_value_encoding.bit_width,
            "source_missing_code": primary_value_encoding.source_missing_code,
            "source_of_truth": "effective occurrence ledger embedded in file footer",
        }
    metadata = _encoded_metadata(
        {
            "nnja.archive.role": "primary_observation",
            "nnja.archive.data_class": "primary_data",
            "nnja.archive.description": leaf.description,
            "nnja.archive.representation": leaf.representation,
            "nnja.archive.units": leaf.units,
            "nnja.archive.source_descriptors": [descriptor],
            "nnja.archive.source_occurrences": "primary instrument channel axis",
            "nnja.archive.source_container": leaf.source_container,
            "nnja.archive.axes": [],
            "nnja.archive.channel": asdict(channel),
            "nnja.archive.value_encoding": (
                None
                if primary_value_encoding is None
                else asdict(primary_value_encoding)
            ),
        }
    )
    name = primary_field_name(leaf, channel.source_channel_id)
    base = reader_schema.channel_field(sensor, channel.source_channel_id, kelvin=False)
    if base.name == name:
        return pa.field(name, base.type, nullable=base.nullable, metadata=metadata)
    return pa.field(name, leaf.data_type, nullable=True, metadata=metadata)


def archive_schema(
    sensor: str,
    channel_manifest: ChannelManifest,
    *,
    source_structure: SourceStructure,
    contract_version: str = CONTRACT_VERSION,
) -> pa.Schema:
    """Build the self-describing logical Arrow schema for one sensor release."""

    try:
        spec = SENSOR_SPECS[sensor]
    except KeyError as exc:
        raise ValueError(f"unsupported sensor {sensor!r}") from exc
    if channel_manifest.sensor != sensor:
        raise ValueError(
            f"manifest is for {channel_manifest.sensor}, schema is for {sensor}"
        )
    channel_manifest.validate(spec.channel_count)
    if source_structure.sensor != sensor:
        raise ValueError(
            f"{source_structure.sensor} source structure cannot build {sensor} schema"
        )
    source_structure.validate()
    resolved_primary_encoding = (
        derive_cris_primary_encoding(source_structure) if sensor == "cris" else None
    )
    primary_value_encoding = (
        asdict(resolved_primary_encoding)
        if resolved_primary_encoding is not None
        else None
    )

    fields = [field_spec.arrow_field(sensor) for field_spec in spec.fields]
    if contract_version == LEGACY_CONTRACT_VERSION:
        index = next(
            index for index, field in enumerate(fields) if field.name == "hpx2048_nest"
        )
        field = fields[index]
        fields[index] = pa.field(
            field.name,
            field.type,
            nullable=False,
            metadata=field.metadata,
        )
    elif contract_version != CONTRACT_VERSION:
        raise ValueError(f"unsupported archive contract version {contract_version!r}")
    for leaf in spec.primary_leaves:
        fields.extend(
            _primary_field(sensor, leaf, channel, resolved_primary_encoding)
            for channel in channel_manifest.channels
        )

    static_contract = {
        "contract_name": CONTRACT_NAME,
        "contract_version": contract_version,
        "sensor": sensor,
        "display_name": spec.display_name,
        "active_template": spec.active_template,
        "row_semantics": "one row per source footprint/subset",
        "primary_layout": "wide primitive columns",
        "auxiliary_layout": "semantic fixed-size lists and nested structs",
        "null_semantics": "Arrow null means source missing; no numeric sentinel",
        # Stale output text: each sensor's evidence string records how the contract
        # was validated, not what the file holds. Fix with the next contract version: schema_id covers this text.
        "complete_template_evidence": spec.complete_template_evidence,
        "channel_manifest": channel_manifest.as_payload(),
        "channel_manifest_sha256": channel_manifest.fingerprint,
        "channel_manifest_source_mnemonics": list(spec.manifest_mnemonics),
        "exclusions": [asdict(exclusion) for exclusion in spec.exclusions],
        "hpx2048": {
            "nside": 2048,
            "level": 11,
            "ordering": "NESTED",
            # Stale output text: nullable since rc4; see null_when_source_geolocation_invalid.
            # Fix with the next contract version: schema_id covers this text.
            "required": True,
            "null_when_source_geolocation_invalid": (
                contract_version != LEGACY_CONTRACT_VERSION
            ),
            "parent_rule": "pix >> (2 * level_difference)",
            "source": "preserved latitude/longitude",
            "implementation": (
                "earth2grid.healpix.Grid(level=11, pixel_order=NEST).ang2pix"
            ),
        },
        "virtual_footprint_id": {
            "stored": False,
            "algorithm": "first 16 bytes of SHA-256",
            "domain_hex": FOOTPRINT_ID_DOMAIN.hex(),
            "payload": [
                "32 raw bytes of source object SHA-256",
                "source_message_index as uint32 big-endian",
                "source_subset_index as uint16 big-endian",
            ],
            "display": "32 lowercase hexadecimal characters",
        },
        "projection_policy": (
            "data_class is source semantics; training/default projections are consumer-owned"
        ),
        "decoder_coverage_policy": (
            "every decoded top-level occurrence and delayed-replication child must "
            "be claimed by this schema before conversion"
        ),
    }
    schema_id_payload = {
        "contract_version": contract_version,
        "sensor": sensor,
        "active_template": spec.active_template,
        "channel_manifest_sha256": channel_manifest.fingerprint,
        "primary_value_encoding": primary_value_encoding,
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
                "metadata": {
                    key.decode(): value.decode()
                    for key, value in sorted((field.metadata or {}).items())
                },
            }
            for field in fields
        ],
    }
    schema_id = hashlib.sha256(_canonical_json(schema_id_payload).encode()).hexdigest()
    static_contract["schema_id"] = schema_id
    metadata = {
        "nnja.archive.contract": static_contract,
        "nnja.archive.schema_id": schema_id,
    }
    if primary_value_encoding is not None:
        metadata["nnja.archive.primary_value_encoding"] = primary_value_encoding
    return pa.schema(
        fields,
        metadata=_encoded_metadata(metadata),
    )


def bind_file_metadata(schema: pa.Schema, provenance: FileProvenance) -> pa.Schema:
    """Attach per-file provenance without changing the logical schema ID."""

    provenance.validate()
    schema_metadata = schema.metadata or {}
    contract = json.loads(schema_metadata[b"nnja.archive.contract"])
    structure_sensor = provenance.source_structures[0].sensor
    if structure_sensor != contract["sensor"]:
        raise ValueError(
            f"{structure_sensor} source structure cannot bind to "
            f"{contract['sensor']} schema"
        )
    if contract["sensor"] == "cris":
        schema_encoding = json.loads(
            schema_metadata[b"nnja.archive.primary_value_encoding"]
        )
        for structure in provenance.source_structures:
            structure_encoding = asdict(derive_cris_primary_encoding(structure))
            if structure_encoding != schema_encoding:
                raise ValueError(
                    "CrIS schema encoding does not match source structure "
                    f"{structure.fingerprint}"
                )
    dynamic = {
        "utc_day": provenance.utc_day.isoformat(),
        "source_files": [source.as_record() for source in provenance.source_files],
        "decoder": {
            "name": provenance.decoder_name,
            "version": provenance.decoder_version,
            "revision": provenance.decoder_revision,
            "mode": provenance.decoder_mode,
        },
        "etl_revision": provenance.etl_revision,
        "effective_packing_ledger_sha256": (provenance.effective_packing_ledger_sha256),
        # Misnamed output key: it hashes descriptor_catalog.csv. Fix with the next contract version: schema_id covers this text.
        "code_flag_catalog_sha256": provenance.code_flag_catalog_sha256,
        "source_subsets_encountered": provenance.source_subsets_encountered,
        "quarantined_subsets": provenance.quarantined_subsets,
        "quarantine_manifest_sha256": provenance.quarantine_manifest_sha256,
        "quarantined_source_messages": list(provenance.quarantined_source_messages),
        "missing_cycles": list(provenance.missing_cycles),
        "empty_windows": list(provenance.empty_windows),
        "unusable_source_objects": list(provenance.unusable_source_objects),
        "created_utc": provenance.created_utc.astimezone(timezone.utc).isoformat(),
    }
    metadata = dict(schema.metadata or {})
    metadata.update(
        _encoded_metadata(
            {
                "nnja.archive.file_provenance": dynamic,
                "nnja.archive.source_structures": [
                    structure.as_record() for structure in provenance.source_structures
                ],
                "nnja.archive.source_structure_hashes": sorted(
                    structure.fingerprint for structure in provenance.source_structures
                ),
            }
        )
    )
    return schema.with_metadata(metadata)


def descriptor_coverage(sensor: str) -> dict[str, list[str]]:
    """Classify every active unique descriptor in the archive contract."""

    spec = SENSOR_SPECS[sensor]
    active = set(DESCRIPTORS[sensor])
    stored = {
        mnemonic
        for field_spec in spec.fields
        for mnemonic in field_spec.source_mnemonics
    }
    stored.update(leaf.source_mnemonic for leaf in spec.primary_leaves)
    manifested = set(spec.manifest_mnemonics)
    structural = spec.structurally_represented_mnemonics
    return {
        "stored": sorted(active & stored),
        "manifest": sorted(active & manifested),
        "stored_or_manifest": sorted(active & (stored | manifested)),
        "structural": sorted(active & structural),
        "unaccounted": sorted(active - stored - manifested - structural),
    }
