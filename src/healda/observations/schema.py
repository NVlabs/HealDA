#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Arrow schemas of the observation tables healda reads and emits.

The documented API is `table_schema` and `validate`. They are the contracts of two ways in:

- the NNJA archive tables, per family (`table_schema("prepbufr")`, ...) and per satellite
  sensor (`table_schema(sensor, ir_channels)`), which the NNJA loaders and
  `adapters.e2s_nnja.analysis_tables` read;
- the unified table, `table_schema("unified")`, one row per observation, which
  `DAModel.run_analysis_unified` takes.

`get_combined_observation_schema` holds the UFS archive's observation columns; the
unified table adds `local_channel_id` and `sensor_id`. Conventional observations with
several components (u, v) are one row per component.
"""

from collections.abc import Mapping, Sequence

import pyarrow as pa

from healda.observations.preprocessing import ir_spectral


GLOBAL_CHANNEL_ID = pa.field("Global_Channel_ID", pa.uint16(), nullable=False)
SENSOR_ID = pa.field("sensor_id", pa.uint16())
# A sensor's own 0-based channel index; GLOBAL_CHANNEL_ID is its offset plus this.
LOCAL_CHANNEL_ID = pa.field("local_channel_id", pa.uint16())


def get_combined_observation_schema() -> pa.Schema:
    """
    Create a combined PyArrow schema for both satellite and conventional observations.

    This schema accommodates:
    1. Common fields present in both data types
    2. Satellite-specific fields (angles, channel info)
    3. Conventional-specific fields (pressure, height, observation types)
    4. Flattened observation structure (all obs types in single Observation column)
    5. Optional analysis fields (present only in 'anl' data)

    Note: Conventional observations (u, v, T, q) are flattened into a single
    'Observation' column with multiple rows per location for multi-component obs.

    Returns:
        pa.Schema: Combined schema for all observation data
    """

    # Common fields present in both satellite and conventional data
    common_fields = [
        # Spatial and temporal information
        pa.field("Latitude", pa.float32()),
        pa.field("Longitude", pa.float32()),
        pa.field(
            "Absolute_Obs_Time", pa.timestamp("ns")
        ),  # nanosecond is excessively precise, but is valid from 1678 --2262, so good enough for our pruposes
        pa.field("DA_window", pa.timestamp("ns")),
        # Platform identification
        pa.field("Platform_ID", pa.uint16()),  # Maps to PLATFORM_NAME_TO_ID
        # Observation data - flattened structure
        pa.field("Observation", pa.float32()),  # Main observation value (required)
        GLOBAL_CHANNEL_ID,
        # pa.field("Obs_ID", pa.uint32(), nullable=True),  # footprint index for fast flat-to-wide pivoting
    ]

    # Satellite-specific fields (from etl.py)
    satellite_fields = [
        # Angular information (satellite observations only)
        pa.field("Sat_Zenith_Angle", pa.float32(), nullable=True),
        pa.field("Sol_Zenith_Angle", pa.float32(), nullable=True),
        pa.field("Scan_Angle", pa.float32(), nullable=True),
    ]

    # Conventional observation specific fields (from etl_conv.py)
    conventional_fields = [
        # Metadata fields
        pa.field("Pressure", pa.float32(), nullable=True),
        pa.field("Height", pa.float32(), nullable=True),
        pa.field("Observation_Type", pa.uint16(), nullable=True),
    ]

    # Analysis fields (present only in 'anl' data)
    analysis_fields = [
        # Quality control and forecast differences
        pa.field("QC_Flag", pa.int32(), nullable=True),  # Satellite QC flag
        pa.field(
            "Analysis_Use_Flag", pa.int8(), nullable=True
        ),  # Conventional analysis flag
        # Forecast differences (flattened to single column)
        # pa.field("Obs_Minus_Forecast_adjusted", pa.float32(), nullable=True),
        # pa.field("Obs_Minus_Forecast_unadjusted", pa.float32(), nullable=True),
    ]

    # Combine all fields
    all_fields = (
        common_fields + satellite_fields + conventional_fields + analysis_fields
    )

    return pa.schema(all_fields)


def get_channel_table_schema():
    return pa.schema(
        [
            GLOBAL_CHANNEL_ID,
            pa.field("min_valid", pa.float32()),
            pa.field("max_valid", pa.float32()),
            SENSOR_ID,
            pa.field("is_conv", pa.bool_()),
            pa.field("name", pa.string()),
            pa.field("mean", pa.float32()),
            pa.field("stddev", pa.float32()),
        ]
    )


def long_table(arrays: dict[str, pa.Array], schema: pa.Schema) -> pa.Table:
    """``arrays`` as a table of ``schema``, every field without an array all null."""
    rows = len(next(iter(arrays.values())))
    return pa.table(
        [
            arrays[field.name] if field.name in arrays else pa.nulls(rows, field.type)
            for field in schema
        ],
        schema=schema,
    )


# --- NNJA archive tables: what the NNJA loaders and `analysis_tables` read ---------
#
# Each field carries a one-line `description` in its metadata. A field whose
# `healda.required` metadata is `false` is read when present.

_DESCRIPTION_KEY = b"description"
_REQUIRED_KEY = b"healda.required"
# Integer bit fields: any integer type is read, as is a fixed-size list of structs whose
# `_FLAG_BITS_KEY` field holds the bits.
_FLAG_BITS_KEY = b"healda.flag_bits"
VALUE_UNIT_KEY = b"nnja.value_unit"
_VALUE_ENCODING_KEY = b"nnja.archive.value_encoding"

# Each sensor's value column prefix in the archive; a channel column is
# f"{prefix}__ch_{channel:05d}". ATMS is antenna temperature. The other microwave columns are
# named brightness temperature but hold antenna temperature, except N15/N16 AMSU-A; the GSI
# UFS replay ingested them as is (ta2tb=false).
SENSOR_VALUE_PREFIX = {
    "amsua": "brightness_temperature",
    "amsub": "brightness_temperature",
    "mhs": "brightness_temperature",
    "atms": "antenna_temperature",
    "airs": "brightness_temperature",
    "iasi": "spectral_radiance",
    "cris": "spectral_radiance_code",
}
# Published as radiance, not kelvin; `transform` inverts Planck before the range check.
RADIANCE_PREFIXES = ("spectral_radiance", "spectral_radiance_code")
# A radiance sensor's channel columns may instead hold brightness temperature under this
# prefix, each declaring VALUE_UNIT_KEY = b"K".
KELVIN_PREFIX = "brightness_temperature"

# CrIS names the detector in its 3x3 array field_of_view and the cross-track look
# field_of_regard; the look is the scan position. Elsewhere field_of_view is the position.
SCAN_POSITION_COLUMN = {"cris": "field_of_regard"}
# Per-footprint source quality flags, each nonzero when the provider marks the footprint
# degraded. fov_quality_bad is derived in the ir48 store from fov_quality, 2025-01-01 on.
# A dotted name reads one field of a per-channel struct; any channel flagged flags the footprint.
SOURCE_QUALITY_FLAGS = {
    "cris": ("scan_quality", "fov_quality_bad"),
    "iasi": ("summary_quality",),
    "atms": (
        "scan_quality",
        "granule_quality",
        "channel_auxiliary.list.element.channel_quality",
    ),
}
# GPS-RO needs one of each pair.
GPSRO_ONE_OF = (
    ("occ_latitude", "tangent_latitude"),
    ("occ_longitude", "tangent_longitude"),
)


def _field(
    name: str,
    type: pa.DataType,
    description: str,
    *,
    required: bool = True,
    nullable: bool = True,
    flag_bits: str | None = None,
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Field:
    meta = {_DESCRIPTION_KEY: description.encode()}
    if not required:
        meta[_REQUIRED_KEY] = b"false"
    if flag_bits is not None:
        meta[_FLAG_BITS_KEY] = flag_bits.encode()
    meta.update(metadata or {})
    return pa.field(name, type, nullable=nullable, metadata=meta)


def _optional(name: str, type: pa.DataType, description: str, **kwargs) -> pa.Field:
    return _field(name, type, description, required=False, **kwargs)


_UTC_NS = pa.timestamp("ns", tz="UTC")

_PREPBUFR_SCHEMA = pa.schema(
    [
        _field("XOB", pa.float32(), "report longitude, degrees east"),
        _field("YOB", pa.float32(), "report latitude, degrees north"),
        _field("DHR", pa.float32(), "report time, hours from the file's cycle"),
        _field("POB", pa.float32(), "pressure, hPa; the ps value on surface reports"),
        _field("TYP", pa.float32(), "PrepBUFR report type (120, 220, 281, ...)"),
        _optional("ZOB", pa.float32(), "level height, m"),
        _optional("CAT", pa.float32(), "PrepBUFR level category"),
        _optional("ELV", pa.float32(), "station elevation, m"),
        _optional("XDR", pa.float32(), "balloon-drifted longitude, degrees east"),
        _optional("YDR", pa.float32(), "balloon-drifted latitude, degrees north"),
        _optional("HRDR", pa.float32(), "balloon-drifted time, hours from the cycle"),
        _optional("SID", pa.string(), "station id"),
        _optional("PQM", pa.float32(), "pressure quality mark"),
        _optional("QOB", pa.float32(), "specific humidity, mg/kg"),
        _optional("QQM", pa.float32(), "specific humidity quality mark"),
        _optional("TOB", pa.float32(), "temperature, degrees C"),
        _optional("TQM", pa.float32(), "temperature quality mark"),
        _optional("UOB", pa.float32(), "eastward wind, m/s"),
        _optional("VOB", pa.float32(), "northward wind, m/s"),
        _optional("WQM", pa.float32(), "wind quality mark"),
    ]
)

_GPSRO_SCHEMA = pa.schema(
    [
        _field("bending_angle", pa.float32(), "bending angle, rad"),
        _field("qfro", pa.int64(), "BUFR QFRO quality flags"),
        _field("obs_time", _UTC_NS, "observation time, UTC"),
        _field("impact_parameter", pa.float64(), "impact parameter, m"),
        _field(
            "earth_radius_curvature", pa.float64(), "local Earth radius of curvature, m"
        ),
        _field("satellite_id", pa.int64(), "receiver satellite id (BUFR SAID)"),
        _optional(
            "tangent_latitude", pa.float64(), "tangent point latitude, degrees north"
        ),
        _optional(
            "tangent_longitude", pa.float64(), "tangent point longitude, degrees east"
        ),
        _optional("occ_latitude", pa.float64(), "occultation latitude, degrees north"),
        _optional("occ_longitude", pa.float64(), "occultation longitude, degrees east"),
        _optional("impact_height_m", pa.float64(), "impact height above the geoid, m"),
        _optional("heit_at_impact_m", pa.float64(), "geopotential height at impact, m"),
        _optional("dry_pressure_hpa", pa.float64(), "dry pressure at impact, hPa"),
        _optional(
            "standard_atmosphere_pressure_hpa",
            pa.float64(),
            "US standard atmosphere pressure at impact, hPa",
        ),
        _optional(
            "blended_pressure_5km_hpa",
            pa.float64(),
            "dry pressure above 5 km blended to standard atmosphere below, hPa",
        ),
    ]
)

_QUALITY_DIAGNOSTICS = pa.list_(
    pa.struct(
        [
            ("source_occurrence", pa.uint8()),
            ("per_cent_confidence", pa.float64()),
            ("first_guess_use_qualifier", pa.uint32()),
            ("generating_application", pa.uint32()),
            ("manual_automatic_qc", pa.uint32()),
            ("nominal_confidence_threshold", pa.float64()),
            ("originating_centre_id", pa.uint32()),
            ("originating_centre_id_long", pa.uint32()),
            ("standard_generating_application", pa.uint32()),
        ]
    )
)

_SATWND_SCHEMA = pa.schema(
    [
        _field("time_utc", pa.timestamp("ns"), "observation time, UTC"),
        _field("da_window", pa.timestamp("ns"), "end-aligned 3-hour window, UTC"),
        _field("latitude", pa.float64(), "degrees north"),
        _field("longitude", pa.float64(), "degrees east"),
        _field("hpx4096_nest", pa.uint32(), "HEALPix NESTED pixel at nside 4096"),
        _field("wind_u_derived", pa.float64(), "eastward wind, m/s"),
        _field("wind_v_derived", pa.float64(), "northward wind, m/s"),
        _field("assigned_pressure", pa.float64(), "wind pressure, Pa"),
        _field("gsi_observation_type", pa.uint16(), "GSI observation type, 240-260"),
        _field("gsi_type_mapping_tier", pa.string(), "how the GSI type was assigned"),
        _field(
            "wind_computation_method", pa.uint32(), "BUFR SWCM code; QC screens on it"
        ),
        _field(
            "ncep_dump_subtype",
            pa.string(),
            "NCEP dump subset (NC005xxx); QC keys on it",
        ),
        # da_window cannot stand in: it splits one cycle across two windows.
        _field(
            "cycle", pa.timestamp("ns"), "cycle of the source file, UTC; QC keys on it"
        ),
        _optional("sdmedit_wind_quality_mark", pa.uint32(), "BUFR SWQM"),
        _optional("satellite_zenith_angle", pa.float64(), "degrees"),
        _optional("gsi_internal_subtype", pa.uint8(), "GSI sattab processing subtype"),
        _optional(
            "quality_diagnostics",
            _QUALITY_DIAGNOSTICS,
            "quality indicator per generating application (GNAP, PCCF); read with the QI screen",
        ),
    ]
)

_FOOTPRINT_FIELDS = (
    _field("latitude", pa.float64(), "degrees north"),
    _field("longitude", pa.float64(), "degrees east"),
    _field("time_utc", _UTC_NS, "observation time", nullable=False),
    _field("da_window", _UTC_NS, "end-aligned 3-hour window", nullable=False),
    _field("platform_id", pa.uint16(), "BUFR satellite id (SAID)"),
    _field("satellite_zenith_angle", pa.float32(), "unsigned, degrees"),
    _field("solar_zenith_angle", pa.float32(), "degrees"),
    _field(
        "field_of_view", pa.uint16(), "scan position; CrIS: detector in the 3x3 array"
    ),
)

_SCAN_POSITION_FIELDS = {
    "cris": _field("field_of_regard", pa.uint16(), "CrIS scan position"),
}

_HPX2048 = _field(
    "hpx2048_nest", pa.uint32(), "HEALPix NESTED pixel at nside 2048", nullable=False
)

_FLAG_FIELDS = {
    "scan_quality": ("scan quality flags", pa.uint32()),
    "granule_quality": ("ATMS granule quality flags", pa.uint16()),
    "summary_quality": ("IASI summary quality flag", pa.uint8()),
    "fov_quality_bad": ("CrIS field-of-view quality, nonzero when bad", pa.uint8()),
    "channel_auxiliary": (
        "ATMS channel quality bits, or the per-channel struct list holding them",
        pa.uint16(),
    ),
}

_VALUE_DESCRIPTION = {
    "antenna_temperature": "antenna temperature, K",
    "brightness_temperature": "brightness temperature, K",
    "spectral_radiance": "spectral radiance, mW m^-2 sr^-1 (cm^-1)^-1",
    "spectral_radiance_code": "spectral radiance, integer code; decoded by "
    "the field's nnja.archive.value_encoding",
}


def _satellite_schema(
    sensor: str,
    ir_channels: str | Mapping[str, Sequence[int]] | None = None,
    *,
    kelvin: bool = True,
) -> pa.Schema:
    """One footprint per row with one column per channel, as `NNJAWideLoader` reads
    `sensor`; `ir_channels` as in `table_schema`. Without `kelvin`, IASI and CrIS columns
    are the archive's radiance instead of brightness temperature."""
    # sensors_nnja imports this module.
    from healda.observations.sensors_nnja import SENSOR_CONFIGS

    if sensor not in SENSOR_VALUE_PREFIX:
        raise ValueError(
            f"unknown sensor {sensor!r}, expected one of {sorted(SENSOR_VALUE_PREFIX)}"
        )
    if isinstance(ir_channels, str):
        ir_channels = ir_spectral.ir_channel_preset(ir_channels)
    named = None if ir_channels is None else ir_channels.get(sensor)
    if named is not None and not len(named):
        raise ValueError(f"ir_channels names no channel for {sensor!r}")
    channels = sorted(SENSOR_CONFIGS[sensor].channels if named is None else named)

    fields = list(_FOOTPRINT_FIELDS)
    if sensor in _SCAN_POSITION_FIELDS:
        fields.append(_SCAN_POSITION_FIELDS[sensor])
    fields.append(_HPX2048)
    for flag in SOURCE_QUALITY_FLAGS.get(sensor, ()):
        column, _, leaf = flag.rpartition(".list.element.")
        column = column or flag
        description, flag_type = _FLAG_FIELDS[column]
        fields.append(
            _optional(
                column,
                flag_type,
                f"{description}; read to drop flagged footprints",
                flag_bits=leaf or column,
            )
        )

    prefix, value_type, metadata = SENSOR_VALUE_PREFIX[sensor], pa.float32(), None
    if kelvin and prefix in RADIANCE_PREFIXES:
        prefix, metadata = KELVIN_PREFIX, {VALUE_UNIT_KEY: b"K"}
    elif prefix == "spectral_radiance_code":
        value_type = pa.int32()
        metadata = {_VALUE_ENCODING_KEY: b'{"reference": int, "scale": int}'}
    for channel in channels:
        fields.append(
            _field(
                f"{prefix}__ch_{int(channel):05d}",
                value_type,
                f"channel {int(channel)} {_VALUE_DESCRIPTION[prefix]}",
                required=named is not None,
                metadata=metadata,
            )
        )
    return pa.schema(fields)


# --- The unified table: one row per observation, what the transform featurizes ----------
#
# Field name -> (description, required). Types are get_combined_observation_schema's.
_UNIFIED_FIELDS = {
    "Latitude": ("degrees north; every family", True),
    "Longitude": ("degrees east, 0 to 360; every family", True),
    "Absolute_Obs_Time": ("observation time, UTC; every family", True),
    "DA_window": (
        "end of the 3-hour window the row belongs to, UTC, on a 00/03/.../21 h "
        "boundary; selects the analyses that read the row; every family",
        True,
    ),
    "Platform_ID": (
        "platform id the model embeds; the packaged ids are "
        "`sensors.PLATFORM_NAME_TO_ID`, conventional rows carrying their variable's "
        "family (gps, ps, q, t, uv); every family",
        True,
    ),
    "Observation": (
        "normalized value, (physical value - mean) / stddev of its channel; the NNJA "
        "loaders take mean and stddev per Global_Channel_ID from "
        "`loaders.combined.channel_table(nnja_conv=True)`, satellite channels in K and "
        "conventional channels per variable and pressure level; every family",
        True,
    ),
    "Global_Channel_ID": (
        "channel id the model embeds; the packaged vocabulary is "
        "`loaders.combined.channel_table`, where a conventional row's channel is its "
        "variable at the pressure level nearest Pressure; every family",
        True,
    ),
    "Sat_Zenith_Angle": (
        "satellite zenith angle, degrees, negative on the first half of the scan; "
        "satellite rows, null on conventional rows",
        False,
    ),
    "Sol_Zenith_Angle": (
        "solar zenith angle, degrees; satellite rows, null on conventional rows",
        False,
    ),
    "Scan_Angle": (
        "scan angle from nadir, degrees, negative on the first half of the scan; "
        "satellite rows, null on conventional rows",
        False,
    ),
    "Pressure": (
        "pressure, hPa; conventional rows, null on satellite rows",
        False,
    ),
    "Height": (
        "height, m; conventional rows, null on satellite rows; a non-null height "
        "marks a row as conventional",
        False,
    ),
    "Observation_Type": (
        "PrepBUFR report type, SATWND GSI type (240-260) or GPS-RO receiver "
        "satellite id; conventional rows, null on satellite rows",
        False,
    ),
    "QC_Flag": ("not read", False),
    "Analysis_Use_Flag": ("not read", False),
    "local_channel_id": (
        "Global_Channel_ID minus its sensor's first Global_Channel_ID; every family",
        True,
    ),
    "sensor_id": (
        "sensor the row is grouped under, one of the transform's sensors; every family",
        True,
    ),
}


def _unified_schema() -> pa.Schema:
    fields = []
    for field in (
        get_combined_observation_schema().append(LOCAL_CHANNEL_ID).append(SENSOR_ID)
    ):
        description, required = _UNIFIED_FIELDS[field.name]
        fields.append(
            _field(
                field.name,
                field.type,
                description,
                required=required,
                nullable=not required,
            )
        )
    return pa.schema(fields)


_CONVENTIONAL_SCHEMAS = {
    "prepbufr": _PREPBUFR_SCHEMA,
    "gpsro": _GPSRO_SCHEMA,
    "satwnd": _SATWND_SCHEMA,
}


def table_schema(
    kind: str, ir_channels: str | Mapping[str, Sequence[int]] | None = None
) -> pa.Schema:
    """The schema of `kind`: ``"prepbufr"``, ``"gpsro"``, ``"satwnd"``, a satellite
    sensor, or ``"unified"``.

    For a satellite sensor, `ir_channels` is a preset name (``"ir32"``, ``"ir48"``) or a
    mapping of sounder to channel numbers; the channels it names are required, and
    without it every channel of the sensor's vocabulary is optional and at least one is
    required. IASI and CrIS columns are brightness temperature in K.

    ``"unified"`` is the table ``DAModel.run_analysis_unified`` takes and the NNJA
    loaders emit: one row per observation, thinned, quality controlled and normalized.
    """
    if kind in _CONVENTIONAL_SCHEMAS:
        return _CONVENTIONAL_SCHEMAS[kind]
    if kind == "unified":
        return _unified_schema()
    return _satellite_schema(kind, ir_channels)


def required_columns(schema: pa.Schema) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in schema
        if (field.metadata or {}).get(_REQUIRED_KEY) != b"false"
    )


def optional_columns(schema: pa.Schema) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in schema
        if (field.metadata or {}).get(_REQUIRED_KEY) == b"false"
    )


def _type_matches(actual: pa.DataType, expected: pa.Field) -> bool:
    """Whether a column of `actual` type reads as `expected`.

    Timestamps match on unit with any time zone, and flag bit fields take any integer type.
    """
    if actual == expected.type:
        return True
    if pa.types.is_timestamp(expected.type):
        return pa.types.is_timestamp(actual) and actual.unit == expected.type.unit
    leaf = (expected.metadata or {}).get(_FLAG_BITS_KEY)
    if leaf is None:
        return False
    if pa.types.is_fixed_size_list(actual) and pa.types.is_struct(actual.value_type):
        index = actual.value_type.get_field_index(leaf.decode())
        if index >= 0:
            actual = actual.value_type.field(index).type
    return pa.types.is_integer(actual)


def validate(
    table: pa.Table,
    kind: str,
    ir_channels: str | Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Raise ValueError naming every column of `table` that `kind`'s reader cannot read.

    `kind` is as in `table_schema`. For a satellite sensor, the value columns are checked
    as brightness temperature when `table` has any, as the loader reads them.
    """
    if kind in SENSOR_VALUE_PREFIX:
        kelvin = any(
            name.startswith(f"{KELVIN_PREFIX}__ch_") for name in table.column_names
        )
        expected = _satellite_schema(kind, ir_channels, kelvin=kelvin)
    else:
        expected = table_schema(kind)

    problems = []
    names = table.schema.names
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{kind}: repeated column names {repeated}")
    missing = [
        name for name in required_columns(expected) if name not in table.schema.names
    ]
    if missing:
        problems.append(f"missing columns {missing}")
    if kind == "gpsro":
        for pair in GPSRO_ONE_OF:
            if not any(name in table.schema.names for name in pair):
                problems.append(f"missing one of {list(pair)}")
    channel_columns = [name for name in expected.names if "__ch_" in name]
    if channel_columns and not any(
        name in table.schema.names for name in channel_columns
    ):
        prefix = channel_columns[0].split("__ch_")[0]
        problems.append(f"no {prefix}__ch_NNNNN channel columns of this sensor")

    for field in expected:
        if field.name not in table.schema.names:
            continue
        actual = table.schema.field(field.name)
        if not _type_matches(actual.type, field):
            problems.append(f"{field.name}: type {actual.type}, expected {field.type}")
            continue
        if not field.nullable and table[field.name].null_count:
            problems.append(f"{field.name}: {table[field.name].null_count} nulls")
        expected_meta = field.metadata or {}
        actual_meta = actual.metadata or {}
        unit = expected_meta.get(VALUE_UNIT_KEY)
        if unit is not None and actual_meta.get(VALUE_UNIT_KEY) != unit:
            problems.append(
                f"{field.name}: field metadata {VALUE_UNIT_KEY.decode()} must be "
                f"{unit.decode()}"
            )
        if _VALUE_ENCODING_KEY in expected_meta:
            try:
                encoding = ir_spectral.archive_value_encoding(actual)
            except (ValueError, KeyError):
                encoding = {}
            if not encoding:
                problems.append(
                    f"{field.name}: field metadata {_VALUE_ENCODING_KEY.decode()} must be "
                    'JSON with "reference" and "scale"'
                )
    if problems:
        raise ValueError(f"{kind}: " + "; ".join(problems))
