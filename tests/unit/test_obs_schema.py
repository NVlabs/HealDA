# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NNJA archive table contract in ``healda.observations.schema``."""

import json

import numpy as np
import pyarrow as pa
import pytest

from healda.observations import schema, sensors_nnja
from healda.observations.preprocessing import ir_spectral

UTC_NS = pa.timestamp("ns", tz="UTC")
NAIVE_NS = pa.timestamp("ns")
TIMES = np.array(["2025-01-01T00:00", "2025-01-01T01:00"], dtype="datetime64[ns]")


def _floats(type: pa.DataType = pa.float32()) -> pa.Array:
    return pa.array([1.0, 2.0], type=type)


def _ints(type: pa.DataType) -> pa.Array:
    return pa.array([1, 2], type=type)


def _prepbufr() -> pa.Table:
    columns = {name: _floats() for name in ("XOB", "YOB", "DHR", "POB", "TYP", "TOB")}
    columns["SID"] = pa.array(["72403", "72404"])
    return pa.table(columns)


def _gpsro() -> pa.Table:
    return pa.table(
        {
            "satellite_id": _ints(pa.int64()),
            "occ_latitude": _floats(pa.float64()),
            "occ_longitude": _floats(pa.float64()),
            "impact_parameter": _floats(pa.float64()),
            "bending_angle": _floats(),
            "qfro": _ints(pa.int64()),
            "earth_radius_curvature": _floats(pa.float64()),
            "obs_time": pa.array(TIMES, type=UTC_NS),
        }
    )


def _satwnd() -> pa.Table:
    return pa.table(
        {
            "ncep_dump_subtype": pa.array(["NC005030", "NC005030"]),
            "gsi_observation_type": _ints(pa.uint16()),
            "time_utc": pa.array(TIMES, type=NAIVE_NS),
            "da_window": pa.array(TIMES, type=NAIVE_NS),
            "cycle": pa.array(TIMES, type=NAIVE_NS),
            "latitude": _floats(pa.float64()),
            "longitude": _floats(pa.float64()),
            "hpx4096_nest": _ints(pa.uint32()),
            "wind_u_derived": _floats(pa.float64()),
            "wind_v_derived": _floats(pa.float64()),
            "wind_computation_method": _ints(pa.uint32()),
            "assigned_pressure": _floats(pa.float64()),
            "gsi_type_mapping_tier": pa.array(["1-direct-gsi-sattab"] * 2),
        }
    )


def _footprints() -> dict[str, pa.Array]:
    return {
        "latitude": _floats(pa.float64()),
        "longitude": _floats(pa.float64()),
        "time_utc": pa.array(TIMES, type=UTC_NS),
        "da_window": pa.array(TIMES, type=UTC_NS),
        "platform_id": _ints(pa.uint16()),
        "satellite_zenith_angle": _floats(),
        "solar_zenith_angle": _floats(),
        "field_of_view": _ints(pa.uint16()),
        "hpx2048_nest": _ints(pa.uint32()),
    }


def _atms() -> pa.Table:
    columns = _footprints()
    quality = pa.StructArray.from_arrays(
        [pa.array(np.zeros(2 * 22, dtype=np.uint16))], ["channel_quality"]
    )
    columns["channel_auxiliary"] = pa.FixedSizeListArray.from_arrays(quality, 22)
    columns["scan_quality"] = _ints(pa.uint32())
    for channel in range(1, 23):
        columns[f"antenna_temperature__ch_{channel:05d}"] = _floats()
    return pa.table(columns)


def _cris_radiance() -> pa.Table:
    encoding = json.dumps({"reference": -100000, "scale": 7}).encode()
    fields, arrays = [], []
    columns = {**_footprints(), "field_of_regard": _ints(pa.uint16())}
    for name, array in columns.items():
        fields.append(pa.field(name, array.type))
        arrays.append(array)
    for channel in ir_spectral.ir_channel_preset("ir48")["cris"]:
        fields.append(
            pa.field(
                f"spectral_radiance_code__ch_{channel:05d}",
                pa.int32(),
                metadata={b"nnja.archive.value_encoding": encoding},
            )
        )
        arrays.append(_ints(pa.int32()))
    return pa.table(arrays, schema=pa.schema(fields))


@pytest.mark.parametrize(
    "kind, table, ir_channels",
    [
        ("prepbufr", _prepbufr(), None),
        ("gpsro", _gpsro(), None),
        ("satwnd", _satwnd(), None),
        ("atms", _atms(), None),
        ("cris", _cris_radiance(), "ir48"),
    ],
)
def test_archive_format_tables_validate(kind, table, ir_channels):
    schema.validate(table, kind, ir_channels=ir_channels)


def test_missing_and_mistyped_columns_are_named():
    table = _prepbufr().drop_columns(["TYP"])
    table = table.set_column(0, "XOB", table["XOB"].cast(pa.string()))
    with pytest.raises(ValueError) as error:
        schema.validate(table, "prepbufr")
    message = str(error.value)
    assert "missing columns ['TYP']" in message
    assert "XOB: type string, expected float" in message


def test_a_preset_requires_its_channels_and_kelvin_columns_declare_their_unit():
    table = _footprints() | {"field_of_regard": _ints(pa.uint16())}
    channels = ir_spectral.ir_channel_preset("ir32")["cris"]
    for channel in channels[1:]:
        table[f"brightness_temperature__ch_{channel:05d}"] = _floats()
    with pytest.raises(ValueError) as error:
        schema.validate(pa.table(table), "cris", ir_channels="ir32")
    message = str(error.value)
    assert f"brightness_temperature__ch_{channels[0]:05d}" in message
    assert "nnja.value_unit must be K" in message


def test_an_empty_channel_selection_is_rejected():
    with pytest.raises(ValueError, match="names no channel for 'cris'"):
        schema.validate(_cris_radiance(), "cris", ir_channels={"cris": []})


def test_cris_codes_need_a_readable_encoding():
    table = _cris_radiance()
    field = table.schema.field(-1).with_metadata(
        {b"nnja.archive.value_encoding": b'{"reference": 0}'}
    )
    table = table.set_column(table.num_columns - 1, field, table.column(-1))
    with pytest.raises(ValueError, match='JSON with "reference" and "scale"'):
        schema.validate(table, "cris", ir_channels="ir48")


def test_repeated_column_names_are_named():
    table = _prepbufr()
    table = table.append_column("XOB", table["XOB"])
    with pytest.raises(ValueError, match=r"repeated column names \['XOB'\]"):
        schema.validate(table, "prepbufr")


def test_gpsro_needs_an_occultation_or_tangent_location():
    with pytest.raises(
        ValueError, match=r"one of \['occ_latitude', 'tangent_latitude'\]"
    ):
        schema.validate(_gpsro().drop_columns(["occ_latitude"]), "gpsro")


def test_every_field_describes_itself():
    schemas = [schema.table_schema(kind) for kind in ("prepbufr", "gpsro", "satwnd")]
    schemas += [schema.table_schema(sensor) for sensor in sensors_nnja.SENSOR_ORDER]
    schemas.append(schema.table_schema("unified"))
    for each in schemas:
        for field in each:
            assert field.metadata[b"description"], field.name


def _unified() -> pa.Table:
    return pa.table(
        {
            "Latitude": _floats(),
            "Longitude": _floats(),
            "Absolute_Obs_Time": pa.array(TIMES, type=NAIVE_NS),
            "DA_window": pa.array(TIMES, type=NAIVE_NS),
            "Platform_ID": _ints(pa.uint16()),
            "Observation": _floats(),
            "Global_Channel_ID": _ints(pa.uint16()),
            "local_channel_id": _ints(pa.uint16()),
            "sensor_id": _ints(pa.uint16()),
            "Height": _floats(),
        }
    )


def test_unified_tables_validate_and_name_missing_or_mistyped_columns():
    schema.validate(_unified(), "unified")
    table = _unified().drop_columns(["DA_window"])
    table = table.set_column(
        table.schema.get_field_index("Height"),
        "Height",
        table["Height"].cast(pa.string()),
    )
    with pytest.raises(ValueError) as error:
        schema.validate(table, "unified")
    message = str(error.value)
    assert "missing columns ['DA_window']" in message
    assert "Height: type string, expected float" in message
