# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
from healda.observations import schema as reader_schema
from healda.observations.preprocessing import ir_spectral

from scripts.etl.nnja.sat import thin_nnja_ir
from scripts.etl.nnja.sat.archive_schema import (
    SENSOR_SPECS,
    ChannelManifest,
    archive_schema,
)
from tests.scripts.etl.test_nnja_etl_sat_archive import _test_structure

FOOTPRINT = [
    "latitude",
    "longitude",
    "time_utc",
    "da_window",
    "platform_id",
    "satellite_zenith_angle",
    "solar_zenith_angle",
    "field_of_view",
]
QC = [
    "scan_line",
    "orbit_number",
    "source_file_ordinal",
    "source_message_index",
    "source_subset_index",
]
THINNED_COLUMNS = {
    "cris": [
        *FOOTPRINT,
        "field_of_regard",
        "hpx2048_nest",
        *QC,
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
        "fov_quality_bad",
    ],
    "iasi": [
        *FOOTPRINT,
        "hpx2048_nest",
        *QC,
        "summary_quality",
        "geometric_quality",
    ],
}


def test_fov_quality_bad_is_set_by_bit_14_or_15_in_any_band():
    band = pa.struct([("fov_quality", pa.uint32())])
    bands = pa.array(
        [
            [{"fov_quality": 0}, {"fov_quality": 1 << 10}, {"fov_quality": 0}],
            [{"fov_quality": 0}, {"fov_quality": 0}, {"fov_quality": 1 << 15}],
            [{"fov_quality": None}, {"fov_quality": 1 << 14}, {"fov_quality": 0}],
            None,
        ],
        pa.list_(band),
    )
    window = pa.table({"platform_id": [224] * 4, "band_calibration_quality": bands})
    out = thin_nnja_ir.with_fov_quality_bad(window)
    assert out.column_names == ["platform_id", "fov_quality_bad"]
    assert out["fov_quality_bad"].to_pylist() == [0, 1, 1, 0]


def test_thinned_columns_keep_their_order_and_reader_types(tmp_path) -> None:
    for sensor, expected in THINNED_COLUMNS.items():
        wanted = sorted(ir_spectral.ir_channel_preset("ir48")[sensor])
        others = [channel for channel in range(1, 10_000) if channel not in wanted]
        count = SENSOR_SPECS[sensor].channel_count
        manifest = ChannelManifest.from_ids(
            sensor, [*wanted, *others[: count - len(wanted)]], release="test"
        )
        schema = archive_schema(
            sensor, manifest, source_structure=_test_structure(sensor)
        )
        path = tmp_path / f"{sensor}.parquet"
        pq.write_table(schema.empty_table(), path)

        thinned = thin_nnja_ir.projection(pq.ParquetFile(path), sensor, "ir48").schema
        channels = [
            reader_schema.channel_field(sensor, channel, kelvin=False).name
            for channel in wanted
        ]
        assert thinned.names == [*expected, *channels]
        reader = {field.name: field for field in reader_schema.footprint_fields(sensor)}
        for name in expected:
            if name in reader:
                assert thinned.field(name).type == reader[name].type, name
