# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Ruff does not infer this co-located contract module as a pytest file.
# ruff: noqa: S101
import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from healda.observations import schema as reader_schema

from scripts.etl.nnja.sat.archive_etl import BufrMessageHeader
from scripts.etl.nnja.sat.archive_runner import (
    _message_header_manifest,
)
from scripts.etl.nnja.sat.archive_schema import (
    SENSOR_SPECS,
    ChannelManifest,
    FileProvenance,
    SourceFile,
    SourceStructure,
    archive_schema,
    bind_file_metadata,
    derive_footprint_id,
    descriptor_coverage,
)
from scripts.etl.nnja.archive_common.archive_writer import (
    _window_parts,
    parquet_leaf_paths,
    writer_options,
)
from scripts.etl.nnja.sat.archive_writer import (
    SounderArchiveDayWriter,
    _delta_encoded_columns,
    validate_parquet_contract,
)


def write_archive_day(table, path, *, sensor, schema, provenance):
    with SounderArchiveDayWriter(path, sensor=sensor, schema=schema) as writer:
        writer.write(table)
        writer.bind_provenance(provenance)
    return writer.summary


def default_test_manifests() -> dict[str, ChannelManifest]:
    """Build synthetic manifests for contract tests only."""

    return {
        sensor: ChannelManifest.from_ids(
            sensor,
            range(1, spec.channel_count + 1),
            release="synthetic-contiguous-contract-test",
        )
        for sensor, spec in SENSOR_SPECS.items()
    }


def _test_nulls(length: int, data_type: pa.DataType) -> pa.Array:
    if pa.types.is_fixed_size_list(data_type):
        return pa.FixedSizeListArray.from_arrays(
            _test_nulls(length * data_type.list_size, data_type.value_type),
            data_type.list_size,
        )
    if pa.types.is_struct(data_type):
        return pa.StructArray.from_arrays(
            [_test_nulls(length, field.type) for field in data_type],
            fields=list(data_type),
        )
    return pa.nulls(length, type=data_type)


def _test_table(schema: pa.Schema) -> pa.Table:
    windows = [
        datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
    ]
    arrays = []
    for field in schema:
        if field.name in {"da_window", "time_utc", "source_message_time"}:
            arrays.append(pa.array(windows, type=field.type))
        elif field.name == "source_file_ordinal":
            arrays.append(pa.array([0, 0, 0, 0], type=field.type))
        elif field.name == "source_message_index":
            arrays.append(pa.array([1, 1, 2, 2], type=field.type))
        elif field.name == "source_subset_index":
            arrays.append(pa.array([0, 1, 0, 1], type=field.type))
        elif field.name == "source_data_category":
            arrays.append(pa.array([21, 21, 21, 21], type=field.type))
        elif field.name == "hpx2048_nest":
            arrays.append(pa.array([0, 1, 2, 3], type=field.type))
        else:
            arrays.append(_test_nulls(4, field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _header_manifest(structure: SourceStructure, subset_count: int) -> dict:
    header = BufrMessageHeader(
        index=0,
        edition=4,
        master_table=0,
        center=7,
        sub_center=0,
        update_sequence=0,
        optional_section_present=False,
        reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        data_category=21,
        international_sub_category=0,
        local_sub_category=0,
        master_table_version=41,
        local_table_version=0,
        subset_count=subset_count,
        compressed=True,
        section3_flags=0x40,
        descriptors=structure.observation_descriptors,
    )
    return _message_header_manifest((header,))


def _test_structure(sensor: str) -> SourceStructure:
    if sensor == "cris":
        occurrences = (
            {
                "decoder_path": "DRF16BIT.SRAD",
                "mnemonic": "SRAD",
                "descriptor": "0-14-044",
                "units": "W M**-2 SR**-1 CM",
                "effective_scale": 7,
                "effective_reference": -100_000,
                "effective_width": 22,
                "source_missing_code": 4_194_303,
            },
        )
        return SourceStructure(
            sensor="cris",
            embedded_table_sha256="1" * 64,
            observation_descriptors=("3-10-198",),
            active_sequences=({"descriptor": "3-10-198"},),
            active_elements=({"descriptor": "0-14-044", "mnemonic": "SRAD"},),
            effective_occurrences=occurrences,
            decoded_physical_schema=({"position": 56, "name": "DRF16BIT"},),
        )
    return SourceStructure(
        sensor=sensor,
        embedded_table_sha256="1" * 64,
        observation_descriptors=("3-00-001",),
        active_sequences=(),
        active_elements=({"descriptor": "0-01-001", "mnemonic": "TEST"},),
        effective_occurrences=(
            {
                "ordinal": 0,
                "path": "TEST[0]",
                "descriptor": "0-01-001",
                "effective_scale": 0,
                "effective_reference": 0,
                "effective_bits": 8,
            },
        ),
        decoded_physical_schema=({"position": 0, "name": "TEST"},),
    )


def test_source_structure_persists_one_semantic_view() -> None:
    payload = _test_structure("mhs").as_payload()

    assert "structural_effective_occurrences" not in payload
    assert "structural_decoded_physical_schema" not in payload
    assert set(payload) == {
        "format_version",
        "sensor",
        "embedded_table_sha256",
        "observation_descriptors",
        "active_sequences",
        "active_elements",
        "effective_occurrences",
        "decoded_physical_schema",
    }


@pytest.mark.parametrize("sensor", sorted(SENSOR_SPECS))
def test_every_active_descriptor_is_accounted_for(sensor: str) -> None:
    coverage = descriptor_coverage(sensor)
    assert coverage["unaccounted"] == []
    assert coverage["stored_or_manifest"]


@pytest.mark.parametrize("sensor", sorted(SENSOR_SPECS))
def test_schema_is_self_describing(sensor: str) -> None:
    manifest = default_test_manifests()[sensor]
    schema = archive_schema(sensor, manifest, source_structure=_test_structure(sensor))
    assert len(schema.names) == len(set(schema.names))
    assert b"nnja.archive.contract" in (schema.metadata or {})
    assert b"nnja.archive.schema_id" in (schema.metadata or {})
    assert "footprint_id" not in schema.names
    assert schema.field("hpx2048_nest").nullable
    hpx = json.loads(schema.metadata[b"nnja.archive.contract"])["hpx2048"]
    assert hpx["required"] is True
    assert hpx["null_when_source_geolocation_invalid"] is True
    assert hpx["implementation"].startswith("earth2grid.healpix.Grid")
    for field in schema:
        metadata = field.metadata or {}
        assert b"nnja.archive.role" in metadata
        assert b"nnja.archive.data_class" in metadata
        assert b"nnja.archive.axes" in metadata
        assert not any(key.startswith(b"e2s.") for key in metadata)


# schema_id of each sensor's contract-test schema at CONTRACT_VERSION; a change here
# changes the published archive's identity.
GOLDEN_SCHEMA_IDS = {
    "amsua": "e00316a40b68d0f8b3d059755116cab37ea29eb488a267e70361b057a05bc13e",
    "amsub": "5adea038f26fa9ba063c9884e8f408f9e8a31d1025751587c80727cd385a8fc0",
    "atms": "90cc009a16c3763288aba25e5793718e6f82ea751b2d56bccee365ed36e08678",
    "mhs": "3b8dc2364cbd80b1702ca43e5f99229255444164297b3c2e9a484045af6b1025",
    "iasi": "3591176998c89243e02dc8b8c81b3999baa5f86b6a7888a9d878ab3ee6bb8b13",
    "cris": "15c43a41c2befb57181037348ee3b0dcf02a1306d9659166303890cfc0ec088c",
    "airs": "c7147012da3d28ad392fe8c5c17642e53bfb0e939b69add7199eb773673c65f2",
}


@pytest.mark.parametrize("sensor", sorted(GOLDEN_SCHEMA_IDS))
def test_schema_id_is_unchanged(sensor: str) -> None:
    schema = archive_schema(
        sensor,
        default_test_manifests()[sensor],
        source_structure=_test_structure(sensor),
    )
    schema_id = json.loads(schema.metadata[b"nnja.archive.schema_id"])
    assert schema_id == GOLDEN_SCHEMA_IDS[sensor]


@pytest.mark.parametrize("sensor", sorted(SENSOR_SPECS))
def test_reader_fields_are_the_archive_fields(sensor: str) -> None:
    manifest = default_test_manifests()[sensor]
    schema = archive_schema(sensor, manifest, source_structure=_test_structure(sensor))
    base = [
        *reader_schema.footprint_fields(sensor),
        *(
            reader_schema.channel_field(sensor, channel.source_channel_id, kelvin=False)
            for channel in manifest.channels
        ),
    ]
    # fov_quality_bad is derived in the thinned IR store, not the archive.
    for field in base:
        if field.name == "fov_quality_bad":
            continue
        archive_field = schema.field(field.name)
        assert archive_field.type == field.type, field.name
        assert archive_field.nullable == field.nullable, field.name
    reader_schema.validate(_test_table(schema), sensor, ir_channels={sensor: [1]})


def test_manifest_is_part_of_schema_identity() -> None:
    first = ChannelManifest.from_ids("atms", range(1, 23), release="first")
    second = ChannelManifest.from_ids("atms", range(2, 24), release="second")
    first_schema = archive_schema(
        "atms", first, source_structure=_test_structure("atms")
    )
    second_schema = archive_schema(
        "atms", second, source_structure=_test_structure("atms")
    )
    assert first_schema.names != second_schema.names
    assert (
        first_schema.metadata[b"nnja.archive.schema_id"]
        != second_schema.metadata[b"nnja.archive.schema_id"]
    )


def test_virtual_footprint_id_has_a_frozen_formula() -> None:
    identifier = derive_footprint_id("00" * 32, 1, 2)
    assert len(identifier) == 16
    assert identifier.hex() == "e3c12ab31d5553157689a5ded3e9929b"
    assert identifier == derive_footprint_id("00" * 32, 1, 2)
    assert identifier != derive_footprint_id("00" * 32, 1, 3)


def test_semantic_classes_axes_and_flag_kind_are_machine_readable() -> None:
    schema = archive_schema(
        "cris",
        default_test_manifests()["cris"],
        source_structure=_test_structure("cris"),
    )
    primary = schema.field("spectral_radiance_code__ch_00001").metadata or {}
    assert json.loads(primary[b"nnja.archive.data_class"]) == "primary_data"

    scenes = schema.field("collocated_imager_scenes").metadata or {}
    assert json.loads(scenes[b"nnja.archive.data_class"]) == "secondary_data"
    assert [axis["length"] for axis in json.loads(scenes[b"nnja.archive.axes"])] == [
        7,
        16,
        2,
    ]

    quality = schema.field("scan_quality").metadata or {}
    descriptors = json.loads(quality[b"nnja.archive.source_descriptors"])
    assert descriptors[0]["descriptor"] == "0-33-075"
    assert descriptors[0]["value_kind"] == "flag"
    assert descriptors[0]["interpretation_table_key"] == "0-33-075"

    bands = schema.field("band_calibration_quality")
    assert pa.types.is_fixed_size_list(bands.type)
    assert bands.type.list_size == 3
    assert [child.name for child in bands.type.value_type] == [
        "band_type",
        "start_wavenumber_m_inverse",
        "end_wavenumber_m_inverse",
        "start_channel",
        "end_channel",
        "calibration_quality",
        "fov_quality",
    ]
    for name in (
        "primary_spectrum_band_type",
        "guard_spectrum_band_type",
        "collocated_imager_band_type",
    ):
        assert schema.get_field_index(name) >= 0


def test_cris_primary_value_encoding_is_explicit_and_exact() -> None:
    schema = archive_schema(
        "cris",
        default_test_manifests()["cris"],
        source_structure=_test_structure("cris"),
    )
    metadata = schema.metadata or {}
    encoding = json.loads(metadata[b"nnja.archive.primary_value_encoding"])

    assert encoding == {
        "bit_width": 22,
        "descriptor": "0-14-044",
        "formula": "(code + reference) * 10**(-scale)",
        "id": "cris_srad_014044_s7_r-100000_w22_v1",
        "null_representation": "arrow_null",
        "physical_dtype": "float32",
        "physical_units": "W m^-2 sr^-1 (cm^-1)^-1",
        "reference": -100000,
        "scale": 7,
        "source_missing_code": 4194303,
        "source_units": "W M**-2 SR**-1 CM",
        "storage_dtype": "int32",
    }
    assert [
        (
            None
            if code == encoding["source_missing_code"]
            else (code + encoding["reference"]) * 10 ** (-encoding["scale"])
        )
        for code in (100000, 100001, encoding["source_missing_code"])
    ] == [0.0, 1e-7, None]


def test_cris_encoding_is_derived_into_schema_fields_and_file_footer() -> None:
    original = _test_structure("cris")
    occurrences = tuple(
        {
            **occurrence,
            "descriptor": "0-14-199",
            "effective_scale": 6,
            "effective_reference": -50_000,
            "effective_width": 21,
            "source_missing_code": 2_097_151,
        }
        for occurrence in original.effective_occurrences
    )
    structure = replace(original, effective_occurrences=occurrences)
    schema = archive_schema(
        "cris",
        default_test_manifests()["cris"],
        source_structure=structure,
    )
    encoding = json.loads(schema.metadata[b"nnja.archive.primary_value_encoding"])
    assert encoding["descriptor"] == "0-14-199"
    assert encoding["scale"] == 6
    assert encoding["reference"] == -50_000
    assert encoding["bit_width"] == 21

    primary_metadata = schema.field("spectral_radiance_code__ch_00001").metadata
    assert primary_metadata is not None
    assert json.loads(primary_metadata[b"nnja.archive.value_encoding"]) == encoding
    descriptor = json.loads(primary_metadata[b"nnja.archive.source_descriptors"])[0]
    assert descriptor["descriptor"] == "0-14-199"
    assert descriptor["effective_scale"] == 6

    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                "cris.bufr",
                "a" * 64,
                1,
                "2024-01-01T00:00:00+00:00",
                "NC021206",
                message_header_manifest=_header_manifest(structure, 1),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=1,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    bound = bind_file_metadata(schema, provenance)
    embedded = json.loads(bound.metadata[b"nnja.archive.source_structures"])
    assert embedded[0]["structure"]["effective_occurrences"][0]["effective_scale"] == 6


def test_manifest_count_is_strict() -> None:
    manifest = ChannelManifest.from_ids("atms", range(1, 22), release="short")
    with pytest.raises(ValueError, match="needs 22 channels"):
        archive_schema("atms", manifest, source_structure=_test_structure("atms"))


@pytest.mark.parametrize("sensor", sorted(SENSOR_SPECS))
def test_physical_writer_accepts_every_sensor_schema(
    sensor: str, tmp_path: Path
) -> None:
    structure = _test_structure(sensor)
    schema = archive_schema(
        sensor, default_test_manifests()[sensor], source_structure=structure
    )
    table = _test_table(schema).slice(0, 1)
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                object_key=f"{sensor}.bufr",
                sha256="c" * 64,
                size_bytes=1,
                cycle="2024-01-01T00:00:00+00:00",
                message_type="NC021000",
                message_header_manifest=_header_manifest(structure, 1),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=1,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    output = tmp_path / f"{sensor}.parquet"
    write_archive_day(
        table,
        output,
        sensor=sensor,
        schema=schema,
        provenance=provenance,
    )
    assert validate_parquet_contract(output)["rows"] == 1
    if sensor == "cris":
        saved_metadata = pq.read_schema(output).metadata or {}
        assert json.loads(
            saved_metadata[b"nnja.archive.primary_value_encoding"]
        ) == json.loads(schema.metadata[b"nnja.archive.primary_value_encoding"])


def test_writer_can_bind_actual_provenance_after_streaming(tmp_path: Path) -> None:
    sensor = "mhs"
    structure = _test_structure(sensor)
    schema = archive_schema(
        sensor, default_test_manifests()[sensor], source_structure=structure
    )
    table = _test_table(schema).slice(0, 1)
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                object_key="mhs.bufr",
                sha256="c" * 64,
                size_bytes=1,
                cycle="2024-01-01T00:00:00+00:00",
                message_type="mhs",
                message_header_manifest=_header_manifest(structure, 1),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=1,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    output = tmp_path / "late-bound.parquet"
    writer = SounderArchiveDayWriter(output, sensor=sensor, schema=schema)
    writer.write(table)
    writer.bind_provenance(provenance)
    writer.close()

    assert validate_parquet_contract(output)["rows"] == 1


def test_mixed_template_source_provenance_round_trips(tmp_path: Path) -> None:
    sensor = "mhs"
    first = _test_structure(sensor)
    second = replace(
        first,
        embedded_table_sha256="2" * 64,
        observation_descriptors=("3-00-002",),
    )
    headers = (
        BufrMessageHeader(
            index=1,
            edition=4,
            master_table=0,
            center=7,
            sub_center=0,
            update_sequence=0,
            optional_section_present=False,
            reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            data_category=21,
            international_sub_category=0,
            local_sub_category=0,
            master_table_version=41,
            local_table_version=0,
            subset_count=2,
            compressed=True,
            section3_flags=0x40,
            descriptors=first.observation_descriptors,
        ),
        BufrMessageHeader(
            index=2,
            edition=4,
            master_table=0,
            center=7,
            sub_center=0,
            update_sequence=0,
            optional_section_present=False,
            reference_time=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
            data_category=21,
            international_sub_category=0,
            local_sub_category=0,
            master_table_version=41,
            local_table_version=0,
            subset_count=2,
            compressed=True,
            section3_flags=0x40,
            descriptors=second.observation_descriptors,
        ),
    )
    source_file = SourceFile(
        object_key="mixed.bufr",
        sha256="c" * 64,
        size_bytes=1,
        cycle="2024-01-01T00:00:00+00:00",
        message_type=sensor,
        source_structure_sha256s=(first.fingerprint, second.fingerprint),
        message_header_manifest=_message_header_manifest(headers),
    )
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(source_file,),
        source_structures=(first, second),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=4,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    schema = archive_schema(
        sensor,
        default_test_manifests()[sensor],
        source_structure=first,
    )
    output = tmp_path / "mixed-template.parquet"

    write_archive_day(
        _test_table(schema),
        output,
        sensor=sensor,
        schema=schema,
        provenance=provenance,
    )

    contract = validate_parquet_contract(output)
    saved_source = contract["source_files"][0]
    assert "source_structure_sha256" not in saved_source
    assert saved_source["source_structure_sha256s"] == [
        first.fingerprint,
        second.fingerprint,
    ]
    assert contract["source_structure_hashes"] == sorted(
        [first.fingerprint, second.fingerprint]
    )
    invalid_source = replace(
        source_file,
        source_structure_sha256s=(first.fingerprint,),
    )
    with pytest.raises(ValueError, match="require two hashes"):
        replace(provenance, source_files=(invalid_source,)).validate()


def test_cris_writer_policy_targets_nested_parquet_leaves(tmp_path: Path) -> None:
    schema = archive_schema(
        "cris",
        default_test_manifests()["cris"],
        source_structure=_test_structure("cris"),
    )
    options = writer_options(schema, _delta_encoded_columns("cris", schema))
    dictionary_paths = set(options["use_dictionary"])
    primary_name = "spectral_radiance_code__ch_00001"
    nested_path = (
        "collocated_imager_scenes.list.element.channels.list.element."
        "statistics.list.element.radiance"
    )

    assert nested_path in parquet_leaf_paths(schema)
    assert nested_path in dictionary_paths
    assert "collocated_imager_scenes" not in dictionary_paths
    assert primary_name not in dictionary_paths
    assert options["column_encoding"][primary_name] == "DELTA_BINARY_PACKED"

    table = _test_table(schema).slice(0, 1)
    table = table.set_column(
        schema.get_field_index(primary_name),
        schema.field(primary_name),
        pa.array([123], type=schema.field(primary_name).type),
    )
    scene = {
        "y_angle": 0.0,
        "z_angle": 0.0,
        "clear_fraction": 100,
        "channels": [
            {
                "channel_number": channel + 1,
                "statistics": [
                    {"statistic_type": 1, "radiance": 1.0},
                    {"statistic_type": 2, "radiance": 2.0},
                ],
            }
            for channel in range(16)
        ],
    }
    imager_name = "collocated_imager_scenes"
    table = table.set_column(
        schema.get_field_index(imager_name),
        schema.field(imager_name),
        pa.array([[scene] * 7], type=schema.field(imager_name).type),
    )

    output = tmp_path / "cris-encoding-policy.parquet"
    pq.write_table(table, output, **options)
    parquet = pq.ParquetFile(output)
    columns = {
        parquet.metadata.row_group(0)
        .column(index)
        .path_in_schema: parquet.metadata.row_group(0).column(index)
        for index in range(parquet.metadata.num_columns)
    }
    assert set(columns) == set(parquet_leaf_paths(schema))
    assert columns[primary_name].compression == "ZSTD"
    assert "DELTA_BINARY_PACKED" in columns[primary_name].encodings
    assert "RLE_DICTIONARY" not in columns[primary_name].encodings
    assert columns[nested_path].compression == "ZSTD"
    assert "RLE_DICTIONARY" in columns[nested_path].encodings


def test_window_aligned_parquet_round_trip(tmp_path: Path) -> None:
    schema = archive_schema(
        "amsua",
        default_test_manifests()["amsua"],
        source_structure=_test_structure("amsua"),
    )
    table = _test_table(schema)
    structure = _test_structure("amsua")
    unusable = {
        "object_key": "gdas.20240101.t06z.1bamua.tm00.bufr_d",
        "cycle": "2024-01-01T06:00:00+00:00",
        "size_bytes": 7,
        "sha256": "b" * 64,
        "reason": "no_observation_subsets",
    }
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                object_key="gdas.20240101.t00z.1bamua.tm00.bufr_d",
                sha256="a" * 64,
                size_bytes=12_345,
                cycle="2024-01-01T00:00:00+00:00",
                message_type="NC021023",
                message_header_manifest=_header_manifest(structure, 4),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=4,
        missing_cycles=("06Z", "12Z", "18Z"),
        unusable_source_objects=(unusable,),
        created_utc=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    output = tmp_path / "amsua-20240101.parquet"
    with SounderArchiveDayWriter(output, sensor="amsua", schema=schema) as writer:
        writer.bind_provenance(provenance)
        writer.write(table.slice(0, 2).to_batches()[0])
        writer.write(table.slice(2))
    summary = writer.summary
    validated = validate_parquet_contract(output)
    assert summary.rows == 4
    assert summary.row_groups == 2
    assert summary.row_group_policy == "one_complete_da_window"
    assert validated["rows"] == 4
    assert validated["row_groups"] == 2
    assert validated["windows"] == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T03:00:00+00:00",
    ]
    parquet = pq.ParquetFile(output)
    file_metadata = {
        **(parquet.schema_arrow.metadata or {}),
        **(parquet.metadata.metadata or {}),
    }
    saved_provenance = json.loads(file_metadata[b"nnja.archive.file_provenance"])
    saved_structures = json.loads(file_metadata[b"nnja.archive.source_structures"])
    assert saved_provenance["effective_packing_ledger_sha256"] == "d" * 64
    assert saved_provenance["code_flag_catalog_sha256"] == "f" * 64
    assert saved_provenance["source_files"][0]["cycle"] == "2024-01-01T00:00:00+00:00"
    assert saved_provenance["source_files"][0]["message_type"] == "NC021023"
    assert saved_structures[0]["sha256"] == structure.fingerprint
    assert validated["source_structure_hashes"] == [structure.fingerprint]
    with pytest.raises(AssertionError, match="expected sensor"):
        validate_parquet_contract(output, expected_sensor="mhs")
    with pytest.raises(AssertionError, match="expected day"):
        validate_parquet_contract(output, expected_day=date(2024, 1, 2))
    with pytest.raises(AssertionError, match="source object identity"):
        validate_parquet_contract(
            output,
            expected_sources=[
                {
                    "object_key": "different.bufr",
                    "cycle": "2024-01-01T00:00:00+00:00",
                    "size_bytes": 12_345,
                }
            ],
        )
    usable = {
        "object_key": "gdas.20240101.t00z.1bamua.tm00.bufr_d",
        "cycle": "2024-01-01T00:00:00+00:00",
        "size_bytes": 12_345,
    }
    identity = {key: unusable[key] for key in usable}
    validate_parquet_contract(output, expected_sources=[usable, identity])
    with pytest.raises(AssertionError, match="source object identity"):
        validate_parquet_contract(output, expected_sources=[usable])
    validate_parquet_contract(output, require_complete_day=True)
    contract = json.loads(file_metadata[b"nnja.archive.contract"])
    assert "embedded_table_sha256" not in contract
    latitude_source = json.loads(
        pq.read_schema(output)
        .field("latitude")
        .metadata[b"nnja.archive.source_descriptors"]
    )
    assert latitude_source[0]["mnemonic"] == "CLAT"
    assert latitude_source[0]["base_scale"] == 2


def test_window_parts_bypass_single_window_and_slice_interleaved_runs() -> None:
    schema = archive_schema(
        "amsua",
        default_test_manifests()["amsua"],
        source_structure=_test_structure("amsua"),
    )
    table = _test_table(schema)
    window_00 = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    window_03 = datetime(2024, 1, 1, 3, tzinfo=timezone.utc)

    single = table.slice(0, 2)
    single_parts = list(_window_parts(single, [window_00]))
    assert single_parts == [(window_00, single)]
    assert single_parts[0][1] is single

    interleaved = table.take(pa.array([0, 2, 1, 3]))
    parts = list(_window_parts(interleaved, [window_00, window_03]))
    assert [window for window, _ in parts] == [
        window_00,
        window_03,
        window_00,
        window_03,
    ]
    assert [part["hpx2048_nest"].to_pylist() for _, part in parts] == [
        [0],
        [2],
        [1],
        [3],
    ]


def test_archive_batches_cannot_revisit_a_flushed_window(tmp_path: Path) -> None:
    schema = archive_schema(
        "amsua",
        default_test_manifests()["amsua"],
        source_structure=_test_structure("amsua"),
    )
    table = _test_table(schema)
    structure = _test_structure("amsua")
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                "amsua.bufr",
                "a" * 64,
                1,
                "2024-01-01T00:00:00+00:00",
                "NC021023",
                message_header_manifest=_header_manifest(structure, 4),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=4,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    output = tmp_path / "out-of-order.parquet"
    writer = SounderArchiveDayWriter(
        output,
        sensor="amsua",
        schema=schema,
    )
    writer.bind_provenance(provenance)
    writer.write(table.slice(2))
    writer.finish_windows([datetime(2024, 1, 1, 3, tzinfo=timezone.utc)])
    with pytest.raises(ValueError, match="already finalized"):
        writer.write(table.slice(0, 2))
    writer.abort()
    assert list(tmp_path.iterdir()) == []


def test_interleaved_batches_form_one_row_group_per_finished_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = archive_schema(
        "amsua",
        default_test_manifests()["amsua"],
        source_structure=_test_structure("amsua"),
    )
    table = _test_table(schema)
    structure = _test_structure("amsua")
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                "amsua.bufr",
                "a" * 64,
                1,
                "2024-01-01T00:00:00+00:00",
                "NC021023",
                message_header_manifest=_header_manifest(structure, 4),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=4,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    output = tmp_path / "interleaved.parquet"
    scratch = tmp_path / "node-local-scratch"
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    writer = SounderArchiveDayWriter(
        output,
        sensor="amsua",
        schema=schema,
    )
    writer.bind_provenance(provenance)
    spool_root = writer._spool_root
    assert spool_root.parent == scratch
    assert writer.temporary.parent == output.parent
    writer.write(table.take(pa.array([0, 2])))
    writer.write(table.take(pa.array([1, 3])))
    writer.finish_windows(
        [
            datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
        ]
    )
    summary = writer.close()

    assert summary.row_groups == 2
    assert not spool_root.exists()
    parquet = pq.ParquetFile(output)
    assert parquet.read_row_group(0, columns=["hpx2048_nest"])[
        "hpx2048_nest"
    ].to_pylist() == [0, 1]
    assert parquet.read_row_group(1, columns=["hpx2048_nest"])[
        "hpx2048_nest"
    ].to_pylist() == [2, 3]


def test_footer_contract_failure_prevents_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = archive_schema(
        "amsub",
        default_test_manifests()["amsub"],
        source_structure=_test_structure("amsub"),
    )
    table = _test_table(schema)
    structure = _test_structure("amsub")
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                "amsub.bufr",
                "e" * 64,
                1,
                "2024-01-01T00:00:00+00:00",
                "NC021024",
                message_header_manifest=_header_manifest(structure, 4),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=4,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    output = tmp_path / "invalid-footer.parquet"

    def reject_footer(_path: Path, **_options) -> None:
        raise AssertionError("invalid footer")

    monkeypatch.setattr(
        "scripts.etl.nnja.sat.archive_writer.validate_parquet_contract",
        reject_footer,
    )

    with pytest.raises(AssertionError, match="invalid footer"):
        write_archive_day(
            table,
            output,
            sensor="amsub",
            schema=schema,
            provenance=provenance,
        )
    assert list(tmp_path.iterdir()) == []


def test_source_population_must_be_fully_accounted(tmp_path: Path) -> None:
    schema = archive_schema(
        "amsub",
        default_test_manifests()["amsub"],
        source_structure=_test_structure("amsub"),
    )
    table = _test_table(schema)
    structure = _test_structure("amsub")
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                "amsub.bufr",
                "e" * 64,
                1,
                "2024-01-01T00:00:00+00:00",
                "NC021024",
                message_header_manifest=_header_manifest(structure, 5),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=5,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    output = tmp_path / "incomplete.parquet"
    with pytest.raises(ValueError, match="do not match the source population"):
        write_archive_day(
            table,
            output,
            sensor="amsub",
            schema=schema,
            provenance=provenance,
        )
    assert list(tmp_path.iterdir()) == []


def test_truncated_source_message_quarantine_accounts_for_population(
    tmp_path: Path,
) -> None:
    schema = archive_schema(
        "amsub",
        default_test_manifests()["amsub"],
        source_structure=_test_structure("amsub"),
    )
    table = _test_table(schema)
    structure = _test_structure("amsub")
    records = [
        {
            "object_key": "amsub.bufr",
            "cycle": "2024-01-01T00:00:00+00:00",
            "source_file_ordinal": 0,
            "source_message_index": 9,
            "subset_count": 1,
            "message_size_bytes": 123,
            "message_sha256": "a" * 64,
            "reason": "Strict BUFR decode read 2 bits beyond the payload",
        }
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                "amsub.bufr",
                "e" * 64,
                1,
                "2024-01-01T00:00:00+00:00",
                "NC021024",
                message_header_manifest=_header_manifest(structure, 5),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=5,
        quarantined_subsets=1,
        quarantine_manifest_sha256=manifest_sha256,
        quarantined_source_messages=tuple(records),
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    output = tmp_path / "quarantined.parquet"

    write_archive_day(
        table,
        output,
        sensor="amsub",
        schema=schema,
        provenance=provenance,
    )
    contract = validate_parquet_contract(output)

    assert contract["rows"] == 4
    assert contract["quarantined_subsets"] == 1
    assert contract["quarantined_source_messages"] == records


def test_missing_required_da_window_is_rejected_before_write(tmp_path: Path) -> None:
    schema = archive_schema(
        "mhs", default_test_manifests()["mhs"], source_structure=_test_structure("mhs")
    )
    table = _test_table(schema)
    index = schema.get_field_index("da_window")
    table = table.set_column(
        index, schema.field(index), pa.nulls(4, schema.field(index).type)
    )
    structure = _test_structure("mhs")
    provenance = FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                "mhs.bufr",
                "b" * 64,
                1,
                "2024-01-01T00:00:00+00:00",
                "NC021027",
                message_header_manifest=_header_manifest(structure, 4),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=4,
        missing_cycles=("06Z", "12Z", "18Z"),
    )
    with pytest.raises(ValueError, match="required archive field 'da_window'"):
        write_archive_day(
            table,
            tmp_path / "invalid.parquet",
            sensor="mhs",
            schema=schema,
            provenance=provenance,
        )


def _single_source_provenance(sha256: str, cycle: str) -> FileProvenance:
    structure = _test_structure("atms")
    return FileProvenance(
        utc_day=date(2024, 1, 1),
        source_files=(
            SourceFile(
                object_key="atms.bufr",
                sha256=sha256,
                size_bytes=1,
                cycle=cycle,
                message_type="NC021000",
                message_header_manifest=_header_manifest(structure, 1),
                source_structure_sha256=structure.fingerprint,
            ),
        ),
        source_structures=(structure,),
        decoder_name="Earth2Bufr",
        decoder_version="test",
        decoder_revision="deadbeef",
        etl_revision="cafebabe",
        effective_packing_ledger_sha256="d" * 64,
        code_flag_catalog_sha256="f" * 64,
        source_subsets_encountered=1,
        missing_cycles=("06Z", "12Z", "18Z"),
    )


def test_provenance_rejects_a_non_hex_source_sha256() -> None:
    _single_source_provenance("c" * 64, "2024-01-01T00:00:00+00:00").validate()
    with pytest.raises(ValueError, match="invalid SHA-256"):
        _single_source_provenance("z" * 64, "2024-01-01T00:00:00+00:00").validate()


def test_provenance_rejects_a_source_cycle_from_another_day() -> None:
    with pytest.raises(ValueError, match="is not on 2024-01-01"):
        _single_source_provenance("c" * 64, "2023-12-31T00:00:00+00:00").validate()
