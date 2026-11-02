# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Ruff does not infer this co-located contract module as a pytest file.
# ruff: noqa: S101
import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from scripts.etl.nnja.sat import archive_etl
from scripts.etl.nnja.sat.archive_etl import (
    Earth2BufrSource,
    SounderPrimaryBatch,
    assign_da_windows,
    compile_archive_layout,
    compute_hpx2048_nest,
    convert_cris_primary_batch,
    convert_sounder_archive_batch,
    convert_sounder_primary_batch,
    iter_open_sounder_archive_batches,
)
from scripts.etl.nnja.sat.archive_schema import (
    SENSOR_SPECS,
    ChannelManifest,
    SourceStructure,
    archive_schema,
    derive_cris_primary_encoding,
)


CRIS_ENCODING = SENSOR_SPECS["cris"].primary_value_encoding
assert CRIS_ENCODING is not None


def _structure(sensor: str) -> SourceStructure:
    return SourceStructure.from_decoder(sensor, _mhs_structure_payload("3-00-001"))


def _mhs_structure_payload(descriptor: str) -> dict:
    return {
        "format_version": 1,
        "embedded_table_sha256": "a" * 64,
        "observation_descriptors": [descriptor],
        "active_sequences": [],
        "active_elements": [{"descriptor": "0-01-001", "mnemonic": "TEST"}],
        "effective_occurrences": [
            {
                "descriptor": "0-01-001",
                "mnemonic": "TEST",
                "decoder_path": "TEST",
            }
        ],
        "decoded_physical_schema": [
            {"position": 0, "name": "TEST", "arrow_type": "double"}
        ],
    }


def test_open_source_derives_one_structure_per_descriptor_template() -> None:
    descriptors = {
        4: ("3-00-001",),
        5: ("3-00-002",),
        6: ("3-00-001",),
    }

    class NativeSource:
        def __init__(self) -> None:
            self.structure_calls = []
            self.info = SimpleNamespace(
                data_messages=[
                    SimpleNamespace(
                        source_message_index=index,
                        edition=4,
                        master_table=0,
                        center=7,
                        sub_center=0,
                        update_sequence=0,
                        optional_section_present=False,
                        year=2024,
                        month=1,
                        day=1,
                        hour=0,
                        minute=index,
                        second=0,
                        data_category=21,
                        international_sub_category=0,
                        local_sub_category=0,
                        master_table_version=41,
                        local_table_version=0,
                        subsets=1,
                        compressed=True,
                        section3_flags=0x40,
                        descriptors=template,
                        table_epoch=0,
                    )
                    for index, template in descriptors.items()
                ]
            )

        def source_structure(self, source_message_index):
            self.structure_calls.append(source_message_index)
            return _mhs_structure_payload(descriptors[source_message_index][0])

    native = NativeSource()
    bindings = Earth2BufrSource(native).source_structure_bindings("mhs")

    assert native.structure_calls == [4, 5]
    assert bindings[4] is bindings[6]
    assert bindings[4].observation_descriptors == ("3-00-001",)
    assert bindings[5].observation_descriptors == ("3-00-002",)


def test_truncated_messages_become_quarantine_records(tmp_path) -> None:
    path = tmp_path / "source.bufr"
    path.write_bytes(b"BUFR payloadTRUNCATED")
    skipped = SimpleNamespace(offset=12, size_bytes=9, subsets=7)

    class Batches:
        decode_failures = (SimpleNamespace(source_message_index=1, reason="truncated"),)

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    native = SimpleNamespace(
        path=path,
        info=SimpleNamespace(messages=(None, skipped)),
        iter_batches=lambda **_options: Batches(),
    )
    source = Earth2BufrSource(native)

    assert source.decode_failures() == ()
    assert list(source.iter_batches(sensor="mhs")) == []
    assert source.decode_failures() == (
        {
            "source_message_index": 1,
            "subset_count": 7,
            "message_size_bytes": 9,
            "message_sha256": hashlib.sha256(b"TRUNCATED").hexdigest(),
            "reason": "truncated",
        },
    )


def test_open_iterator_selects_and_caches_layout_per_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = SourceStructure.from_decoder("mhs", _mhs_structure_payload("3-00-001"))
    second = SourceStructure.from_decoder("mhs", _mhs_structure_payload("3-00-002"))
    batches = [
        pa.record_batch(
            [pa.array([index], type=pa.uint64())],
            names=["_source_message_index"],
        )
        for index in (4, 5, 6)
    ]

    class OpenedSource:
        message_headers = tuple(
            SimpleNamespace(
                index=index,
                reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
            for index in (4, 5, 6)
        )

        def iter_batches(self, **_options):
            return iter(batches)

    validated = []
    converted = []
    manifest = ChannelManifest.from_ids("mhs", range(1, 6), release="test")

    def fake_validate(_batch, *, sensor, source_structure):
        assert sensor == "mhs"
        validated.append(source_structure.fingerprint)

    def fake_convert(_batch, **options):
        structure = options["source_structure"]
        layout = options["compiled_layout"]
        converted.append(
            (
                structure.fingerprint,
                None if layout is None else layout.source_structure.fingerprint,
            )
        )
        if layout is None:
            layout = compile_archive_layout(
                sensor="mhs",
                channel_manifest=manifest,
                source_structure=structure,
            )
        return SimpleNamespace(layout=layout)

    monkeypatch.setattr(archive_etl, "validate_source_structure_binding", fake_validate)
    monkeypatch.setattr(archive_etl, "convert_sounder_archive_batch", fake_convert)

    results = list(
        iter_open_sounder_archive_batches(
            OpenedSource(),
            sensor="mhs",
            cycle=datetime(2024, 1, 1, tzinfo=timezone.utc),
            source_file_ordinal=0,
            release="test",
            structures={
                4: first,
                5: second,
                6: first,
            },
        )
    )

    assert len(results) == 3
    assert validated == [first.fingerprint, second.fingerprint]
    assert converted == [
        (first.fingerprint, None),
        (second.fingerprint, None),
        (first.fingerprint, first.fingerprint),
    ]


def test_structural_occurrence_map_does_not_require_hashable_schema() -> None:
    """The occurrence map accepts unhashable schemas, so it is not cached on them."""
    schema = pa.schema(
        [
            pa.field("SIID", pa.float64(), metadata={"note": "sounder"}),
            pa.field("SIID", pa.float64(), metadata={"note": "imager"}),
        ]
    )
    mapping = dict(archive_etl._structural_top_level_occurrence_map(schema, "SIID"))
    assert mapping == {0: 0, 1: 1}


def test_cris_siid_occurrences_bind_sounder_and_imager() -> None:
    batch = pa.record_batch(
        [
            pa.array([620.0, 620.0]),
            pa.array([616.0, 616.0]),
        ],
        schema=pa.schema(
            [pa.field("SIID", pa.float64()), pa.field("SIID", pa.float64())],
            metadata={b"earth2bufr.fixed_replication": b"structural"},
        ),
    )

    primary = archive_etl._common_scalar_array(
        batch,
        sensor="cris",
        logical_name="instrument_id",
        unsigned=True,
    )
    secondary_spec = next(
        field
        for field in SENSOR_SPECS["cris"].fields
        if field.name == "secondary_instrument_id"
    )
    secondary = archive_etl._simple_secondary_array(batch, secondary_spec)

    assert primary.to_pylist() == [620, 620]
    assert secondary.to_pylist() == [616, 616]


def test_airs_common_geometry_uses_primary_instrument_occurrences() -> None:
    batch = pa.record_batch(
        [
            pa.array([10.0]),
            pa.array([20.0]),
            pa.array([30.0]),
            pa.array([40.0]),
            pa.array([50.0]),
            pa.array([60.0]),
            pa.array([1.0]),
            pa.array([2.0]),
            pa.array([3.0]),
        ],
        names=[
            "SAZA",
            "SAZA",
            "SAZA",
            "BEARAZ",
            "BEARAZ",
            "BEARAZ",
            "FOVN",
            "FOVN",
            "FOVN",
        ],
    )

    assert archive_etl._common_scalar_array(
        batch, sensor="airs", logical_name="satellite_zenith_angle"
    ).to_pylist() == [10.0]
    assert archive_etl._common_scalar_array(
        batch, sensor="airs", logical_name="satellite_azimuth_angle"
    ).to_pylist() == [40.0]
    assert archive_etl._common_scalar_array(
        batch,
        sensor="airs",
        logical_name="field_of_view",
        unsigned=True,
    ).to_pylist() == [1]


def _cris_source_structure(**packing_overrides: str | int) -> SourceStructure:
    packing = {
        "descriptor": "0-14-044",
        "effective_scale": 7,
        "effective_reference": -100_000,
        "effective_width": 22,
        "source_missing_code": 4_194_303,
        "units": "W M**-2 SR**-1 CM",
        **packing_overrides,
    }
    occurrences = (
        {
            "decoder_path": "DRF16BIT.SRAD",
            "mnemonic": "SRAD",
            **packing,
        },
    )
    return SourceStructure(
        sensor="cris",
        embedded_table_sha256="a" * 64,
        observation_descriptors=("3-10-198",),
        active_sequences=({"descriptor": "3-10-198"},),
        active_elements=({"descriptor": "0-14-044", "mnemonic": "SRAD"},),
        effective_occurrences=occurrences,
        decoded_physical_schema=({"position": 56, "name": "DRF16BIT"},),
    )


def _raw_cris_batch(
    *,
    vary_second_axis: bool = False,
    native_codes: bool = True,
    metadata_overrides: dict[bytes, bytes] | None = None,
) -> pa.RecordBatch:
    channel_ids = [19 + 5 * position for position in range(431)]
    rows: list[list[dict[str, float | int | None]]] = []
    for row in range(2):
        this_axis = channel_ids.copy()
        if row == 1 and vary_second_axis:
            this_axis[-1] += 1
        rows.append(
            [
                {
                    "CHNM": float(channel_id),
                    "SRAD": None
                    if row == 1 and position == 3
                    else (
                        100_000 + row * 500 + position
                        if native_codes
                        else float(100_000 + row * 500 + position)
                    ),
                }
                for position, channel_id in enumerate(this_axis)
            ]
        )
    if native_codes:
        metadata = {
            b"representation": b"bufr_raw_code",
            b"descriptor": b"0-14-044",
            b"scale": b"7",
            b"reference": b"-100000",
            b"bit_width": b"22",
            b"missing_code": b"4194303",
            b"source_units": b"W M**-2 SR**-1 CM",
            b"physical_formula": b"(code - 100000) * 1e-7",
            **(metadata_overrides or {}),
        }
        srad = pa.field("SRAD", pa.int32(), metadata=metadata)
    else:
        srad = pa.field("SRAD", pa.float64())
    repeated_type = pa.list_(pa.struct([pa.field("CHNM", pa.float64()), srad]))
    return pa.record_batch(
        [
            pa.array(rows, type=repeated_type),
            pa.array([21.0, 21.0]),
            pa.array([7, 7], type=pa.uint64()),
            pa.array([0, 1], type=pa.uint32()),
        ],
        names=[
            "DRF16BIT",
            "_data_category",
            "_source_message_index",
            "_source_subset_index",
        ],
    )


def _raw_mhs_batch() -> pa.RecordBatch:
    values: list[pa.Array] = []
    names: list[str] = []

    def add(name: str, first: float, second: float) -> None:
        names.append(name)
        values.append(pa.array([first, second], type=pa.float64()))

    for name, pair in (
        ("SAID", (4, 4)),
        ("SIID", (203, 203)),
        ("YEAR", (2024, 2024)),
        ("MNTH", (1, 1)),
        ("DAYS", (1, 1)),
        ("HOUR", (0, 3)),
        ("MINU", (0, 12)),
        ("SECO", (0, 0)),
        ("CLAT", (0, 45)),
        ("CLON", (0, 10)),
        ("SAZA", (1, 2)),
        ("BEARAZ", (3, 4)),
        ("SOZA", (5, 6)),
        ("SOLAZI", (7, 8)),
        ("FOVN", (1, 2)),
        ("LSQL", (0, 1)),
        ("HOLS", (100, 200)),
        ("HMSL", (800_000, 800_000)),
    ):
        add(name, *pair)
    for channel in range(1, 6):
        add("CHNM", channel, channel)
        add("TMBR", 250 + channel, 251 + channel)
        add("CSTC", 0.1 * channel, 0.2 * channel)
    names.extend(["_source_message_index", "_source_subset_index", "_data_category"])
    values.extend(
        [
            pa.array([7, 7], type=pa.uint64()),
            pa.array([0, 1], type=pa.uint32()),
            pa.array([21, 21], type=pa.uint8()),
        ]
    )
    return pa.record_batch(values, names=names)


def _raw_iasi_batch(*, nonzero_unknown_scale: bool = False) -> pa.RecordBatch:
    starts = [16, 150, 300, 566, 1536, 2910, 3322, 4257, 5833, 6982]
    stops = [148, 299, 559, 1532, 2907, 3312, 4234, 5817, 6977, 8078]
    channel_ids = [
        int(channel)
        for start, stop, count in zip(
            starts,
            stops,
            [61, 61, 61, 61, 61, 61, 61, 61, 61, 67],
            strict=True,
        )
        for channel in np.linspace(start, stop, count, dtype=np.int64)
    ]
    source_values = [0.0] * 61 + [1.0] * (len(channel_ids) - 61)
    if nonzero_unknown_scale:
        source_values[0] = 1.0
    primary_type = pa.list_(
        pa.struct([pa.field("CHNM", pa.float64()), pa.field("SCRA", pa.float64())])
    )
    arrays: list[pa.Array] = []
    names: list[str] = []
    for start, stop, scale in zip(
        starts,
        stops,
        [None, 8, 8, 8, 8, 9, 9, 9, 9, 10],
        strict=True,
    ):
        names.extend(("STCH", "ENCH", "CHSF"))
        arrays.extend(
            (
                pa.array([start], type=pa.float64()),
                pa.array([stop], type=pa.float64()),
                pa.array([scale], type=pa.float64()),
            )
        )
    names.append("DRF16BIT")
    arrays.append(
        pa.array(
            [
                [
                    {"CHNM": float(channel), "SCRA": value}
                    for channel, value in zip(channel_ids, source_values, strict=True)
                ]
            ],
            type=primary_type,
        )
    )
    return pa.record_batch(arrays, names=names)


def test_cris_primary_batch_is_wide() -> None:
    converted = convert_cris_primary_batch(
        _raw_cris_batch(),
        source_structure=_cris_source_structure(),
        release="synthetic-c0431",
    )

    assert converted.table.num_rows == 2
    assert converted.table.num_columns == 431
    assert converted.channel_manifest.channels[0].source_channel_id == 19
    assert converted.channel_manifest.channels[-1].source_channel_id == 2169
    assert converted.table["spectral_radiance_code__ch_00019"].to_pylist() == [
        100_000,
        100_500,
    ]
    assert converted.table["spectral_radiance_code__ch_00034"].to_pylist() == [
        100_003,
        None,
    ]


def test_cris_primary_batch_requires_decoder_raw_codes() -> None:
    raw = _raw_cris_batch(native_codes=False)
    with pytest.raises(TypeError, match="int32 raw codes"):
        convert_cris_primary_batch(
            raw, source_structure=_cris_source_structure(), release="synthetic-c0431"
        )
    with pytest.raises(TypeError, match="not int32"):
        archive_etl._decoder_arrow_shape(
            raw.schema.field("DRF16BIT"), {"DRF16BIT.SRAD": CRIS_ENCODING}
        )


def test_cris_primary_batch_rejects_raw_code_metadata_drift() -> None:
    with pytest.raises(ValueError, match="metadata does not match source structure"):
        convert_cris_primary_batch(
            _raw_cris_batch(
                native_codes=True,
                metadata_overrides={b"scale": b"6"},
            ),
            source_structure=_cris_source_structure(),
            release="synthetic-c0431",
        )


def test_decoder_shape_normalizes_validated_raw_code_field() -> None:
    decoder_field = _raw_cris_batch(native_codes=True).schema.field("DRF16BIT")

    assert archive_etl._decoder_arrow_shape(
        decoder_field,
        {"DRF16BIT.SRAD": CRIS_ENCODING},
    ) == (
        "list<struct>",
        (("CHNM", ("float64",)), ("SRAD", ("float64",))),
    )


def test_cris_primary_batch_requires_one_invariant_channel_axis() -> None:
    with pytest.raises(ValueError, match="channel axis varies"):
        convert_cris_primary_batch(
            _raw_cris_batch(vary_second_axis=True),
            source_structure=_cris_source_structure(),
            release="synthetic-c0431",
        )


def test_cris_primary_packing_is_derived_from_the_srad_member() -> None:
    structure = _cris_source_structure(
        descriptor="0-14-199",
        effective_scale=6,
        effective_reference=-50_000,
        effective_width=21,
        source_missing_code=2_097_151,
    )
    encoding = derive_cris_primary_encoding(structure)
    assert encoding.descriptor == "0-14-199"
    assert encoding.scale == 6
    assert encoding.reference == -50_000
    assert encoding.bit_width == 21
    assert encoding.source_missing_code == 2_097_151


def test_cris_primary_packing_rejects_ambiguous_srad_or_unknown_units() -> None:
    structure = _cris_source_structure()
    for occurrences in ((), structure.effective_occurrences * 2):
        with pytest.raises(ValueError, match="one DRF16BIT.SRAD member"):
            derive_cris_primary_encoding(
                replace(
                    structure,
                    effective_occurrences=(
                        *occurrences,
                        {"mnemonic": "CHNM", "decoder_path": "DRF16BIT.CHNM"},
                    ),
                )
            )
    with pytest.raises(ValueError, match="units"):
        derive_cris_primary_encoding(_cris_source_structure(units="mW/m2/sr/cm-1"))


def test_all_sounder_dispatch_includes_cris() -> None:
    converted = convert_sounder_primary_batch(
        _raw_cris_batch(),
        sensor="cris",
        source_structure=_cris_source_structure(),
        release="synthetic-c0431",
    )

    assert isinstance(converted, SounderPrimaryBatch)
    assert converted.sensor == "cris"


def test_iasi_null_scale_zero_fill_becomes_null_physical_radiance() -> None:
    converted = convert_sounder_primary_batch(
        _raw_iasi_batch(),
        sensor="iasi",
        release="synthetic-iasi",
        source_structure=_structure("iasi"),
    )

    assert converted.table["spectral_radiance__ch_00016"].to_pylist() == [None]
    assert converted.table["spectral_radiance__ch_00150"].to_pylist() == [
        pytest.approx(0.001)
    ]


def test_iasi_null_scale_rejects_nonzero_source_value() -> None:
    with pytest.raises(ValueError, match="scale is null for a nonzero SCRA"):
        convert_sounder_primary_batch(
            _raw_iasi_batch(nonzero_unknown_scale=True),
            sensor="iasi",
            release="synthetic-iasi",
            source_structure=_structure("iasi"),
        )


def test_hpx2048_masks_invalid_source_geolocation() -> None:
    pixels = compute_hpx2048_nest(
        pa.array([0.0, 10.0, -180.0]),
        pa.array([0.0, 45.0, -90.0]),
    )
    assert pixels.type == pa.uint32()
    assert pixels.null_count == 0
    assert pixels.to_pylist() == [19_922_944, 2_824_601, 41_943_040]

    masked = compute_hpx2048_nest(
        pa.array([0.0, None, 10.0]),
        pa.array([0.0, 45.0, 100.0]),
    )
    assert masked.to_pylist() == [19_922_944, None, None]


def test_complete_mhs_batch_matches_generated_archive_contract() -> None:
    cycle = datetime(2024, 1, 1, tzinfo=timezone.utc)
    converted = convert_sounder_archive_batch(
        _raw_mhs_batch(),
        sensor="mhs",
        cycle=cycle,
        source_file_ordinal=2,
        message_reference_times={7: cycle},
        release="synthetic-mhs",
        source_structure=_structure("mhs"),
    )
    expected = archive_schema(
        "mhs", converted.channel_manifest, source_structure=_structure("mhs")
    )

    assert converted.table.schema.equals(expected, check_metadata=True)
    assert converted.table.schema.equals(converted.layout.schema, check_metadata=True)
    assert converted.table.num_columns == 27
    assert converted.table["hpx2048_nest"].to_pylist() == [19_922_944, 2_824_601]
    assert converted.table["da_window"].to_pylist() == [
        cycle,
        cycle + timedelta(hours=3),
    ]
    assert converted.table["time_utc"].to_pylist()[1] == cycle + timedelta(
        hours=3, minutes=12
    )
    assert converted.table["cold_space_temperature_correction"].to_pylist()[0][
        -1
    ] == pytest.approx(0.5)


def test_complete_layout_is_compiled_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = datetime(2024, 1, 1, tzinfo=timezone.utc)
    schema_calls = 0
    original_archive_schema = archive_etl.archive_schema

    def counted_archive_schema(*args, **kwargs):
        nonlocal schema_calls
        schema_calls += 1
        return original_archive_schema(*args, **kwargs)

    monkeypatch.setattr(archive_etl, "archive_schema", counted_archive_schema)
    first = convert_sounder_archive_batch(
        _raw_mhs_batch(),
        sensor="mhs",
        cycle=cycle,
        source_file_ordinal=0,
        message_reference_times={7: cycle},
        release="synthetic-mhs",
        source_structure=_structure("mhs"),
    )
    second = convert_sounder_archive_batch(
        _raw_mhs_batch(),
        sensor="mhs",
        cycle=cycle,
        source_file_ordinal=0,
        message_reference_times={7: cycle},
        release="synthetic-mhs",
        source_structure=_structure("mhs"),
        compiled_layout=first.layout,
    )

    assert schema_calls == 1
    assert second.layout is first.layout
    assert second.channel_manifest is first.channel_manifest
    assert second.table.schema.equals(first.table.schema, check_metadata=True)


def test_compiled_layout_rejects_source_structure_drift() -> None:
    cycle = datetime(2024, 1, 1, tzinfo=timezone.utc)
    structure = _cris_source_structure()
    primary = convert_cris_primary_batch(
        _raw_cris_batch(),
        release="synthetic-c0431",
        source_structure=structure,
    )
    layout = compile_archive_layout(
        sensor="cris",
        channel_manifest=primary.channel_manifest,
        source_structure=structure,
    )
    changed = replace(structure, embedded_table_sha256="b" * 64)

    with pytest.raises(ValueError, match="source structure changed"):
        convert_sounder_archive_batch(
            _raw_cris_batch(),
            sensor="cris",
            cycle=cycle,
            source_file_ordinal=0,
            message_reference_times={7: cycle},
            release="synthetic-c0431",
            source_structure=changed,
            compiled_layout=layout,
        )


def test_cycle_assignment_matches_original_healda_two_window_rule() -> None:
    cycle = datetime(2024, 1, 1, 6, tzinfo=timezone.utc)
    times = pa.array(
        [
            cycle - timedelta(hours=3, minutes=6),
            cycle,
            cycle + timedelta(microseconds=1),
            cycle + timedelta(hours=3, minutes=12),
        ],
        type=pa.timestamp("ns", tz="UTC"),
    )
    windows = assign_da_windows(times, cycle)
    assert windows.to_pylist() == [
        cycle,
        cycle,
        cycle + timedelta(hours=3),
        cycle + timedelta(hours=3),
    ]


def test_missing_nested_fields_are_arrow_nulls() -> None:
    from scripts.etl.nnja.sat.archive_etl import _null_array

    nested = pa.struct([("a", pa.list_(pa.float32(), 3)), ("b", pa.int16())])
    array = _null_array(4, nested)
    assert array.null_count == 4
    assert array.field("a").null_count == 4
    fixed = _null_array(2, pa.list_(pa.float32(), 5))
    assert fixed.null_count == 2
    assert fixed.values.null_count == 10
