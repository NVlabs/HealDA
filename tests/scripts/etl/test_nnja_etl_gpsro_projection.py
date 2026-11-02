# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from healda.observations import schema

from scripts.etl.nnja.conv.nnja_prepbufr_to_parquet import CycleParquetBuilder
from scripts.etl.nnja.gpsro.gpsro_projection import (
    BMISS,
    empty_gpsro_table,
    project_gpsro_batch,
)

FREQ_NAMES = ["MEFR", "IMPP", "BNDA", "FOST", "BNDA", "FOST"]
REFR_NAMES = ["HEIT", "ARFR", "FOST", "ARFR", "FOST", "PCCF"]


def freq(mefr, impact, bending, error):
    return [mefr, impact, bending, 13.0, error, 13.0]


def list_of_structs(rows, names, child=None):
    """rows: per-row list of entries; each entry a list of field values."""
    offsets = np.concatenate(([0], np.cumsum([len(r) for r in rows]))).astype(np.int32)
    entries = [e for r in rows for e in r]
    children = [
        pa.array([e[j] for e in entries], type=pa.float64()) for j in range(len(names))
    ]
    if child is not None:
        children.insert(child[0], child[1])
        names = names[: child[0]] + [child[2]] + names[child[0] :]
    struct = pa.StructArray.from_arrays(children, names=names)
    return pa.ListArray.from_arrays(pa.array(offsets), struct)


def make_batch(levels_on=0):
    # occultation `levels_on` has two levels, the other none. Level 0 has L1, L2
    # and a corrected entry; level 1 has no corrected entry, so its last entry is
    # primary.
    freqs = list_of_structs(
        [
            [
                freq(1.5e9, 6.4e6, 0.010, 0.001),
                freq(0.0, 6.40001e6, 0.011, 0.002),
                freq(1.2e9, 6.4e6, 0.012, 0.003),
            ],
            [freq(1.5e9, 6.41e6, 0.020, 0.004), freq(1.2e9, 6.41e6, 0.021, BMISS)],
        ],
        FREQ_NAMES,
    )
    two_levels = [[10.0, 20.0, 90.0, 100.0], [10.5, 20.5, 91.0, 99.0]]
    levels = list_of_structs(
        [two_levels, []] if levels_on == 0 else [[], two_levels],
        ["CLATH", "CLONH", "BEARAZ", "PCCF"],
        child=(3, freqs, "DRF8BIT"),
    )
    profile = list_of_structs(
        [
            [
                [h, 300.0 * np.exp(-h / 7000.0), 13, 1, 13, 100]
                for h in range(0, 40000, 2000)
            ]
        ]
        * 2,
        REFR_NAMES,
    )
    scalars = {
        "SAID": [44.0, None],
        "SIID": [1.0, 1.0],
        "OGCE": [60.0, 60.0],
        "PTID": [5.0, 7.0],
        "QFRO": [0.0, 0.0],
        "CLATH": [10.0, 0.0],
        "CLONH": [20.0, 0.0],
        "ELRC": [6.38e6, 6.38e6],
        "GEODU": [10.0, 10.0],
        "YEAR": [2025.0, 2025.0],
        "MNTH": [1.0, 1.0],
        "DAYS": [2.0, 2.0],
        "HOUR": [0.0, 0.0],
        "MINU": [30.0, 0.0],
        "SECO": [0.0, 0.0],
    }
    arrays = [pa.array(v, type=pa.float64()) for v in scalars.values()]
    names = list(scalars)
    # The decoder repeats top-level names; the first column of a name is used.
    arrays.append(pa.array([-50.0, -50.0]))
    names.append("CLATH")
    arrays += [levels, profile]
    names += ["DRF16BIT", "DRF16BIT_2"]
    return pa.RecordBatch.from_arrays(arrays, names=names)


def test_project_gpsro_batch_picks_frequencies_and_keeps_levels():
    table = project_gpsro_batch(
        make_batch(), source_file="f", cycle_date="20250102", cycle_hour="00"
    )
    rows = table.to_pylist()

    assert table.num_rows == 2
    assert [r["level_index"] for r in rows] == [0, 1]
    assert {r["occultation_id"] for r in rows} == {"44|2025010200|5|0"}
    assert [r["occ_latitude"] for r in rows] == [10.0, 10.0]
    assert rows[0]["impact_parameter"] == 6.40001e6
    assert np.float32(rows[0]["bending_angle"]) == np.float32(0.011)
    assert np.float32(rows[0]["bending_angle_error"]) == np.float32(0.002)
    assert np.float32(rows[0]["bending_angle_l1"]) == np.float32(0.010)
    assert np.float32(rows[0]["bending_angle_l2"]) == np.float32(0.012)
    # no corrected entry: the last entry (L2) is primary; BMISS becomes null
    assert np.float32(rows[1]["bending_angle"]) == np.float32(0.021)
    assert rows[1]["bending_angle_error"] is None
    assert rows[1]["impact_height_m"] == 6.41e6 - 6.38e6
    assert all(r["dry_pressure_hpa"] > 0 for r in rows)
    assert str(rows[0]["obs_time"]) == "2025-01-02 00:30:00+00:00"


def test_occultation_id_spells_missing_satellite_as_none():
    table = project_gpsro_batch(
        make_batch(levels_on=1), source_file="f", cycle_date="20250102", cycle_hour="00"
    )
    assert table.column("occultation_id").to_pylist() == ["None|2025010200|7|1"] * 2
    assert table.column("satellite_id").to_pylist() == [None, None]


def test_written_cycles_have_the_reader_types(tmp_path):
    # A null satellite_id leaves pandas a float64 column.
    table = project_gpsro_batch(
        make_batch(levels_on=1), source_file="f", cycle_date="20250102", cycle_hour="00"
    )
    base = schema.table_schema("gpsro")
    full = CycleParquetBuilder(string_type=pa.large_string(), base_schema=base)
    full.add_table(table)
    full.write(tmp_path / "full.parquet")
    empty = CycleParquetBuilder(string_type=pa.large_string(), base_schema=base)
    empty.write(tmp_path / "empty.parquet", empty_table=empty_gpsro_table())

    for name in ("full", "empty"):
        written = pq.read_table(tmp_path / f"{name}.parquet")
        for field in base:
            assert written.schema.field(field.name).type == field.type, field.name
            assert written.schema.field(field.name).nullable == field.nullable
        schema.validate(written, "gpsro")
    assert pq.read_table(tmp_path / "full.parquet")["satellite_id"].to_pylist() == [
        None,
        None,
    ]
    assert set(empty_gpsro_table().column_names) <= set(
        pq.read_schema(tmp_path / "empty.parquet").names
    )
