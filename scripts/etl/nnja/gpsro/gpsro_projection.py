# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project NNJA GPSRO (GPS radio-occultation) BUFR into per-level rows.

GPSRO files decode with earth2bufr.read_bufr (they embed NCEP DX tables). The
decoded category-3 batch has, per occultation:

  * scalars: SAID (receiver sat), SIID (instrument), OGCE (originating centre),
    PTID (transmitter), CLATH/CLONH (occultation lat/lon), QFRO (RO quality
    flags), ELRC (Earth local radius of curvature), GEODU (geoid undulation),
    YEAR..SECO (absolute time).
  * DRF16BIT: list of vertical LEVELS. Each level has per-level CLATH/CLONH/
    BEARAZ/PCCF and DRF8BIT = a list of FREQUENCY entries (MEFR = 1.5 GHz L1,
    1.2 GHz L2, 0 = ionospheric-corrected), each with IMPP (impact parameter),
    BNDA (bending angle), FOST (observation error).
  * DRF16BIT_2: refractivity profile HEIT + ARFR (N-units). Used to derive
    source-only dry hydrostatic pressure and true (refraction-corrected) height
    at each bending-angle impact parameter.
  * DRF16BIT_3: provider 1D-Var T/q/p, dropped (background-influenced).

The primary observation is the DRF16BIT bending angle. ``impact_height_m`` (IMPP -
ELRC) is the GSI-diagnostic-convention height for a bending-angle observation
(verified byte-identical to GSI's own diagnostic ``Height`` field) and is the
recommended height coordinate for the model. It is a ray-geometric quantity,
not the true tangent-point height, so it is *not* used for pressure lookups:
``heit_at_impact_m`` (HEIT interpolated to IMPP via Bouguer's rule, i.e. the
same refractive-radius mapping used for ``dry_pressure_at_impact_hpa``) is the
refraction-corrected height, used only as input to
``standard_atmosphere_pressure_hpa``. ``dry_pressure_hpa`` is a derived
coordinate (dry-air assumption; biased high in moist lower troposphere), not
GSI physical pressure. ``blended_pressure_5km_hpa`` mixes
``standard_atmosphere_pressure_hpa`` (weight 0.8) with ``dry_pressure_hpa``
(weight 0.2) below ``BLEND_HEIGHT_M`` and uses pure ``dry_pressure_hpa`` above
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from healda.observations.preprocessing.gpsro_pressure import (
    derive_level_coordinates_batch,
)

BMISS = 1.0e10
GPSRO_CODE = 3
DERIVED_COLUMNS = (
    "impact_height_m",
    "heit_at_impact_m",
    "dry_pressure_hpa",
    "standard_atmosphere_pressure_hpa",
    "blended_pressure_5km_hpa",
)


def _floats(array: pa.Array) -> np.ndarray:
    """float64 values, NaN where null, non-finite or BUFR-missing (>= BMISS)."""
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
    usable = np.isfinite(values) & (np.abs(values) < BMISS)
    return np.where(usable, values, np.nan)


def _field_index(struct_type: pa.StructType, name: str, occurrence: int = 0) -> int:
    """Index of the n-th field called ``name``; RO structs repeat names (BNDA, FOST)."""
    matches = [
        i for i in range(struct_type.num_fields) if struct_type.field(i).name == name
    ]
    return matches[occurrence]


def _entries(column: pa.Array) -> tuple[pa.StructArray, np.ndarray]:
    """The entries of a list<struct> column and the entry count per row (0 if null)."""
    counts = pc.fill_null(pc.list_value_length(column), 0).to_numpy()
    return pc.list_flatten(column), counts.astype(np.int64)


def _starts(counts: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(counts)))


def _pick(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    out = np.full(len(index), np.nan)
    found = index >= 0
    out[found] = values[index[found]]
    return out


def _float_array(values: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    dtype = np.float32 if arrow_type == pa.float32() else np.float64
    return pa.array(values.astype(dtype), type=arrow_type, mask=np.isnan(values))


def _int_array(values: np.ndarray) -> pa.Array:
    valid = ~np.isnan(values)
    return pa.array(
        np.where(valid, values, 0).astype(np.int64), type=pa.int64(), mask=~valid
    )


def project_gpsro_batch(
    batch: pa.RecordBatch,
    *,
    source_file: str,
    cycle_date: str,
    cycle_hour: str,
) -> pa.Table:
    """One row per (occultation, DRF16BIT level) of a decoded GPS-RO batch."""
    names = batch.schema.names

    # RO batches repeat top-level names; the first column of each name is used.
    def raw(name: str) -> np.ndarray:
        return batch.column(names.index(name)).to_numpy(zero_copy_only=False)

    def scalar(name: str) -> np.ndarray:
        return _floats(batch.column(names.index(name)))

    said, ptid = scalar("SAID"), scalar("PTID")
    olat, elrc, geodu = scalar("CLATH"), scalar("ELRC"), scalar("GEODU")
    obs_time = pd.to_datetime(
        pd.DataFrame(
            {
                "year": raw("YEAR"),
                "month": raw("MNTH"),
                "day": raw("DAYS"),
                "hour": raw("HOUR"),
                "minute": raw("MINU"),
                "second": raw("SECO"),
            }
        ),
        errors="coerce",
        utc=True,
    )

    levels, levels_per_occ = _entries(batch.column(names.index("DRF16BIT")))
    level_type = levels.type
    freqs, freqs_per_level = _entries(levels.field(_field_index(level_type, "DRF8BIT")))
    n_levels = len(levels)

    # Frequency entries: MEFR 0 = ionosphere-corrected, ~1.5 GHz = L1, ~1.2 GHz = L2.
    # The last matching entry of a level wins; without a corrected entry the
    # level's last entry is the primary one.
    freq_type = freqs.type
    mefr = _floats(freqs.field(_field_index(freq_type, "MEFR")))
    freq_level = np.repeat(np.arange(n_levels), freqs_per_level)
    freq_index = np.arange(len(freqs))

    def last_entry(selected: np.ndarray) -> np.ndarray:
        found = np.full(n_levels, -1, dtype=np.int64)
        np.maximum.at(found, freq_level[selected], freq_index[selected])
        return found

    corrected = last_entry(mefr == 0.0)
    l1 = last_entry(np.abs(mefr - 1.5e9) < 1e8)
    l2 = last_entry(np.abs(mefr - 1.2e9) < 1e8)
    last = np.where(freqs_per_level > 0, _starts(freqs_per_level)[1:] - 1, -1)
    primary = np.where(corrected >= 0, corrected, last)

    def freq_field(name: str, occurrence: int = 0) -> np.ndarray:
        return _floats(freqs.field(_field_index(freq_type, name, occurrence)))

    bnda = freq_field("BNDA")
    impact = _pick(freq_field("IMPP"), primary)

    # DRF16BIT_2: refractivity profile; the first ARFR is the observation,
    # the second the paired uncertainty.
    refr, refr_per_occ = _entries(batch.column(names.index("DRF16BIT_2")))
    height = _floats(refr.field(_field_index(refr.type, "HEIT")))
    refractivity = _floats(refr.field(_field_index(refr.type, "ARFR")))

    level_starts, refr_starts = _starts(levels_per_occ), _starts(refr_per_occ)
    derived = derive_level_coordinates_batch(
        impact,
        level_starts,
        height,
        refractivity,
        refr_starts,
        latitude_degrees=olat,
        earth_radius_of_curvature_m=elrc,
        geoid_undulation_m=np.where(np.isnan(geodu), 0.0, geodu),
    )

    occ = np.repeat(np.arange(batch.num_rows), levels_per_occ)
    ids = pa.array(
        [
            f"{'None' if np.isnan(s) else int(s)}|{cycle_date}{cycle_hour}|"
            f"{'None' if np.isnan(p) else int(p)}|{i}"
            for i, (s, p) in enumerate(zip(said, ptid))
        ],
        type=pa.string(),
    )

    def level_field(name: str) -> np.ndarray:
        return _floats(levels.field(_field_index(level_type, name)))

    def constant(value: str) -> pa.Array:
        return pa.array([value], type=pa.string()).take(np.zeros(n_levels, np.int64))

    f32, f64 = pa.float32(), pa.float64()
    return pa.table(
        {
            "occultation_id": ids.take(occ),
            "satellite_id": _int_array(said[occ]),
            "instrument_id": _int_array(scalar("SIID")[occ]),
            "originating_centre": _int_array(scalar("OGCE")[occ]),
            "transmitter_id": _int_array(ptid[occ]),
            "level_index": pa.array(
                np.arange(n_levels) - level_starts[:-1][occ], type=pa.int32()
            ),
            "occ_latitude": _float_array(olat[occ], f64),
            "occ_longitude": _float_array(scalar("CLONH")[occ], f64),
            "tangent_latitude": _float_array(level_field("CLATH"), f64),
            "tangent_longitude": _float_array(level_field("CLONH"), f64),
            "azimuth": _float_array(level_field("BEARAZ"), f64),
            "impact_parameter": _float_array(impact, f64),
            "bending_angle": _float_array(_pick(bnda, primary), f32),
            # BUFR first-order statistics: FOST is the stat-type marker (code 13);
            # the RMS/error value follows it as the second BNDA.
            "bending_angle_error": _float_array(
                _pick(freq_field("BNDA", 1), primary), f32
            ),
            "bending_angle_l1": _float_array(_pick(bnda, l1), f32),
            "bending_angle_l2": _float_array(_pick(bnda, l2), f32),
            "level_pccf": _float_array(level_field("PCCF"), f32),
            "qfro": _int_array(scalar("QFRO")[occ]),
            "earth_radius_curvature": _float_array(elrc[occ], f64),
            "geoid_undulation": _float_array(geodu[occ], f64),
            **{name: _float_array(derived[name], f64) for name in DERIVED_COLUMNS},
            "obs_time": pa.array(
                pd.DatetimeIndex(obs_time).take(occ),
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "obs_date": constant(cycle_date),
            "cycle_date": constant(cycle_date),
            "cycle_hour": constant(cycle_hour),
            "source_file": constant(source_file),
            "data_category": pa.array(
                np.full(n_levels, GPSRO_CODE, dtype=np.int64), type=pa.int64()
            ),
            "data_category_name": constant("GPSRO"),
        }
    )


def empty_gpsro_table() -> pa.Table:
    return pa.table(
        {
            "occultation_id": pa.array([], type=pa.string()),
            "data_category": pa.array([], type=pa.int64()),
            "data_category_name": pa.array([], type=pa.string()),
            "obs_time": pa.array([], type=pa.timestamp("ns", tz="UTC")),
            "cycle_date": pa.array([], type=pa.string()),
            "cycle_hour": pa.array([], type=pa.string()),
        }
    )
