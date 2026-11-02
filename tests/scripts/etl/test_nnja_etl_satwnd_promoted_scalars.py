# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`resolve_promoted_occurrence`: which occurrence supplies a promoted scalar.

A position null for every row is skipped; the first populated position is
returned whole, so per-row nulls stay null. Integer columns are covered because
`np.isfinite` is always true on integer arrays.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from scripts.etl.nnja.satwnd.archive_etl import (
    _column_cache,
    resolve_promoted_occurrence,
)


def _resolve(tbl: pa.Table, mnemonic: str) -> np.ndarray:
    return resolve_promoted_occurrence(tbl, mnemonic, _column_cache(tbl))


def test_resolves_past_a_structurally_empty_occurrence():
    """GOES-R layout: occurrence 0 empty for the whole template."""
    tbl = pa.table(
        {
            "PCCF": pa.array([None, None, None], pa.float64()),  # occ 0, structural
            "PCCF_1": pa.array([96.0, 95.0, 97.0], pa.float64()),  # occ 1, the real one
            "PCCF_2": pa.array([69.0, 70.0, 71.0], pa.float64()),
        }
    ).rename_columns(["PCCF", "PCCF", "PCCF"])
    out = _resolve(tbl, "PCCF")
    assert np.allclose(
        out, [96.0, 95.0, 97.0]
    ), "did not skip the structurally-empty occurrence 0"


def test_does_not_fabricate_from_a_later_occurrence():
    """A row missing inside a populated position stays null."""
    tbl = pa.table(
        {
            "WSPD": pa.array([10.0, None, 12.0], pa.float64()),  # occ 0, populated
            "WSPD_1": pa.array([99.0, 99.0, 99.0], pa.float64()),
        }
    ).rename_columns(["WSPD", "WSPD"])
    out = _resolve(tbl, "WSPD")
    assert out[0] == 10.0 and out[2] == 12.0
    assert np.isnan(out[1]), "fabricated a value from a different occurrence"


def test_unchanged_when_occurrence_zero_is_populated():
    tbl = pa.table(
        {
            "PRLC": pa.array([50000.0, 60000.0], pa.float64()),
            "PRLC_1": pa.array([1.0, 2.0], pa.float64()),
        }
    ).rename_columns(["PRLC", "PRLC"])
    assert np.allclose(_resolve(tbl, "PRLC"), [50000.0, 60000.0])


def test_missing_mnemonic_returns_all_nan_not_an_error():
    tbl = pa.table({"OTHER": pa.array([1.0, 2.0], pa.float64())})
    out = _resolve(tbl, "PCCF")
    assert out.shape == (2,)
    assert np.isnan(out).all()


def test_every_occurrence_empty_stays_honestly_null():
    tbl = pa.table(
        {
            "SWCM": pa.array([None, None], pa.float64()),
            "SWCM_1": pa.array([None, None], pa.float64()),
        }
    ).rename_columns(["SWCM", "SWCM"])
    out = _resolve(tbl, "SWCM")
    assert np.isnan(out).all()


def test_integer_code_table_resolves_too():
    """Integer code tables resolve because `_to_np` returns float64 with NaN."""
    tbl = pa.table(
        {
            "SAID": pa.array([None, None], pa.int64()),  # structurally empty
            "SAID_1": pa.array([270, 270], pa.int64()),  # the real one
        }
    ).rename_columns(["SAID", "SAID"])
    out = _resolve(tbl, "SAID")
    assert out.dtype.kind == "f", "_to_np must normalise ints to float64"
    assert np.allclose(
        out, [270.0, 270.0]
    ), "integer code table did not resolve past the empty occurrence"


def test_integer_column_with_genuine_per_row_nulls_does_not_fabricate():
    tbl = pa.table(
        {
            "HAMD": pa.array([1, None, 3], pa.int64()),  # occ 0, populated
            "HAMD_1": pa.array([9, 9, 9], pa.int64()),
        }
    ).rename_columns(["HAMD", "HAMD"])
    out = _resolve(tbl, "HAMD")
    assert out[0] == 1.0 and out[2] == 3.0
    assert np.isnan(out[1])


def test_to_np_normalises_every_numeric_dtype_to_float64_nan():
    """`resolve_promoted_occurrence` relies on this float64-with-NaN contract."""
    from scripts.etl.nnja.satwnd.archive_etl import _to_np

    for label, arr in (
        ("int64", pa.array([1, None, 3], pa.int64())),
        ("int32", pa.array([1, None, 3], pa.int32())),
        ("uint8", pa.array([1, None, 3], pa.uint8())),
        ("bool", pa.array([True, None, False], pa.bool_())),
        ("float32", pa.array([1.0, None, 3.0], pa.float32())),
    ):
        out = _to_np(pa.table({"x": arr}), 0)
        assert out.dtype == np.float64, f"{label} came back as {out.dtype}"
        assert np.isnan(out[1]), f"{label} null did not become NaN"

    all_null = _to_np(pa.table({"x": pa.array([None, None], pa.int64())}), 0)
    assert not np.isfinite(
        all_null
    ).any(), "an all-null integer column must read as structurally empty"
