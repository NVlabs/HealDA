# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy equivalence tests for the SATWND kernels."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from healda.observations.preprocessing import satwnd_kernels, satwnd_qc
from healda.observations.preprocessing.conv_plevel import (
    _PRESSURE_LEVEL_EDGES,
    nearest_pressure_level_index,
    nearest_pressure_level_index_scalar,
)
from healda.observations.loaders.nnja_satwnd import (
    ARCHIVE_HPX_LEVEL,
    SATWND_TYPE_MIN,
    _winners_by_cell,
)

WINDOW = np.datetime64("2019-08-13T12:00:00", "ns")
ELEMENTWISE_DISABLE = {
    None: {},
    "structural": {"require_structural": False},
    "out_of_window": {"require_in_window": False},
    "bad_swcm": {"require_valid_swcm": False},
    "pressure_floor": {"pressure_floor_hpa": None},
    "zenith_limb": {"zenith_limit_deg": None},
    "speed_implausible": {"maximum_wind_speed": None},
    "cawv_slow": {"apply_cawv_speed": False},
    "layer_wind": {"exclude_layer_winds": False},
}


def make_qc_inputs(n, seed):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-95.0, 95.0, n)
    lon = rng.uniform(-180.0, 360.0, n)
    pressure_pa = rng.uniform(-1_000.0, 110_000.0, n)
    u = rng.normal(0.0, 60.0, n)
    v = rng.normal(0.0, 60.0, n)
    u[rng.random(n) < 0.05] *= 6.0
    slow = rng.random(n) < 0.05
    u[slow] = rng.uniform(-4.0, 4.0, int(slow.sum()))
    v[slow] = rng.uniform(-4.0, 4.0, int(slow.sum()))
    swcm = rng.integers(0, 9, n).astype(np.float64)
    zenith = rng.uniform(-90.0, 90.0, n)
    istype_raw = rng.integers(-1, 21, n).astype(np.float64)
    report_type = rng.integers(240, 261, n).astype(np.int32)
    hour_ns = np.int64(3_600_000_000_000)
    offset_ns = rng.integers(-4 * hour_ns, 4 * hour_ns, n, dtype=np.int64)
    times = WINDOW + offset_ns.astype("timedelta64[ns]")

    for values in (lat, lon, pressure_pa, u, v, swcm, zenith, istype_raw):
        values[rng.random(n) < 0.02] = np.nan
    times[rng.random(n) < 0.02] = np.datetime64("NaT", "ns")
    if n >= 8:
        boundary_ns = 3 * hour_ns
        times[:8] = WINDOW + np.array(
            [
                -boundary_ns - 999_999_999,
                -boundary_ns - 1,
                -boundary_ns,
                -boundary_ns + 1,
                boundary_ns - 1,
                boundary_ns,
                boundary_ns + 1,
                boundary_ns + 999_999_999,
            ],
            dtype="timedelta64[ns]",
        )

    table = pa.table(
        {
            "latitude": lat,
            "longitude": lon,
            "assigned_pressure": pressure_pa,
            "wind_u_derived": u,
            "wind_v_derived": v,
            "wind_computation_method": swcm,
            "satellite_zenith_angle": zenith,
            "gsi_internal_subtype": istype_raw,
            "time_utc": pa.array(times, type=pa.timestamp("ns")),
        }
    )
    arrays = {
        "lat": lat,
        "lon": lon,
        "pressure": pressure_pa / 100.0,
        "u": u,
        "v": v,
        "swcm": swcm,
        "zenith": zenith,
        "istype": np.where(np.isfinite(istype_raw), istype_raw, -1).astype(np.int8),
        "times": times,
    }
    return table, report_type, arrays


def numpy_qc_result(arrays, report_type, config):
    lat = arrays["lat"]
    lon = arrays["lon"]
    pressure = arrays["pressure"]
    u = arrays["u"]
    v = arrays["v"]
    swcm = arrays["swcm"]
    zenith = arrays["zenith"]
    istype = arrays["istype"]
    times = arrays["times"]

    reject = np.zeros(lat.size, dtype=np.uint32)
    counts = {name: 0 for name in satwnd_qc.SCREEN_NAMES}
    diagnostics = {"no_family": int(np.count_nonzero(istype < 0))}

    def fire(flag, mask, name):
        counts[name] = int(np.count_nonzero(mask))
        reject[mask] |= np.uint32(flag)

    # 1. Structural validity
    if config.require_structural:
        bad = ~(
            np.isfinite(lat)
            & np.isfinite(lon)
            & np.isfinite(pressure)
            & np.isfinite(u)
            & np.isfinite(v)
        )
        bad |= np.isnat(times)
        bad |= (lat < -90.0) | (lat > 90.0) | (pressure <= 0.0)
        fire(satwnd_qc.Reject.STRUCTURAL, bad, "structural")

    # 2. Window membership
    if config.require_in_window:
        delta = np.abs(times - WINDOW).astype("timedelta64[s]").astype(np.float64)
        out = ~np.isfinite(delta) | (delta > 3.0 * 3600.0)
        out &= ~np.isnat(times)
        fire(satwnd_qc.Reject.OUT_OF_WINDOW, out, "out_of_window")

    # 3. SWCM validity
    if config.require_valid_swcm:
        bad = (
            ~np.isfinite(swcm)
            | (swcm < satwnd_qc.SWCM_MIN)
            | (swcm > satwnd_qc.SWCM_MAX)
        )
        fire(satwnd_qc.Reject.BAD_SWCM, bad, "bad_swcm")

    # 4. Pressure floor
    if config.pressure_floor_hpa is not None:
        bad = np.isfinite(pressure) & (pressure < config.pressure_floor_hpa)
        fire(satwnd_qc.Reject.PRESSURE_FLOOR, bad, "pressure_floor")

    # 5. GEO limb angle
    if config.zenith_limit_deg is not None:
        geo_lookup = np.zeros(128, dtype=bool)
        geo_lookup[list(satwnd_qc.GEO_LIMB_CASES)] = True
        geo = (istype >= 0) & geo_lookup[np.maximum(istype, 0)]
        diagnostics["zenith_not_applicable"] = int(np.count_nonzero(~geo))
        bad = geo & np.isfinite(zenith) & (zenith > config.zenith_limit_deg)
        fire(satwnd_qc.Reject.ZENITH_LIMB, bad, "zenith_limb")

    # 7. Speed rules
    speed_sq = u * u + v * v
    if config.maximum_wind_speed is not None:
        bad = np.isfinite(speed_sq) & (speed_sq > float(config.maximum_wind_speed) ** 2)
        fire(satwnd_qc.Reject.SPEED_IMPLAUSIBLE, bad, "speed_implausible")

    if config.apply_cawv_speed:
        bad = (
            (report_type == satwnd_qc.CAWV_TYPE)
            & (istype == satwnd_qc.CAWV_ISTYPE)
            & np.isfinite(speed_sq)
            & (speed_sq < satwnd_qc.CAWV_MIN_SPEED**2)
        )
        fire(satwnd_qc.Reject.CAWV_SLOW, bad, "cawv_slow")

    # 8. Layer winds
    if config.exclude_layer_winds:
        bad = np.isfinite(swcm) & (swcm >= satwnd_qc.LAYER_SWCM_MIN)
        fire(satwnd_qc.Reject.LAYER_WIND, bad, "layer_wind")

    return reject, counts, diagnostics


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("disabled_screen", ELEMENTWISE_DISABLE)
def test_qc_kernel_matches_numpy(seed, disabled_screen):
    table, report_type, arrays = make_qc_inputs(20_000, seed)
    config = satwnd_qc.SatwndQCConfig(
        apply_quarantine=False,
        **ELEMENTWISE_DISABLE[disabled_screen],
    )
    expected_reject, expected_counts, expected_diagnostics = numpy_qc_result(
        arrays, report_type, config
    )
    result = satwnd_qc.evaluate(table, WINDOW, report_type, config=config)

    np.testing.assert_array_equal(result.reject, expected_reject)
    assert result.counts == expected_counts
    assert result.diagnostics == expected_diagnostics
    assert result.kept == int(np.count_nonzero(expected_reject == 0))


def test_qc_kernel_rejects_short_geo_lookup(monkeypatch):
    table, report_type, _ = make_qc_inputs(1, 0)
    monkeypatch.setattr(satwnd_qc, "_case_lookup", lambda _: np.zeros(127, dtype=bool))
    config = satwnd_qc.SatwndQCConfig(apply_quarantine=False)

    with pytest.raises(ValueError):
        satwnd_qc.evaluate(table, WINDOW, report_type, config=config)


def numpy_winners_by_cell(key, score):
    codes, uniques = pd.factorize(key, sort=False)
    owner = np.zeros(len(uniques), dtype=np.int64)
    order = np.argsort(score, kind="stable")[::-1]
    owner[codes[order]] = order
    return owner


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_winner_kernel_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    key = rng.integers(0, 100, 5_000, dtype=np.uint64)
    score = rng.integers(0, 20, 5_000, dtype=np.int64)

    np.testing.assert_array_equal(
        _winners_by_cell(key, score), numpy_winners_by_cell(key, score)
    )


def numpy_key_score(
    rows,
    pixel_f64,
    pressure_hpa,
    time_ns,
    report_type,
    window_ns,
    thin_hpx_level,
    pressure_bin_hpa,
    time_bin_hours,
):
    pixel = pixel_f64[rows].astype(np.uint64)
    horizontal = pixel >> np.uint64(2 * (ARCHIVE_HPX_LEVEL - thin_hpx_level))

    if pressure_bin_hpa is None:
        pressure_cell = nearest_pressure_level_index(pressure_hpa[rows]).astype(
            np.uint64
        )
    else:
        pressure_cell = np.floor(
            (1200.0 - pressure_hpa[rows]) / pressure_bin_hpa
        ).astype(np.int64)
        pressure_cell = np.clip(pressure_cell, 0, 31).astype(np.uint64)

    offset_ns = time_ns[rows] - window_ns
    distance_ns = np.abs(offset_ns)
    offset_hours = offset_ns / 3_600_000_000_000
    time_cell = np.floor((offset_hours + 3.0) / time_bin_hours).astype(np.uint64)

    horizontal_bits = 2 * thin_hpx_level + 4
    key = horizontal.astype(np.uint64)
    key |= pressure_cell << np.uint64(horizontal_bits)
    key |= time_cell << np.uint64(horizontal_bits + 5)
    key |= (
        report_type[rows].astype(np.uint64) - np.uint64(SATWND_TYPE_MIN)
    ) << np.uint64(horizontal_bits + 9)

    return key, distance_ns


def test_scalar_pressure_buckets_match_numpy_at_edges():
    pressure = np.concatenate(
        [
            np.nextafter(
                _PRESSURE_LEVEL_EDGES,
                np.full_like(_PRESSURE_LEVEL_EDGES, -np.inf),
            ),
            _PRESSURE_LEVEL_EDGES,
            np.nextafter(
                _PRESSURE_LEVEL_EDGES,
                np.full_like(_PRESSURE_LEVEL_EDGES, np.inf),
            ),
        ]
    )
    expected = nearest_pressure_level_index(pressure)
    actual = np.array(
        [nearest_pressure_level_index_scalar(value) for value in pressure],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(actual, expected)


def make_inputs(n, seed, thin_hpx_level=5):
    rng = np.random.default_rng(seed)
    # Sparse pixel domain so cells actually collide (that's the whole point of thinning).
    pixel_int = rng.integers(0, 4096, n).astype(np.uint64) << np.uint64(
        2 * (ARCHIVE_HPX_LEVEL - thin_hpx_level)
    )
    pixel_f64 = pixel_int.astype(np.float64)
    pressure_hpa = rng.uniform(50.0, 1050.0, n)
    window_ns = np.int64(1_600_000_000_000_000_000)
    time_ns = window_ns + rng.integers(-3 * 3600, 3 * 3600, n).astype(
        np.int64
    ) * np.int64(1_000_000_000)
    report_type = rng.integers(240, 261, n).astype(np.int32)
    rows = np.arange(n, dtype=np.int64)
    return rows, pixel_f64, pressure_hpa, time_ns, report_type, window_ns


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pressure_bin_hpa", [None, 25.0])
def test_thin_kernel_matches_numpy(seed, pressure_bin_hpa):
    all_rows, pixel_f64, pressure_hpa, time_ns, report_type, window_ns = make_inputs(
        5_000, seed
    )
    rows = np.sort(np.random.default_rng(seed + 10).choice(all_rows, 2_000, False))
    thin_hpx_level = 5
    time_bin_hours = 2.0

    expected_key, expected_score = numpy_key_score(
        rows,
        pixel_f64,
        pressure_hpa,
        time_ns,
        report_type,
        window_ns,
        thin_hpx_level,
        pressure_bin_hpa,
        time_bin_hours,
    )
    got_key, got_score = satwnd_kernels.build_thinning_keys(
        rows=rows,
        pixel_f64=pixel_f64,
        pressure_hpa=pressure_hpa,
        time_ns=time_ns[rows],
        report_type=report_type,
        window_ns=int(window_ns),
        thin_hpx_level=thin_hpx_level,
        archive_hpx_level=ARCHIVE_HPX_LEVEL,
        pressure_bin_hpa=pressure_bin_hpa,
        time_bin_hours=time_bin_hours,
        report_type_min=SATWND_TYPE_MIN,
    )
    np.testing.assert_array_equal(got_key, expected_key)
    np.testing.assert_array_equal(got_score, expected_score)


def test_thin_empty_rows():
    rows = np.array([], dtype=np.int64)
    pixel_f64 = np.array([], dtype=np.float64)
    pressure_hpa = np.array([], dtype=np.float64)
    time_ns = np.array([], dtype=np.int64)
    report_type = np.array([], dtype=np.int32)
    key, score = satwnd_kernels.build_thinning_keys(
        rows=rows,
        pixel_f64=pixel_f64,
        pressure_hpa=pressure_hpa,
        time_ns=time_ns,
        report_type=report_type,
        window_ns=0,
        thin_hpx_level=5,
        archive_hpx_level=ARCHIVE_HPX_LEVEL,
        pressure_bin_hpa=None,
        time_bin_hours=2.0,
        report_type_min=SATWND_TYPE_MIN,
    )
    assert key.size == 0 and score.size == 0
