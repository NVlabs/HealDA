# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numba kernels for SATWND elementwise QC and thinning keys."""

from __future__ import annotations

import numba
import numpy as np

from healda.observations.preprocessing.conv_plevel import (
    nearest_pressure_level_index_scalar,
)


# fmt: off
@numba.njit(cache=True, nogil=True, fastmath=False)
def _qc_kernel(
    lat, lon, pressure, u, v, swcm, zenith,
    istype, times_ns, report_type, geo_lookup,
    f_structural, f_out_of_window, f_bad_swcm, f_pressure_floor,
    f_zenith_limb, f_speed, f_cawv, f_layer,
    do_structural, do_window, do_swcm, do_floor,
    do_zenith, do_speed, do_cawv, do_layer,
    window_ns, half_width_s, swcm_min, swcm_max,
    pressure_floor, zenith_limit, speed_limit_sq,
    cawv_type, cawv_istype, cawv_min_sq, layer_swcm_min, nat_ns,
):
# fmt: on
    n = lat.shape[0]
    reject = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        bits = np.uint32(0)
        la = lat[i]
        pr = pressure[i]
        uu = u[i]
        vv = v[i]
        sw = swcm[i]
        ze = zenith[i]
        it = istype[i]
        t = times_ns[i]
        is_nat = t == nat_ns

        if do_structural:
            bad = not (
                np.isfinite(la)
                and np.isfinite(lon[i])
                and np.isfinite(pr)
                and np.isfinite(uu)
                and np.isfinite(vv)
            )
            if is_nat or la < -90.0 or la > 90.0 or pr <= 0.0:
                bad = True
            if bad:
                bits |= f_structural

        if do_window and not is_nat:
            d = t - window_ns
            if d < 0:
                d = -d
            if np.float64(d // 1_000_000_000) > half_width_s:
                bits |= f_out_of_window

        if do_swcm:
            if (not np.isfinite(sw)) or sw < swcm_min or sw > swcm_max:
                bits |= f_bad_swcm

        if do_floor:
            if np.isfinite(pr) and pr < pressure_floor:
                bits |= f_pressure_floor

        geo = it >= 0 and geo_lookup[it]

        if do_zenith:
            if geo and np.isfinite(ze) and ze > zenith_limit:
                bits |= f_zenith_limb

        if do_speed or do_cawv:
            sq = uu * uu + vv * vv
            if do_speed and np.isfinite(sq) and sq > speed_limit_sq:
                bits |= f_speed
            if (
                do_cawv
                and report_type[i] == cawv_type
                and it == cawv_istype
                and np.isfinite(sq)
                and sq < cawv_min_sq
            ):
                bits |= f_cawv

        if do_layer:
            if np.isfinite(sw) and sw >= layer_swcm_min:
                bits |= f_layer

        reject[i] = bits
    return reject


# fmt: off
def apply_elementwise_screens(
    *,
    lat, lon, pressure, u, v, swcm, zenith,
    istype, times, report_type, geo_lookup,
    flags, config, window_ns, window_half_width_hours,
    swcm_min, swcm_max, layer_swcm_min,
    cawv_type, cawv_istype, cawv_min_speed,
):
# fmt: on
    """Return reject bits and diagnostics for the elementwise screens."""

    nat = np.iinfo(np.int64).min
    times_ns = np.ascontiguousarray(times.view(np.int64))
    istype = np.ascontiguousarray(istype, dtype=np.int8)
    geo_lookup = np.ascontiguousarray(geo_lookup, dtype=np.bool_)
    # Numba disables bounds checks; cover every non-negative istype value.
    if geo_lookup.size <= np.iinfo(istype.dtype).max:
        raise ValueError("geo lookup does not cover istype")
    # Reject flags come from satwnd_qc.Reject to avoid duplicate bit definitions.
    # fmt: off
    reject_bits = _qc_kernel(
        np.ascontiguousarray(lat, dtype=np.float64),
        np.ascontiguousarray(lon, dtype=np.float64),
        np.ascontiguousarray(pressure, dtype=np.float64),
        np.ascontiguousarray(u, dtype=np.float64),
        np.ascontiguousarray(v, dtype=np.float64),
        np.ascontiguousarray(swcm, dtype=np.float64),
        np.ascontiguousarray(zenith, dtype=np.float64),
        istype, times_ns,
        np.ascontiguousarray(report_type, dtype=np.int32),
        geo_lookup,
        np.uint32(flags["structural"]), np.uint32(flags["out_of_window"]),
        np.uint32(flags["bad_swcm"]), np.uint32(flags["pressure_floor"]),
        np.uint32(flags["zenith_limb"]), np.uint32(flags["speed_implausible"]),
        np.uint32(flags["cawv_slow"]), np.uint32(flags["layer_wind"]),
        config.require_structural, config.require_in_window,
        config.require_valid_swcm, config.pressure_floor_hpa is not None,
        config.zenith_limit_deg is not None, config.maximum_wind_speed is not None,
        config.apply_cawv_speed, config.exclude_layer_winds,
        np.int64(window_ns), np.float64(window_half_width_hours * 3600.0),
        np.float64(swcm_min), np.float64(swcm_max),
        np.float64(config.pressure_floor_hpa or 0.0),
        np.float64(config.zenith_limit_deg or 0.0),
        np.float64((config.maximum_wind_speed or 0.0) ** 2),
        np.int64(cawv_type), np.int64(cawv_istype),
        np.float64(cawv_min_speed**2), np.float64(layer_swcm_min), np.int64(nat),
    )
    # fmt: on
    diagnostics = {}
    if config.zenith_limit_deg is not None:
        istype_arr = np.asarray(istype)
        geo = (istype_arr >= 0) & geo_lookup[np.maximum(istype_arr, 0)]
        diagnostics["zenith_not_applicable"] = int(np.count_nonzero(~geo))
    return reject_bits, diagnostics


@numba.njit(cache=True, nogil=True)
def _build_thinning_keys_kernel(
    rows,
    pixel_f64,
    pressure_hpa,
    time_ns,
    report_type,
    window_ns,
    shift,
    horizontal_bits,
    has_pressure_bin,
    pressure_bin_hpa,
    time_bin_hours,
    report_type_min,
):
    n = rows.shape[0]
    key = np.empty(n, dtype=np.uint64)
    score = np.empty(n, dtype=np.int64)

    for i in range(n):
        r = rows[i]
        horizontal = np.uint64(pixel_f64[r]) >> shift

        if has_pressure_bin:
            value = pressure_hpa[r]
            pressure_cell_float = np.floor((1200.0 - value) / pressure_bin_hpa)
            if pressure_cell_float < 0.0:
                pressure_cell_float = 0.0
            elif pressure_cell_float > 31.0:
                pressure_cell_float = 31.0
            pressure_cell = np.uint64(np.int64(pressure_cell_float))
        else:
            pressure_cell = np.uint64(
                nearest_pressure_level_index_scalar(np.float32(pressure_hpa[r]))
            )

        offset_ns = time_ns[i] - window_ns
        distance_ns = offset_ns if offset_ns >= 0 else -offset_ns
        offset_hours = offset_ns / 3_600_000_000_000.0
        time_cell = np.uint64(np.floor((offset_hours + 3.0) / time_bin_hours))
        report_type_cell = np.uint64(np.int64(report_type[r]) - report_type_min)

        packed_key = horizontal
        packed_key |= pressure_cell << np.uint64(horizontal_bits)
        packed_key |= time_cell << np.uint64(horizontal_bits + 5)
        packed_key |= report_type_cell << np.uint64(horizontal_bits + 9)

        key[i] = packed_key
        score[i] = distance_ns

    return key, score


def build_thinning_keys(
    *,
    rows: np.ndarray,
    pixel_f64: np.ndarray,
    pressure_hpa: np.ndarray,
    time_ns: np.ndarray,
    report_type: np.ndarray,
    window_ns: int,
    thin_hpx_level: int,
    archive_hpx_level: int,
    pressure_bin_hpa: float | None,
    time_bin_hours: float,
    report_type_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return packed thinning keys and time distances for the selected rows."""
    return _build_thinning_keys_kernel(
        np.ascontiguousarray(rows, dtype=np.int64),
        np.ascontiguousarray(pixel_f64, dtype=np.float64),
        np.ascontiguousarray(pressure_hpa, dtype=np.float64),
        np.ascontiguousarray(time_ns, dtype=np.int64),
        np.ascontiguousarray(report_type, dtype=np.int32),
        np.int64(window_ns),
        np.uint64(2 * (archive_hpx_level - thin_hpx_level)),
        int(2 * thin_hpx_level + 4),
        pressure_bin_hpa is not None,
        np.float64(pressure_bin_hpa or 0.0),
        np.float64(time_bin_hours),
        np.int64(report_type_min),
    )
