# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPS-RO vertical coordinates from the occultation's own refractivity profile.

Shared by the NNJA GPS-RO ETL and the in-memory table path, so both derive the
same archive columns."""

from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray

_REFRACTIVITY_CONSTANT_K_HPA = 77.60
_DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
_EARTH_SEMI_MAJOR_M = 6_378_137.0
_EARTH_ECCENTRICITY_SQUARED = 6.69437999013e-3
_EQUATORIAL_GRAVITY_M_S2 = 9.7803253359
_SOMIGLIANA_K = 1.93185265241e-3
_DEFAULT_TOP_SCALE_HEIGHT_M = 7_000.0
_MIN_TOP_SCALE_HEIGHT_M = 2_000.0
_MAX_TOP_SCALE_HEIGHT_M = 15_000.0

# Below this height the moist lower troposphere biases dry pressure high, so the
# fixed standard atmosphere carries the unbiased mean; above it dry hydrostatic
# wins. 5 km is the observed RMSE crossover (healda_gpsro_debug 2022 benchmark).
# A std-weighted mix below the crossover keeps std's mean while borrowing dry's
# profile structure; 0.8/0.2 is that benchmark's least-squares-optimal weight.
BLEND_HEIGHT_M = 5_000.0
BLEND_LOW_STD_WEIGHT = 0.8

# US 1976 standard atmosphere (same layering as healda nnja_gpsro_loader).
_BASE_HEIGHT_M = np.asarray(
    (0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0)
)
_LAPSE_RATE_K_M = np.asarray((-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028))
_STD_G = 9.80665
_STD_RD = 287.05287
_STD_T0 = 288.15
_STD_P0 = 1013.25


def _std_bases() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    temperature = np.empty(len(_BASE_HEIGHT_M), dtype=np.float64)
    pressure = np.empty(len(_BASE_HEIGHT_M), dtype=np.float64)
    temperature[0] = _STD_T0
    pressure[0] = _STD_P0
    for layer, lapse_rate in enumerate(_LAPSE_RATE_K_M):
        delta = _BASE_HEIGHT_M[layer + 1] - _BASE_HEIGHT_M[layer]
        temperature[layer + 1] = temperature[layer] + lapse_rate * delta
        if lapse_rate == 0.0:
            pressure[layer + 1] = pressure[layer] * np.exp(
                -_STD_G * delta / (_STD_RD * temperature[layer])
            )
        else:
            pressure[layer + 1] = pressure[layer] * (
                temperature[layer + 1] / temperature[layer]
            ) ** (-_STD_G / (_STD_RD * lapse_rate))
    return temperature, pressure


_BASE_T, _BASE_P = _std_bases()


def standard_atmosphere_pressure_hpa(
    height_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """US Standard Atmosphere pressure (hPa) at geometric height (m), clipped 0-60 km."""
    height = np.clip(np.asarray(height_m, dtype=np.float64), 0.0, 60_000.0)
    output = np.full(height.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(height)
    if not np.any(finite):
        return output
    values = height[finite]
    layer = np.clip(
        np.searchsorted(_BASE_HEIGHT_M, values, side="right") - 1,
        0,
        len(_LAPSE_RATE_K_M) - 1,
    )
    delta = values - _BASE_HEIGHT_M[layer]
    lapse = _LAPSE_RATE_K_M[layer]
    temperature = _BASE_T[layer] + lapse * delta
    pressure = np.empty_like(values)
    gradient = lapse != 0.0
    pressure[gradient] = _BASE_P[layer[gradient]] * (
        temperature[gradient] / _BASE_T[layer[gradient]]
    ) ** (-_STD_G / (_STD_RD * lapse[gradient]))
    pressure[~gradient] = _BASE_P[layer[~gradient]] * np.exp(
        -_STD_G * delta[~gradient] / (_STD_RD * _BASE_T[layer[~gradient]])
    )
    output[finite] = pressure
    return output


@numba.njit(cache=True, nogil=True)
def _dry_pressure_profile_kernel(
    height, refractivity, latitude_degrees, top_fit_levels, output
):
    count = 0
    for k in range(height.size):
        if (
            np.isfinite(height[k])
            and np.isfinite(refractivity[k])
            and refractivity[k] > 0.0
        ):
            count += 1
    if count < 2:
        return
    source = np.empty(count, dtype=np.int64)
    count = 0
    for k in range(height.size):
        if (
            np.isfinite(height[k])
            and np.isfinite(refractivity[k])
            and refractivity[k] > 0.0
        ):
            source[count] = k
            count += 1
    order = source[np.argsort(height[source], kind="mergesort")]
    keep = np.empty(order.size, dtype=np.bool_)
    keep[0] = True
    for k in range(1, order.size):
        keep[k] = height[order[k]] - height[order[k - 1]] > 0.0
    order = order[keep]
    m = order.size
    if m < 2:
        return
    sorted_height = height[order]
    sorted_refractivity = refractivity[order]
    sin_squared = np.sin(np.deg2rad(latitude_degrees)) ** 2
    surface = (
        _EQUATORIAL_GRAVITY_M_S2
        * (1.0 + _SOMIGLIANA_K * sin_squared)
        / np.sqrt(1.0 - _EARTH_ECCENTRICITY_SQUARED * sin_squared)
    )
    weight = np.empty(m)
    for k in range(m):
        gravity = (
            surface
            * (_EARTH_SEMI_MAJOR_M / (_EARTH_SEMI_MAJOR_M + sorted_height[k])) ** 2
        )
        density = (
            100.0
            * sorted_refractivity[k]
            / (_REFRACTIVITY_CONSTANT_K_HPA * _DRY_AIR_GAS_CONSTANT_J_KG_K)
        )
        weight[k] = density * gravity
    # Least-squares slope of log N on height over the top levels (np.polyfit degree 1, closed form).
    take = min(top_fit_levels, m)
    x = sorted_height[m - take :]
    y = np.log(sorted_refractivity[m - take :])
    x_mean = x.mean()
    y_mean = y.mean()
    sxy = 0.0
    sxx = 0.0
    for k in range(take):
        sxy += (x[k] - x_mean) * (y[k] - y_mean)
        sxx += (x[k] - x_mean) * (x[k] - x_mean)
    slope = sxy / sxx
    scale_height = _DEFAULT_TOP_SCALE_HEIGHT_M
    if np.isfinite(slope) and slope < 0.0:
        candidate = -1.0 / slope
        if _MIN_TOP_SCALE_HEIGHT_M <= candidate <= _MAX_TOP_SCALE_HEIGHT_M:
            scale_height = candidate
    top_pressure_pa = weight[m - 1] * scale_height
    tail = 0.0
    output[order[m - 1]] = (top_pressure_pa + tail) / 100.0
    for k in range(m - 2, -1, -1):
        tail += (
            0.5
            * (sorted_height[k + 1] - sorted_height[k])
            * (weight[k] + weight[k + 1])
        )
        output[order[k]] = (top_pressure_pa + tail) / 100.0


@numba.njit(cache=True, nogil=True)
def _interp_sorted(impact, abscissa, ordinate, log_scale, output):
    """np.interp of `impact` onto strictly increasing `abscissa`, inside its range only; exp() back when `log_scale`.

    Same arithmetic as numpy's interp: binary search, slope * (x - xp[j]) + fp[j], the last point returned exactly.
    """
    last = abscissa.size - 1
    for k in range(impact.size):
        x = impact[k]
        if not (np.isfinite(x) and abscissa[0] <= x <= abscissa[last]):
            continue
        if x == abscissa[last]:
            value = ordinate[last]
        else:
            j = np.searchsorted(abscissa, x, side="right") - 1
            slope = (ordinate[j + 1] - ordinate[j]) / (abscissa[j + 1] - abscissa[j])
            value = slope * (x - abscissa[j]) + ordinate[j]
            if np.isnan(value):
                value = slope * (x - abscissa[j + 1]) + ordinate[j + 1]
                if np.isnan(value) and ordinate[j] == ordinate[j + 1]:
                    value = ordinate[j]
        output[k] = np.exp(value) if log_scale else value


@numba.njit(cache=True, nogil=True)
def _derive_kernel(
    impact, impact_offsets, height, refractivity, profile_offsets,
    latitude, radius_of_curvature, geoid, top_fit_levels, dry_out, heit_out,
):  # fmt: skip
    for o in range(impact_offsets.size - 1):
        i0, i1 = impact_offsets[o], impact_offsets[o + 1]
        p0, p1 = profile_offsets[o], profile_offsets[o + 1]
        lat, curvature = latitude[o], radius_of_curvature[o]
        if i1 == i0 or p1 - p0 < 2 or np.isnan(curvature) or not np.isfinite(lat):
            continue
        h = height[p0:p1]
        n = refractivity[p0:p1]
        x = impact[i0:i1]
        pressure = np.full(h.size, np.nan)
        _dry_pressure_profile_kernel(h, n, lat, top_fit_levels, pressure)
        radius = (1.0 + 1.0e-6 * n) * (curvature + geoid[o] + h)
        # Dry pressure at impact: levels with a finite radius and positive pressure, sorted by radius.
        count = 0
        for k in range(h.size):
            if (
                np.isfinite(radius[k])
                and np.isfinite(pressure[k])
                and pressure[k] > 0.0
            ):
                count += 1
        if count >= 2:
            valid = np.empty(count, dtype=np.int64)
            count = 0
            for k in range(h.size):
                if (
                    np.isfinite(radius[k])
                    and np.isfinite(pressure[k])
                    and pressure[k] > 0.0
                ):
                    valid[count] = k
                    count += 1
            order = valid[np.argsort(radius[valid], kind="mergesort")]
            sorted_radius = radius[order]
            keep = np.empty(order.size, dtype=np.bool_)
            keep[0] = True
            for k in range(1, order.size):
                keep[k] = sorted_radius[k] - sorted_radius[k - 1] > 0.0
            _interp_sorted(
                x,
                sorted_radius[keep],
                np.log(pressure[order][keep]),
                True,
                dry_out[i0:i1],
            )
        # HEIT at impact: valid refractivity levels sorted by height, then mapped to refractive radius.
        count = 0
        for k in range(h.size):
            if np.isfinite(h[k]) and np.isfinite(n[k]) and n[k] > 0.0:
                count += 1
        if count >= 2:
            valid = np.empty(count, dtype=np.int64)
            count = 0
            for k in range(h.size):
                if np.isfinite(h[k]) and np.isfinite(n[k]) and n[k] > 0.0:
                    valid[count] = k
                    count += 1
            order = valid[np.argsort(h[valid], kind="mergesort")]
            sorted_h = h[order]
            level_radius = (1.0 + 1.0e-6 * n[order]) * (curvature + geoid[o] + sorted_h)
            keep = np.empty(order.size, dtype=np.bool_)
            keep[0] = True
            for k in range(1, order.size):
                keep[k] = level_radius[k] - level_radius[k - 1] > 0.0
            if keep.sum() >= 2:
                _interp_sorted(
                    x, level_radius[keep], sorted_h[keep], False, heit_out[i0:i1]
                )


def derive_level_coordinates_batch(
    impact_parameter_m: NDArray[np.float64],
    impact_offsets: NDArray[np.int64],
    height_m: NDArray[np.float64],
    refractivity_n_units: NDArray[np.float64],
    profile_offsets: NDArray[np.int64],
    *,
    latitude_degrees: NDArray[np.float64],
    earth_radius_of_curvature_m: NDArray[np.float64],
    geoid_undulation_m: NDArray[np.float64],
    top_fit_levels: int = 8,
) -> dict[str, NDArray[np.float64]]:
    """Vertical coordinates of the bending-angle levels of many occultations at once.

    Occultation ``o`` owns ``impact_parameter_m[impact_offsets[o]:impact_offsets[o + 1]]`` and the refractivity
    profile ``height_m`` / ``refractivity_n_units`` over ``profile_offsets[o]:profile_offsets[o + 1]``, both raw
    as decoded. Per occultation, a NaN radius of curvature means none was reported and a NaN latitude skips the
    profile derivation; geoid undulation is used as given (callers pass 0.0 when none was reported). Returns the
    archive columns ``impact_height_m``, ``heit_at_impact_m``, ``dry_pressure_hpa``,
    ``standard_atmosphere_pressure_hpa`` and ``blended_pressure_5km_hpa``, aligned to ``impact_parameter_m``.
    """
    if top_fit_levels < 2:
        raise ValueError("top_fit_levels must be at least 2")
    impact = np.ascontiguousarray(impact_parameter_m, dtype=np.float64)
    impact_offsets = np.ascontiguousarray(impact_offsets, dtype=np.int64)
    curvature = np.ascontiguousarray(earth_radius_of_curvature_m, dtype=np.float64)
    dry = np.full(impact.shape, np.nan, dtype=np.float64)
    heit = dry.copy()
    _derive_kernel(
        impact, impact_offsets,
        np.ascontiguousarray(height_m, dtype=np.float64), np.ascontiguousarray(refractivity_n_units, dtype=np.float64),
        np.ascontiguousarray(profile_offsets, dtype=np.int64),
        np.ascontiguousarray(latitude_degrees, dtype=np.float64), curvature,
        np.ascontiguousarray(geoid_undulation_m, dtype=np.float64), top_fit_levels, dry, heit,
    )  # fmt: skip
    impact_height = impact - np.repeat(curvature, np.diff(impact_offsets))
    isa = standard_atmosphere_pressure_hpa(heit)
    low = impact_height < BLEND_HEIGHT_M
    weighted = BLEND_LOW_STD_WEIGHT * isa + (1.0 - BLEND_LOW_STD_WEIGHT) * dry
    blended = np.where(low, weighted, dry)
    fallback = np.where(
        low,
        np.where(np.isfinite(isa), isa, dry),
        np.where(np.isfinite(dry), dry, isa),
    )
    blended = np.where(np.isfinite(blended), blended, fallback)
    return {
        "impact_height_m": impact_height,
        "heit_at_impact_m": heit,
        "dry_pressure_hpa": dry,
        "standard_atmosphere_pressure_hpa": isa,
        "blended_pressure_5km_hpa": blended,
    }


def derive_level_coordinates(
    impact_parameter_m: NDArray[np.float64],
    height_m: NDArray[np.float64],
    refractivity_n_units: NDArray[np.float64],
    *,
    latitude_degrees: float | None,
    earth_radius_of_curvature_m: float | None,
    geoid_undulation_m: float | None,
    top_fit_levels: int = 8,
) -> dict[str, NDArray[np.float64]]:
    """One occultation through ``derive_level_coordinates_batch``; ``None`` means not reported."""
    impact = np.asarray(impact_parameter_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    return derive_level_coordinates_batch(
        impact,
        np.array([0, impact.size]),
        height,
        np.asarray(refractivity_n_units, dtype=np.float64),
        np.array([0, height.size]),
        latitude_degrees=np.array(
            [np.nan if latitude_degrees is None else latitude_degrees], dtype=np.float64
        ),
        earth_radius_of_curvature_m=np.array(
            [
                np.nan
                if earth_radius_of_curvature_m is None
                else earth_radius_of_curvature_m
            ],
            dtype=np.float64,
        ),
        geoid_undulation_m=np.array(
            [0.0 if geoid_undulation_m is None else geoid_undulation_m],
            dtype=np.float64,
        ),
        top_fit_levels=top_fit_levels,
    )
