# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPS-RO vertical coordinates from the occultation's own refractivity profile.

Shared by the NNJA GPS-RO ETL and the in-memory table path, so both derive the
same archive columns."""

from __future__ import annotations

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


def normal_gravity_m_s2(
    latitude_degrees: float, height_m: NDArray[np.float64]
) -> NDArray[np.float64]:
    height = np.asarray(height_m, dtype=np.float64)
    sin_squared = np.sin(np.deg2rad(latitude_degrees)) ** 2
    surface = (
        _EQUATORIAL_GRAVITY_M_S2
        * (1.0 + _SOMIGLIANA_K * sin_squared)
        / np.sqrt(1.0 - _EARTH_ECCENTRICITY_SQUARED * sin_squared)
    )
    return surface * (_EARTH_SEMI_MAJOR_M / (_EARTH_SEMI_MAJOR_M + height)) ** 2


def dry_density_kg_m3(refractivity_n_units: NDArray[np.float64]) -> NDArray[np.float64]:
    refractivity = np.asarray(refractivity_n_units, dtype=np.float64)
    return (
        100.0
        * refractivity
        / (_REFRACTIVITY_CONSTANT_K_HPA * _DRY_AIR_GAS_CONSTANT_J_KG_K)
    )


def _top_scale_height_m(
    height_m: NDArray[np.float64],
    refractivity_n_units: NDArray[np.float64],
    *,
    top_fit_levels: int,
) -> float:
    if top_fit_levels < 2:
        raise ValueError("top_fit_levels must be at least 2")
    take = min(top_fit_levels, height_m.size)
    slope = np.polyfit(height_m[-take:], np.log(refractivity_n_units[-take:]), 1)[0]
    if not np.isfinite(slope) or slope >= 0.0:
        return _DEFAULT_TOP_SCALE_HEIGHT_M
    scale_height = -1.0 / slope
    if not _MIN_TOP_SCALE_HEIGHT_M <= scale_height <= _MAX_TOP_SCALE_HEIGHT_M:
        return _DEFAULT_TOP_SCALE_HEIGHT_M
    return float(scale_height)


def dry_pressure_profile_hpa(
    height_m: NDArray[np.float64],
    refractivity_n_units: NDArray[np.float64],
    *,
    latitude_degrees: float,
    top_fit_levels: int = 8,
) -> NDArray[np.float64]:
    height = np.asarray(height_m, dtype=np.float64)
    refractivity = np.asarray(refractivity_n_units, dtype=np.float64)
    if height.shape != refractivity.shape:
        raise ValueError("height and refractivity must have identical shapes")
    output = np.full(height.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(height) & np.isfinite(refractivity) & (refractivity > 0.0)
    if np.count_nonzero(valid) < 2:
        return output

    source_indices = np.flatnonzero(valid)
    order = source_indices[np.argsort(height[source_indices], kind="stable")]
    sorted_height = height[order]
    sorted_refractivity = refractivity[order]
    unique = np.concatenate(([True], np.diff(sorted_height) > 0.0))
    order = order[unique]
    sorted_height = sorted_height[unique]
    sorted_refractivity = sorted_refractivity[unique]
    if sorted_height.size < 2:
        return output

    gravity = normal_gravity_m_s2(latitude_degrees, sorted_height)
    density = dry_density_kg_m3(sorted_refractivity)
    scale_height = _top_scale_height_m(
        sorted_height, sorted_refractivity, top_fit_levels=top_fit_levels
    )
    weight = density * gravity
    segment = 0.5 * np.diff(sorted_height) * (weight[:-1] + weight[1:])
    top_pressure_pa = weight[-1] * scale_height
    tail_sum = np.concatenate((np.cumsum(segment[::-1])[::-1], [0.0]))
    output[order] = (top_pressure_pa + tail_sum) / 100.0
    return output


def refractive_radius_m(
    height_m: NDArray[np.float64],
    refractivity_n_units: NDArray[np.float64],
    *,
    earth_radius_of_curvature_m: float,
    geoid_undulation_m: float,
) -> NDArray[np.float64]:
    height = np.asarray(height_m, dtype=np.float64)
    refractivity = np.asarray(refractivity_n_units, dtype=np.float64)
    geometric_radius = earth_radius_of_curvature_m + geoid_undulation_m + height
    return (1.0 + 1.0e-6 * refractivity) * geometric_radius


def dry_pressure_at_impact_hpa(
    impact_parameter_m: NDArray[np.float64],
    refractive_radius_profile_m: NDArray[np.float64],
    dry_pressure_profile: NDArray[np.float64],
) -> NDArray[np.float64]:
    impact = np.asarray(impact_parameter_m, dtype=np.float64)
    radius = np.asarray(refractive_radius_profile_m, dtype=np.float64)
    pressure = np.asarray(dry_pressure_profile, dtype=np.float64)
    if radius.shape != pressure.shape:
        raise ValueError("refractive radius and pressure must have identical shapes")
    valid = np.isfinite(radius) & np.isfinite(pressure) & (pressure > 0.0)
    output = np.full(impact.shape, np.nan, dtype=np.float64)
    if np.count_nonzero(valid) < 2:
        return output
    order = np.argsort(radius[valid], kind="stable")
    sorted_radius = radius[valid][order]
    sorted_pressure = pressure[valid][order]
    unique = np.concatenate(([True], np.diff(sorted_radius) > 0.0))
    sorted_radius = sorted_radius[unique]
    sorted_pressure = sorted_pressure[unique]
    inside = (
        np.isfinite(impact)
        & (impact >= sorted_radius[0])
        & (impact <= sorted_radius[-1])
    )
    output[inside] = np.exp(
        np.interp(impact[inside], sorted_radius, np.log(sorted_pressure))
    )
    return output


def heit_at_impact_m(
    impact_parameter_m: NDArray[np.float64],
    height_m: NDArray[np.float64],
    refractivity_n_units: NDArray[np.float64],
    *,
    earth_radius_of_curvature_m: float,
    geoid_undulation_m: float,
) -> NDArray[np.float64]:
    """True geometric HEIT interpolated to each bending level's impact parameter.

    Bouguer's rule ties impact parameter to refractive radius, x = (1+N*1e-6)*r
    (see ``refractive_radius_m``). ``impact_parameter_m - earth_radius_of_curvature_m``
    (the GSI-convention "impact height") is a ray-geometric quantity, not the true
    tangent-point height; it diverges from HEIT by ~2 km at the surface because it
    has no refraction correction. This inverts the same mapping ``dry_pressure_at_impact_hpa``
    uses for pressure, but for height, so the returned value is the physically correct
    geometric height at the exact impact parameter (not a same-index-paired approximation).
    """
    impact = np.asarray(impact_parameter_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    refractivity = np.asarray(refractivity_n_units, dtype=np.float64)
    output = np.full(impact.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(height) & np.isfinite(refractivity) & (refractivity > 0.0)
    if np.count_nonzero(valid) < 2:
        return output
    order = np.argsort(height[valid], kind="stable")
    sorted_height = height[valid][order]
    sorted_refractivity = refractivity[valid][order]
    radius = refractive_radius_m(
        sorted_height,
        sorted_refractivity,
        earth_radius_of_curvature_m=earth_radius_of_curvature_m,
        geoid_undulation_m=geoid_undulation_m,
    )
    unique = np.concatenate(([True], np.diff(radius) > 0.0))
    radius = radius[unique]
    sorted_height = sorted_height[unique]
    if radius.size < 2:
        return output
    inside = np.isfinite(impact) & (impact >= radius[0]) & (impact <= radius[-1])
    output[inside] = np.interp(impact[inside], radius, sorted_height)
    return output


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
    """Vertical coordinates of one occultation's bending-angle levels.

    ``height_m``/``refractivity_n_units`` are the occultation's refractivity
    profile as decoded; levels without a finite HEIT and a positive ARFR are
    screened out by the functions above, so callers pass it raw. Returns the
    archive columns ``impact_height_m``, ``heit_at_impact_m``,
    ``dry_pressure_hpa``, ``standard_atmosphere_pressure_hpa`` and
    ``blended_pressure_5km_hpa``, aligned to ``impact_parameter_m``.
    """
    impact = np.asarray(impact_parameter_m, dtype=np.float64)
    dry = np.full(impact.shape, np.nan, dtype=np.float64)
    heit = dry.copy()
    if (
        impact.size
        and np.asarray(height_m).size >= 2
        and earth_radius_of_curvature_m is not None
        and latitude_degrees is not None
        and np.isfinite(latitude_degrees)
    ):
        geoid = 0.0 if geoid_undulation_m is None else float(geoid_undulation_m)
        pressure_profile = dry_pressure_profile_hpa(
            height_m,
            refractivity_n_units,
            latitude_degrees=float(latitude_degrees),
            top_fit_levels=top_fit_levels,
        )
        radius_profile = refractive_radius_m(
            height_m,
            refractivity_n_units,
            earth_radius_of_curvature_m=float(earth_radius_of_curvature_m),
            geoid_undulation_m=geoid,
        )
        dry = dry_pressure_at_impact_hpa(impact, radius_profile, pressure_profile)
        heit = heit_at_impact_m(
            impact,
            height_m,
            refractivity_n_units,
            earth_radius_of_curvature_m=float(earth_radius_of_curvature_m),
            geoid_undulation_m=geoid,
        )
    if earth_radius_of_curvature_m is not None:
        impact_height = impact - float(earth_radius_of_curvature_m)
    else:
        impact_height = np.full(impact.shape, np.nan, dtype=np.float64)
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
