# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spectral axis and brightness temperature for the hyperspectral IR sounders.

IASI is stored as radiance and CrIS as integer codes. Converting either to kelvin needs the
wavenumber of every channel, which the archive does not carry, so the instrument grids are
kept here. AIRS is already in kelvin.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# CODATA 2018, in the units CRTM and GSI use, and the same pair earth2studio calls
# `PLANCK_C1`/`PLANCK_C2`. CRTM derives its own from the 1998 values instead, and that vintage
# difference alone is why a GSI diagnostic cannot be reproduced closer than 2.8e-4 K, measured
# at 250 K and 700 cm^-1. Nothing from the SpcCoeff file is being skipped to arrive there: its
# Planck_C1 and Planck_C2 are only C1 * wavenumber**3 and C2 * wavenumber on the grids below,
# with no fitted effective wavenumber, and its Band_C1/Band_C2 correction is identity for the
# replay's IASI and CrIS.
# https://physics.nist.gov/cuu/Constants/Table/allascii.txt
C1 = 1.191042972e-5  # mW m^-2 sr^-1 cm^4
C2 = 1.438776877  # cm K

# The EUMETSAT L1C grid. Verified against the GSI diagnostics: channel 269 gives 712.00.
IASI_FIRST_WAVENUMBER_CM = 645.0
IASI_SPACING_CM = 0.25
IASI_CHANNELS = 8461

# CrIS FSR science channels, numbered contiguously across three bands in the CRTM order GSI
# uses. Verified against the archive's `band_calibration_quality` edges: channels 19, 713,
# 714 and 1570 land on 661.25, 1095.00, 1210.00 and 1745.00 cm^-1.
CRIS_SPACING_CM = 0.625
CRIS_BANDS = (
    (650.0, 1, 713),  # long wave
    (1210.0, 714, 1578),  # mid wave
    (2155.0, 1579, 2211),  # short wave
)

# AIRS wavenumbers are irregular, so they come from a table rather than a formula. The source
# repeats them on every footprint, but they are constant over time.
AIRS_TABLE_PATH = Path(__file__).with_name("airs_wavenumbers.csv")


@functools.cache
def airs_wavenumber_cm() -> dict[int, float]:
    table = pd.read_csv(AIRS_TABLE_PATH)
    return dict(zip(table["channel"].tolist(), table["wavenumber_cm_inverse"].tolist()))


IR_CHANNEL_SET_PATH = Path(__file__).with_name("ir_channel_ranking.csv")
IR_CHANNEL_SET_LENGTH = 48

# The two set sizes the ranking was built for, as prefix lengths of it rather than as literal
# channel lists, so ir32 stays a subset of ir48.
IR_CHANNEL_PRESETS = {"ir32": 32, "ir48": 48}


@functools.cache
def ir_channel_sets() -> dict[str, tuple[int, ...]]:
    """Up to 48 channels per IR sounder, not in channel order."""
    table = pd.read_csv(IR_CHANNEL_SET_PATH)
    curated = {
        sensor: tuple(rows["channel"].tolist())
        for sensor, rows in table.groupby("sensor", sort=False)
    }
    curated["cris"] = curated["cris-fsr"]
    return curated


def ir_channel_preset(name: str) -> dict[str, tuple[int, ...]]:
    """Channel numbers per IR sounder for a named preset. Microwave sensors are absent."""
    if name not in IR_CHANNEL_PRESETS:
        raise ValueError(
            f"unknown IR channel preset {name!r}, expected one of "
            f"{sorted(IR_CHANNEL_PRESETS)}"
        )
    length = IR_CHANNEL_PRESETS[name]
    return {sensor: channels[:length] for sensor, channels in ir_channel_sets().items()}


def wavenumber_cm_inverse(
    sensor: str, channels: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Channel centre wavenumbers in cm^-1, indexed like `channels`."""
    channels = np.asarray(channels, dtype=np.int64)
    if sensor == "iasi":
        _check_range(sensor, channels, 1, IASI_CHANNELS)
        return IASI_FIRST_WAVENUMBER_CM + IASI_SPACING_CM * (channels - 1)
    if sensor in ("cris", "cris-fsr"):
        _check_range(sensor, channels, CRIS_BANDS[0][1], CRIS_BANDS[-1][2])
        result = np.zeros(channels.shape, dtype=np.float64)
        for start, first, last in CRIS_BANDS:
            inside = (channels >= first) & (channels <= last)
            result[inside] = start + CRIS_SPACING_CM * (channels[inside] - first)
        return result
    if sensor == "airs":
        table = airs_wavenumber_cm()
        missing = sorted(set(channels.tolist()) - table.keys())
        if missing:
            raise ValueError(f"airs channels are not in the published table: {missing}")
        return np.array([table[int(c)] for c in channels], dtype=np.float64)
    raise ValueError(f"no spectral axis for sensor {sensor!r}")


def _check_range(sensor: str, channels: np.ndarray, low: int, high: int) -> None:
    if channels.size and (channels.min() < low or channels.max() > high):
        raise ValueError(
            f"{sensor} channels must lie in {low}-{high}, got "
            f"{channels.min()}-{channels.max()}"
        )


def archive_value_encoding(field) -> dict[str, int]:
    """`reference` and `scale` for `radiance_mw`, from an Arrow field's own metadata.

    Empty for a column the archive stores as a physical value.
    """
    spec = (field.metadata or {}).get(b"nnja.archive.value_encoding")
    if not spec or spec == b"null":
        return {}
    parsed = json.loads(spec)
    return {"reference": parsed["reference"], "scale": parsed["scale"]}


def radiance_mw(
    sensor: str,
    values: np.ndarray,
    *,
    reference: int | None = None,
    scale: int | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Archive column to radiance in mW m^-2 sr^-1 (cm^-1)^-1.

    CrIS codes reconstruct to watts, so they need a further 10**3. IASI already arrives on
    this scale, its ETL having folded 10**5 into the CHSF exponent.

    `reference` and `scale` belong to the CrIS column and come from
    `archive_value_encoding`. Every file sampled across the archive states them, always as
    -100000 and 7, so there is no default: a file that stopped stating them should fail
    rather than be reconstructed against a guess.
    """
    values = np.asarray(values, dtype=dtype)
    if sensor in ("cris", "cris-fsr"):
        if reference is None or scale is None:
            raise ValueError("cris needs the reference and scale its column states")
        return (values + reference) * 10.0 ** (3 - scale)
    if sensor == "iasi":
        return values
    raise ValueError(f"{sensor} does not publish radiance")


def brightness_temperature(
    radiance: np.ndarray, wavenumber: np.ndarray, *, dtype: np.dtype = np.float64
) -> np.ndarray:
    """Monochromatic Planck inversion, radiance in mW m^-2 sr^-1 (cm^-1)^-1."""
    radiance = np.asarray(radiance, dtype=dtype)
    # Computed in float64 and cast once, so float32 only coarsens the array arithmetic. The
    # cast is needed: a float64 scalar left in the expression promotes the whole array back.
    wavenumber = np.asarray(wavenumber, dtype=np.float64)
    numerator = np.asarray(C2 * wavenumber, dtype=dtype)
    argument = np.asarray(C1 * wavenumber**3, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        temperature = numerator / np.log(1 + argument / radiance)
    return np.where(radiance > 0, temperature, np.nan)


def spectral_radiance_mw(temperature: np.ndarray, wavenumber: np.ndarray) -> np.ndarray:
    """Planck function, the inverse of `brightness_temperature`."""
    temperature = np.asarray(temperature, dtype=np.float64)
    wavenumber = np.asarray(wavenumber, dtype=np.float64)
    return C1 * wavenumber**3 / np.expm1(C2 * wavenumber / temperature)
