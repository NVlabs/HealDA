# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GSI report type of a raw SATWND wind, from ``(subset, SAID, SWCM)``.

The raw dump carries no report type. Each rule maps an ``NC005xxx`` subset, its computation
methods (SWCM) and satellite ids (SAID) to GSI's type (240-260), its direct-reader processing
case (-1 where that reader skips the wind) and the route the mapping comes from, which the
loader carries in ``QC_Flag``. The NNJA SATWND archive was typed with these rules.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import pandas as pd

# Routes, in the order the loader encodes them in QC_Flag (loaders.nnja_satwnd.TIER_CODES).
DIRECT = "1-direct-gsi-sattab"  # GSI read_satwnd.f90 sattab
DIRECT_UNUSED = (
    "1-direct-gsi-sattab-unusable-istype-1"  # typed, but the reader skips it
)
PRE_REFACTOR = "2-pre-refactor-gsi"  # SWCM >= 4 -> 247, dropped by a GSI refactor
PREPDATA = "3-prepdata"  # NCEP PREPDATA route, where the direct reader cannot ingest

GOES = (*range(250, 260), *range(270, 274), *range(731, 736))
GOES_ID_RANGE = tuple(range(250, 300))
HIMAWARI = (150, 151, 152, 153, 154, 171, 172, 173, 174, 253)
METEOSAT = (50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 70, 71)
MODIS = (783, 784)
AVHRR = (3, 4, 5, 206, 207, 208, 209, 223, 225, 226)
VIIRS = (224, 225, 226)
LEO_GEO = (854,)
INSAT = (410, 430, 431, 432, 450, 451, 452, 470)
INSAT_3D = (471, 473, 474)
INSAT_ALL = INSAT + INSAT_3D
TC_AMV = tuple(sorted(set(GOES + HIMAWARI + METEOSAT + VIIRS)))

# (subset, SWCMs, SAIDs, report type, processing case, route)
_RULES: tuple[tuple[str, tuple[int, ...], tuple[int, ...], int, int, str], ...] = (
    ("NC005001", (1,), GOES, 245, 6, DIRECT),
    ("NC005001", (4, 5, 6, 7), GOES_ID_RANGE, 247, 6, PRE_REFACTOR),
    ("NC005002", (2,), GOES, 251, 6, DIRECT),
    ("NC005002", (4, 5, 6, 7), GOES_ID_RANGE, 247, 6, PRE_REFACTOR),
    ("NC005003", (3,), GOES, 246, 6, DIRECT),
    ("NC005003", (4, 5, 6, 7), GOES_ID_RANGE, 247, 6, PRE_REFACTOR),
    ("NC005004", (4,), GOES, 245, 6, DIRECT),
    ("NC005005", (1,), GOES, 245, 6, DIRECT),
    ("NC005006", (3,), GOES, 251, 6, DIRECT),
    ("NC005008", (2,), GOES, 246, 6, DIRECT),
    ("NC005009", (4,), GOES, 245, 6, DIRECT),
    ("NC005010", (1,), GOES, 245, 7, DIRECT),
    ("NC005010", (4, 5, 6, 7), GOES_ID_RANGE, 247, 7, PRE_REFACTOR),
    ("NC005011", (3,), GOES, 246, 7, DIRECT),
    ("NC005011", (4, 5, 6, 7), GOES_ID_RANGE, 247, 7, PRE_REFACTOR),
    ("NC005012", (2,), GOES, 251, 7, DIRECT),
    ("NC005012", (4, 5, 6, 7), GOES_ID_RANGE, 247, 7, PRE_REFACTOR),
    ("NC005013", (4,), GOES, 245, 7, DIRECT),
    ("NC005015", (1,), GOES, 245, 7, DIRECT),
    ("NC005016", (3,), GOES, 246, 7, DIRECT),
    ("NC005017", (2,), GOES, 251, 7, DIRECT),
    ("NC005019", (1,), GOES, 240, 11, DIRECT),
    ("NC005021", (1,), INSAT, 256, -1, DIRECT_UNUSED),
    ("NC005022", (2,), INSAT, 256, -1, DIRECT_UNUSED),
    ("NC005023", (3,), INSAT, 256, -1, DIRECT_UNUSED),
    ("NC005024", (1,), INSAT_ALL, 241, -1, PREPDATA),
    ("NC005025", (2,), INSAT_ALL, 241, -1, PREPDATA),
    ("NC005026", (3,), INSAT, 256, -1, DIRECT_UNUSED),
    ("NC005026", (3,), INSAT_3D, 256, -1, PREPDATA),
    ("NC005026", (4, 5, 6, 7), INSAT_ALL, 256, -1, PREPDATA),
    ("NC005030", (1,), GOES, 245, 15, DIRECT),
    ("NC005031", (4, 5), GOES, 247, 19, DIRECT),
    ("NC005032", (2,), GOES, 251, 17, DIRECT),
    ("NC005034", (3,), GOES, 246, 18, DIRECT),
    ("NC005039", (1,), GOES, 240, 16, DIRECT),
    ("NC005041", (1,), HIMAWARI, 252, 3, DIRECT),
    ("NC005042", (2,), HIMAWARI, 242, 3, DIRECT),
    ("NC005043", (3, 4, 5), HIMAWARI, 250, 3, DIRECT),
    ("NC005044", (1,), HIMAWARI, 252, 4, DIRECT),
    ("NC005045", (2,), HIMAWARI, 242, 4, DIRECT),
    ("NC005046", (3, 4, 5), HIMAWARI, 250, 4, DIRECT),
    ("NC005047", (1,), HIMAWARI, 253, 5, DIRECT),
    ("NC005048", (2,), HIMAWARI, 242, 5, DIRECT),
    ("NC005049", (3, 4, 5), HIMAWARI, 250, 5, DIRECT),
    ("NC005052", (1,), GOES, 245, 15, DIRECT),
    ("NC005053", (4,), GOES, 247, 19, DIRECT),
    ("NC005054", (2,), GOES, 251, 17, DIRECT),
    ("NC005055", (3,), GOES, 246, 18, DIRECT),
    ("NC005056", (1,), GOES, 240, 16, DIRECT),
    ("NC005061", (1,), METEOSAT, 253, 0, DIRECT),
    ("NC005062", (2,), METEOSAT, 243, 0, DIRECT),
    ("NC005063", (3, 4, 5), METEOSAT, 254, 0, DIRECT),
    ("NC005064", (1,), METEOSAT, 253, 1, DIRECT),
    ("NC005065", (2,), METEOSAT, 243, 1, DIRECT),
    ("NC005066", (3, 4, 5), METEOSAT, 254, 1, DIRECT),
    ("NC005067", (1,), METEOSAT, 253, 2, DIRECT),
    ("NC005068", (2,), METEOSAT, 243, 2, DIRECT),
    ("NC005069", (3, 4, 5), METEOSAT, 254, 2, DIRECT),
    ("NC005070", (1,), MODIS, 257, 8, DIRECT),
    ("NC005071", (3,), MODIS, 258, 8, DIRECT),
    ("NC005071", (4, 5), MODIS, 259, 8, DIRECT),
    ("NC005072", (1,), LEO_GEO, 255, 12, DIRECT),
    ("NC005080", (1,), AVHRR, 244, 9, DIRECT),
    ("NC005081", (1,), AVHRR, 244, 10, DIRECT),
    ("NC005090", (1,), VIIRS, 260, 13, DIRECT),
    ("NC005091", (1,), VIIRS, 260, 14, DIRECT),
    ("NC005099", (1,), TC_AMV, 241, 20, DIRECT),
    ("NC005099", (2, 3, 4, 5, 6), GOES, 241, 20, DIRECT),
)
_MAX_SAID = 1023


class GsiTypes(NamedTuple):
    """Per-row resolution; ``report_type``/``tier`` are -1/None where untyped."""

    report_type: np.ndarray  # int64, -1 untyped
    internal_subtype: np.ndarray  # int64, -1 when the direct reader has no case
    tier: np.ndarray  # object, route name or None


@functools.cache
def _lookup() -> (
    tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        tuple[str, ...],
    ]
):
    """Per-subset dense arrays indexed by ``(said << 3) | swcm``."""
    size = (_MAX_SAID + 1) << 3
    tier_names = (DIRECT, DIRECT_UNUSED, PRE_REFACTOR, PREPDATA)
    types: dict[str, np.ndarray] = {}
    subtypes: dict[str, np.ndarray] = {}
    tiers: dict[str, np.ndarray] = {}
    for subset, methods, saids, report_type, case, tier in _RULES:
        if subset not in types:
            types[subset] = np.full(size, -1, dtype=np.int64)
            subtypes[subset] = np.full(size, -1, dtype=np.int64)
            tiers[subset] = np.full(size, -1, dtype=np.int64)
        keys = [(said << 3) | swcm for said in saids for swcm in methods]
        types[subset][keys] = report_type
        subtypes[subset][keys] = case
        tiers[subset][keys] = tier_names.index(tier)
    return types, subtypes, tiers, tier_names


def resolve(
    subset: np.ndarray,
    said: np.ndarray,
    swcm: np.ndarray,
    swcm_local: np.ndarray | None = None,
) -> GsiTypes:
    """Type every wind.

    ``subset`` holds ``NC005xxx`` names; ``said``/``swcm`` are float with NaN
    for missing. A subset whose rows carry no SWCM at all falls back to the
    NCEP-local ``swcm_local`` (CMCM), as the ETL does per template; the
    fallback is never applied row by row.
    """
    types, subtypes, tier_codes, tier_names = _lookup()
    n = len(said)
    said = np.asarray(said, dtype=np.float64)
    swcm = np.array(swcm, dtype=np.float64)
    subset = np.array(subset, dtype=object)
    present = pd.notna(subset)
    subset[~present] = None  # pd.NA would poison the per-subset comparisons below
    report_type = np.full(n, -1, dtype=np.int64)
    internal = np.full(n, -1, dtype=np.int64)
    tier_index = np.full(n, -1, dtype=np.int64)
    for name in np.unique(subset[present]):
        rows = subset == name
        if swcm_local is not None and not np.isfinite(swcm[rows]).any():
            swcm[rows] = np.asarray(swcm_local, dtype=np.float64)[rows]
        if name not in types:
            continue
        usable = (
            rows
            & np.isfinite(said)
            & np.isfinite(swcm)
            & (said >= 0)
            & (said <= _MAX_SAID)
            & (swcm >= 1)
            & (swcm <= 7)
        )
        index = np.flatnonzero(usable)
        keys = (said[index].astype(np.int64) << 3) | swcm[index].astype(np.int64)
        report_type[index] = types[name][keys]
        internal[index] = subtypes[name][keys]
        tier_index[index] = tier_codes[name][keys]
    tier = np.array([None, *tier_names], dtype=object)[tier_index + 1]
    return GsiTypes(report_type, internal, tier)
