# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forecast-independent QC for NNJA SATWND/AMV observations.

The ML input gate. Not the archive contract (the archive keeps everything), and
not GSI's "used" decision -- nothing here reads a first guess, an innovation, or
convinfo iuse. Every screen records why it fired, and a screen that cannot be
evaluated says so rather than passing silently.

Thresholds come from GSI ``read_satwnd.f90`` @860d1374, cited at each use site.
Almost nothing there is global: screens dispatch on ``istype``, stored per row as
``gsi_internal_subtype``. Applying one globally rejects data GSI keeps.

QI is resolved by GENERATING APPLICATION, never by position -- GNAP 1 is ``qify``
for EUMETSAT and ``qifn`` for NESDIS. Where no qifn is resolvable the screen is
skipped and counted in ``diagnostics``; "not screened" is not "passed".
"""

from __future__ import annotations

import dataclasses
import functools
from enum import IntFlag
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from healda.observations.preprocessing import satwnd_kernels

# --- GSI dispatch tables, all verified against read_satwnd.f90 --------------

#: istype with an ACTIVE `zangl > 68 -> cycle`. Excludes istype 6 (line is
#: commented out), istype 20 (angle forced to 61.23), and every LEO family --
#: which is why the signed cross-track angle needs no abs().
# gsi_internal_subtype is a small enum; the lookup tables span its domain.
_ISTYPE_DOMAIN = 128
GEO_LIMB_CASES = frozenset({1, 2, 4, 5, 7, 11, 15, 16, 17, 18, 19})

#: istype values with ``if(qifn < 85) qm=15``. MODIS(8), AVHRR(9), METOP(10),
#: NESDIS(11) and LEOGEO(12) READ qifn but never screen on it.
QI85_CASES = frozenset({1, 2, 4, 5, 7, 13})

#: The with-forecast twin of QIFN_GNAP_BY_CASE. Resolved only so a caller can
#: DEMONSTRATE that the two differ; screening on it would leak the forecast.
QIFY_GNAP_BY_CASE: Mapping[int, int] = {1: 1, 4: 101, 7: 3, 8: 3, 9: 3, 11: 3, 13: 3}

#: GOES-R (istype 15-20) uses a different rule entirely: a lower threshold AND
#: an upper bound, ``if (qifn < 80 .or. qifn > 100) qm=15``.
GOESR_CASES = frozenset({15, 16, 17, 18, 19, 20})
GOESR_QI_MIN, GOESR_QI_MAX = 80.0, 100.0

#: Per-family GNAP number that denotes qifn (QI WITHOUT forecast).
QIFN_GNAP_BY_CASE: Mapping[int, int] = {
    1: 2,  # EUMETSAT
    4: 102,  # JMA
    7: 1,  # NESDIS / GOES legacy
    8: 1,  # MODIS
    9: 1,  # AVHRR
    11: 1,  # NESDIS
    13: 1,  # VIIRS
}

#: These read QI from the ``AMVQIC`` sequence, not GNAP triplets. Their
#: ``generating_application`` is NULL; the value lives under
#: ``standard_generating_application`` (GNAPS) with ``per_cent_confidence``.
AMVQIC_CASES = frozenset({2, 5, 10, 14, 15, 16, 17, 18, 19, 20})

#: GSI reads AMVQIC positionally (`amvqic(2,2)`); we select by GNAPS instead,
#: because the layout varies: [6,5], [6,5,4,2], [4,5,7] ascending, [5,7]. In
#: the last, GSI's slot holds GNAPS 7, not 5. Divergence is counted.
AMVQIC_QIFN_GNAPS = 5
AMVQIC_EE_GNAPS = 2
GSI_AMVQIC_QIFN_POSITION = 1  # what GSI would have taken, for comparison

#: `if(itype == 247 .and. obsdat(4) < 10)` sits inside case(7), so it reaches
#: istype 7 only. Type 247 also arises from istype 6, which has no speed rule.
CAWV_TYPE = 247
CAWV_ISTYPE = 7
CAWV_MIN_SPEED = 10.0

#: Type 247 is explicitly exempt from the qifn screen in istype 7:
#: ``if(qifn <85 .and. itype /= 247) qm=15``, commented "QI not applied to
#: CAWV for now".
QI_EXEMPT_TYPES = frozenset({247})

#: Populations whose values are bit-corrupt, keyed by identity. NC005064 on
#: three 2005 cycles decodes WDIR constant 323.00, WSPD 16 distinct values
#: spaced 25.60 m/s, PRLC 32 spaced 19.20 hPa; NCEPLIBS reproduces them, so it
#: is a source defect. Must be an identity screen: the lattice starts at
#: 25.4 m/s, so a speed cap leaves most of it in place.
QUARANTINED_POPULATIONS: tuple[tuple[str, str], ...] = (
    ("NC005064", "2005-06-07T12"),
    ("NC005064", "2005-06-07T18"),
    ("NC005064", "2005-06-08T00"),
)

PRESSURE_FLOOR_HPA = 125.0  # global hard reject in GSI, before the dispatch
ZENITH_LIMIT_DEG = 68.0
SWCM_MIN, SWCM_MAX = 1, 7
LAYER_SWCM_MIN = 4  # SWCM >= 4 are deep-layer / layer-mean winds


class Reject(IntFlag):
    """Why a row was rejected. Bits compose; a row may fail several screens."""

    NONE = 0
    STRUCTURAL = 1 << 0  # missing/non-finite geolocation, time, wind, pressure
    OUT_OF_WINDOW = 1 << 1
    BAD_SWCM = 1 << 2
    PRESSURE_FLOOR = 1 << 3  # above 125 hPa
    ZENITH_LIMB = 1 << 4  # GEO families only
    QI_LOW = 1 << 5  # GNAP-resolved qifn below the family threshold
    SPEED_IMPLAUSIBLE = 1 << 6
    CAWV_SLOW = 1 << 7  # type 247 below 10 m/s
    LAYER_WIND = 1 << 8  # SWCM >= 4, a representation contract issue
    DUPLICATE = 1 << 9
    QUARANTINED = 1 << 10  # known bit-corrupt source population, by identity


@dataclasses.dataclass(frozen=True)
class SatwndQCConfig:
    """Every screen individually toggleable, so each can be ablated."""

    require_structural: bool = True
    require_in_window: bool = True
    require_valid_swcm: bool = True
    pressure_floor_hpa: float | None = PRESSURE_FLOOR_HPA
    zenith_limit_deg: float | None = ZENITH_LIMIT_DEG
    #: OFF: qifn is unresolvable for ~45% of 2019 GOES-R rows, so enabling it
    #: screens some products and not others. Read QI_UNAVAILABLE if you do.
    apply_qifn: bool = False
    apply_cawv_speed: bool = True
    #: Above any credible AMV (jet maxima ~120 m/s).
    maximum_wind_speed: float | None = 160.0
    #: SWCM >= 4 are layer means; the model treats each token as a point wind
    #: at one pressure. Not a quality judgement -- the winds are fine.
    exclude_layer_winds: bool = True
    #: OFF: thinning already collapses exact duplicates into one cell. Verified
    #: identical output on five windows. On only to COUNT them.
    deduplicate: bool = False
    #: ON: these rows are bit-corrupt and no magnitude screen catches them.
    apply_quarantine: bool = True


#: Every screen name, so a result always reports all of them. A screen that is
#: switched off must not be indistinguishable from one that rejected nothing --
#: that difference is the whole point of an ablation.
SCREEN_NAMES: tuple[str, ...] = (
    "structural",
    "out_of_window",
    "bad_swcm",
    "pressure_floor",
    "zenith_limb",
    "qi_low",
    "speed_implausible",
    "cawv_slow",
    "layer_wind",
    "duplicate",
    "quarantined",
)


@dataclasses.dataclass(frozen=True)
class QCResult:
    reject: np.ndarray  # uint32 bitmask per row, 0 == kept
    counts: dict[str, int]  # per-screen fired counts (NOT mutually exclusive)
    diagnostics: dict[str, int]  # facts, not drops -- never subtracted from kept
    disabled: frozenset[str]  # screens that did not run at all
    kept: int
    total: int

    @property
    def keep_mask(self) -> np.ndarray:
        return self.reject == 0


@functools.lru_cache(maxsize=8)
def _case_lookup(cases: frozenset[int]) -> np.ndarray:
    # Spans the istype domain, not max(cases): a table sized to the set makes
    # every istype above it alias onto the last slot.
    table = np.zeros(_ISTYPE_DOMAIN, dtype=bool)
    table[list(cases)] = True
    return table


def _in_cases(values: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    # -1 is "no family" and never a member.
    return (values >= 0) & lookup[np.maximum(values, 0)]


def _f64(table: pa.Table, name: str, n: int) -> np.ndarray:
    if name not in table.schema.names:
        return np.full(n, np.nan)
    column = table.column(name).combine_chunks()
    if isinstance(column, pa.ChunkedArray):
        column = column.chunk(0) if column.num_chunks else pa.array([], pa.float64())
    return column.cast(pa.float64()).to_numpy(zero_copy_only=False)


class AmvqicLayoutError(ValueError):
    """A row carries more than one AMVQIC occurrence claiming to be qifn."""


def _list_parts(table: pa.Table, name: str):
    column = table.column(name).combine_chunks()
    if isinstance(column, pa.ChunkedArray):
        column = column.chunk(0) if column.num_chunks else None
    if column is None or len(column) == 0:
        return None
    return column


def resolve_qi(table: pa.Table, istype: np.ndarray):
    """Forecast-independent QI per row, plus its with-forecast twin and ee.

    Returns ``(qifn, qify, ee, resolvable, diverged)``. Two dispatch paths,
    because the archive carries two encodings:

      * GNAP triplet families -- select the occurrence whose GNAP equals the
        FAMILY'S qifn code. GNAP 1 is qifn for NESDIS but qify for EUMETSAT, so
        this is per-family and never global.
      * AMVQIC families -- select semantically, by GNAPS, because the position
        varies. `diverged` counts rows where position and GNAPS disagree.

    ``qify`` is returned only for comparison against ``qifn`` in tests; the two
    differ, and screening on qify is the leak this module exists to avoid.
    """
    n = len(istype)
    qifn = np.full(n, np.nan)
    qify = np.full(n, np.nan)
    ee = np.full(n, np.nan)
    if "quality_diagnostics" not in table.schema.names:
        return qifn, qify, ee, np.zeros(n, dtype=bool), 0

    column = _list_parts(table, "quality_diagnostics")
    if column is None:
        return qifn, qify, ee, np.zeros(n, dtype=bool), 0
    offsets = column.offsets.to_numpy()
    flat = column.flatten()
    if len(flat) == 0:
        return qifn, qify, ee, np.zeros(n, dtype=bool), 0

    names = set(flat.type.names)

    def field(name: str) -> np.ndarray:
        # Absent is not zero: a missing column must leave the value
        # unresolvable, not silently compare equal to something.
        if name not in names:
            return np.full(len(flat), np.nan)
        return np.asarray(
            flat.field(name).to_numpy(zero_copy_only=False), dtype="float64"
        )

    gnap = field("generating_application")
    gnaps = field("standard_generating_application")
    pccf = field("per_cent_confidence")
    counts = np.diff(offsets)
    row_of = np.repeat(np.arange(n), counts)
    position = np.arange(len(row_of)) - np.repeat(offsets[:-1], counts)

    is_amvqic = _in_cases(istype, _case_lookup(AMVQIC_CASES))

    def first_where(mask: np.ndarray, out: np.ndarray) -> None:
        if not mask.any():
            return
        rows = row_of[mask]
        vals = pccf[mask]
        out[rows[::-1]] = vals[::-1]  # earliest occurrence wins

    # --- GNAP triplet families --------------------------------------------
    want_n = np.full(n, -1.0)
    want_y = np.full(n, -1.0)
    for case, code in QIFN_GNAP_BY_CASE.items():
        want_n[istype == case] = code
    for case, code in QIFY_GNAP_BY_CASE.items():
        want_y[istype == case] = code
    finite = np.isfinite(gnap) & np.isfinite(pccf)
    first_where(finite & (gnap == want_n[row_of]), qifn)
    first_where(finite & (gnap == want_y[row_of]), qify)

    # --- AMVQIC families, selected SEMANTICALLY by GNAPS -------------------
    seq = is_amvqic[row_of] & np.isfinite(pccf) & np.isfinite(gnaps)
    is_qifn = seq & (gnaps == AMVQIC_QIFN_GNAPS)
    # More than one qifn occurrence in a row would make "first" arbitrary.
    if is_qifn.any():
        per_row = np.bincount(row_of[is_qifn], minlength=n)
        if (per_row > 1).any():
            raise AmvqicLayoutError(
                f"{int(np.count_nonzero(per_row > 1)):,} AMVQIC rows carry more than one "
                f"GNAPS={AMVQIC_QIFN_GNAPS} occurrence, so qifn is ambiguous."
            )
    first_where(is_qifn, qifn)
    first_where(seq & (gnaps == AMVQIC_EE_GNAPS), ee)

    # How often our semantic choice differs from GSI's positional slot. Not an
    # error -- GSI is the one reading the wrong occurrence for NC005091 -- but
    # it must be visible rather than assumed away.
    diverged = int(np.count_nonzero(is_qifn & (position != GSI_AMVQIC_QIFN_POSITION)))

    return qifn, qify, ee, np.isfinite(qifn), diverged


def evaluate(
    table: pa.Table,
    window: np.datetime64,
    report_type: np.ndarray,
    config: SatwndQCConfig = SatwndQCConfig(),
    window_half_width_hours: float = 3.0,
    columns: dict[str, np.ndarray] | None = None,
) -> QCResult:
    """Screen a projected SATWND table. Returns per-row bitmasks, never drops.

    ``report_type`` is the GSI observation type already resolved by the caller
    (read-time GNAP/table resolution lives in the loader, not here), with -1
    for rows that could not be typed.

    ``columns`` supplies arrays the caller has already materialised, so a
    loader that needs u/v/pressure anyway does not pay for them twice.
    """
    supplied = columns or {}
    # Normalise up front: an int64 comparison against a datetime64 of another
    # unit silently mis-scales, which made the quarantine's window check miss.
    window = np.datetime64(window, "ns")
    n = table.num_rows
    reject = np.zeros(n, dtype=np.uint32)
    diagnostics: dict[str, int] = {}
    counts: dict[str, int] = {name: 0 for name in SCREEN_NAMES}
    disabled: set[str] = set()
    if not config.require_structural:
        disabled.add("structural")
    if not config.require_in_window:
        disabled.add("out_of_window")
    if not config.require_valid_swcm:
        disabled.add("bad_swcm")
    if config.pressure_floor_hpa is None:
        disabled.add("pressure_floor")
    if config.zenith_limit_deg is None:
        disabled.add("zenith_limb")
    if not config.apply_qifn:
        disabled.add("qi_low")
    if config.maximum_wind_speed is None:
        disabled.add("speed_implausible")
    if not config.apply_cawv_speed:
        disabled.add("cawv_slow")
    if not config.exclude_layer_winds:
        disabled.add("layer_wind")
    if not config.deduplicate:
        disabled.add("duplicate")
    if not config.apply_quarantine:
        disabled.add("quarantined")

    def fire(flag: Reject, mask: np.ndarray, label: str) -> None:
        hits = int(np.count_nonzero(mask))
        counts[label] = hits
        if hits:
            reject[mask] |= np.uint32(int(flag))

    # Threshold comparisons use float64 for caller-independent boundary behavior.
    def column(name: str) -> np.ndarray:
        preloaded = supplied.get(name)
        if preloaded is None:
            return _f64(table, name, n)
        return np.asarray(preloaded, np.float64)

    lat = column("latitude")
    lon = column("longitude")
    preloaded_pressure = supplied.get("pressure_hpa")
    pressure = (
        _f64(table, "assigned_pressure", n) / 100.0  # Pa -> hPa
        if preloaded_pressure is None
        else np.asarray(preloaded_pressure, np.float64)
    )
    u = column("wind_u_derived")
    v = column("wind_v_derived")
    swcm = _f64(table, "wind_computation_method", n)
    zenith = _f64(table, "satellite_zenith_angle", n)

    istype_raw = _f64(table, "gsi_internal_subtype", n)
    istype = np.where(np.isfinite(istype_raw), istype_raw, -1).astype(np.int8)
    diagnostics["no_family"] = int(np.count_nonzero(istype < 0))

    # --- 0. quarantine, before anything reads a value -----------------------
    # First on purpose: every screen below trusts the numbers, and for these
    # rows the numbers are the thing that is wrong.
    if config.apply_quarantine:
        bad = np.zeros(n, dtype=bool)
        missing = {"ncep_dump_subtype", "cycle"} - set(table.schema.names)
        if missing and QUARANTINED_POPULATIONS:
            raise ValueError(
                f"apply_quarantine is on but the table lacks {sorted(missing)}, "
                "so the bit-corrupt populations cannot be identified. Project "
                "those columns or set apply_quarantine=False deliberately."
            )
        # Every quarantined cycle is on a known day, and `window` is within 3h
        # of the rows' cycle, so a window far from any of them cannot contain
        # one -- checked before materialising the timestamp column.
        near = any(
            abs(
                int(np.datetime64(c, "ns").astype("int64"))
                - int(window.astype("int64"))
            )
            <= 86_400 * 1_000_000_000
            for _, c in QUARANTINED_POPULATIONS
        )
        cycle_ns = (
            table.column("cycle")
            .combine_chunks()
            .cast(pa.timestamp("ns"))
            .to_numpy(zero_copy_only=False)
            .astype("int64")
            if near and QUARANTINED_POPULATIONS
            else np.zeros(n, dtype="int64")
        )
        for want_subtype, want_cycle in QUARANTINED_POPULATIONS:
            on_cycle = cycle_ns == np.datetime64(want_cycle, "ns").astype("int64")
            if not on_cycle.any():
                continue  # ~every window; the string column is never touched
            # Arrow compares the dictionary, not the strings. to_pylist here
            # cost 0.5 s per quarantined window building Python objects only to
            # compare them elementwise.
            same = pc.equal(table.column("ncep_dump_subtype"), want_subtype)
            bad |= on_cycle & same.fill_null(False).to_numpy(zero_copy_only=False)
        fire(Reject.QUARANTINED, bad, "quarantined")

    if "time_utc" in table.schema.names:
        times = (
            table.column("time_utc")
            .combine_chunks()
            .cast(pa.timestamp("ns"))
            .to_numpy(zero_copy_only=False)
        )
        times = np.asarray(times, dtype="datetime64[ns]")
    else:
        times = np.full(n, np.datetime64("NaT", "ns"))

    elementwise_flags = {
        "structural": int(Reject.STRUCTURAL),
        "out_of_window": int(Reject.OUT_OF_WINDOW),
        "bad_swcm": int(Reject.BAD_SWCM),
        "pressure_floor": int(Reject.PRESSURE_FLOOR),
        "zenith_limb": int(Reject.ZENITH_LIMB),
        "speed_implausible": int(Reject.SPEED_IMPLAUSIBLE),
        "cawv_slow": int(Reject.CAWV_SLOW),
        "layer_wind": int(Reject.LAYER_WIND),
    }
    # fmt: off
    elementwise_reject, elementwise_diagnostics = satwnd_kernels.apply_elementwise_screens(
        lat=lat, lon=lon, pressure=pressure, u=u, v=v, swcm=swcm, zenith=zenith,
        istype=istype, times=times, report_type=report_type,
        geo_lookup=_case_lookup(GEO_LIMB_CASES), flags=elementwise_flags, config=config,
        window_ns=int(window.astype("int64")),
        window_half_width_hours=window_half_width_hours,
        swcm_min=SWCM_MIN, swcm_max=SWCM_MAX, layer_swcm_min=LAYER_SWCM_MIN,
        cawv_type=CAWV_TYPE, cawv_istype=CAWV_ISTYPE, cawv_min_speed=CAWV_MIN_SPEED,
    )
    # fmt: on
    reject |= elementwise_reject
    counts.update(
        {
            name: int(np.count_nonzero(elementwise_reject & np.uint32(flag)))
            for name, flag in elementwise_flags.items()
        }
    )
    diagnostics.update(elementwise_diagnostics)

    # --- 6. QI, product-aware, GNAP-resolved --------------------------------
    if config.apply_qifn:
        qifn, _qify, _ee, resolvable, diverged = resolve_qi(table, istype)
        diagnostics["gsi_position_divergence"] = diverged
        legacy = _in_cases(istype, _case_lookup(QI85_CASES))
        goesr = _in_cases(istype, _case_lookup(GOESR_CASES))
        exempt = np.isin(report_type, list(QI_EXEMPT_TYPES))
        screened = (legacy | goesr) & ~exempt
        # A row we mean to screen but cannot evaluate is recorded, not passed
        # silently and not rejected.
        diagnostics["qi_unavailable"] = int(np.count_nonzero(screened & ~resolvable))

        bad = np.zeros(n, dtype=bool)
        bad |= legacy & ~exempt & resolvable & (qifn < 85.0)
        bad |= goesr & resolvable & ((qifn < GOESR_QI_MIN) | (qifn > GOESR_QI_MAX))
        fire(Reject.QI_LOW, bad, "qi_low")

    # --- 9. exact duplicates -------------------------------------------------
    if config.deduplicate:
        alive = reject == 0
        bad = np.zeros(n, dtype=bool)
        if alive.any():
            idx = np.flatnonzero(alive)
            key = np.stack(
                [
                    times[idx].astype("int64").astype("float64"),
                    lat[idx],
                    lon[idx],
                    pressure[idx],
                    u[idx],
                    v[idx],
                    report_type[idx].astype("float64"),
                ],
                axis=1,
            )
            key = np.nan_to_num(key, nan=-9.99e18)
            # Hash-based, NOT np.unique(axis=0). That call lexsorts the whole
            # of the loader -- to find duplicates that exist only in 2005 and
            # number zero in every later era. `duplicated` is a hash table,
            # O(n) rather than O(n log n), with the same keep-first semantics
            # np.unique(return_index=True) had.
            dup = pd.DataFrame(key).duplicated(keep="first").to_numpy()
            bad[idx[dup]] = True
        fire(Reject.DUPLICATE, bad, "duplicate")

    kept = int(np.count_nonzero(reject == 0))
    return QCResult(
        reject=reject,
        counts=counts,
        diagnostics=diagnostics,
        disabled=frozenset(disabled),
        kept=kept,
        total=n,
    )
