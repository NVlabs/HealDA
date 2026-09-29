# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SATWND QC: per-family dispatch, GNAP-resolved QI, and the order assertion."""

import numpy as np
import pyarrow as pa
import pytest

from healda.observations.preprocessing import satwnd_qc as qc

WINDOW = np.datetime64("2019-08-01T12:00:00", "ns")

QD_TYPE = pa.list_(
    pa.struct(
        [
            ("per_cent_confidence", pa.float64()),
            ("generating_application", pa.uint32()),
            ("standard_generating_application", pa.uint32()),
        ]
    )
)


def _qd(rows):
    """rows: list of [(pccf, gnap, gnaps), ...] per observation."""
    return pa.array(
        [
            [
                {
                    "per_cent_confidence": p,
                    "generating_application": g,
                    "standard_generating_application": s,
                }
                for p, g, s in occ
            ]
            for occ in rows
        ],
        type=QD_TYPE,
    )


def _table(
    n=1,
    istype=7,
    pressure_hpa=500.0,
    swcm=1,
    zenith=10.0,
    u=10.0,
    v=0.0,
    qd=None,
    times=None,
    subtype="NC005030",
    cycle=WINDOW,
):
    def col(value, dtype):
        return pa.array(
            [value] * n if not isinstance(value, list) else value, type=dtype
        )

    data = {
        "time_utc": pa.array(times or [WINDOW] * n, type=pa.timestamp("ns")),
        # The quarantine screen keys on these two and refuses to run without
        # them, so every fixture carries them; the defaults are a clean
        # population that no quarantine entry matches.
        "ncep_dump_subtype": col(subtype, pa.string()),
        "cycle": pa.array([cycle] * n, type=pa.timestamp("ns")),
        "latitude": col(0.0, pa.float64()),
        "longitude": col(0.0, pa.float64()),
        "assigned_pressure": col(pressure_hpa * 100.0, pa.float64()),
        "wind_u_derived": col(u, pa.float64()),
        "wind_v_derived": col(v, pa.float64()),
        "wind_computation_method": col(swcm, pa.float64()),
        "satellite_zenith_angle": col(zenith, pa.float64()),
        "gsi_internal_subtype": col(istype, pa.float64()),
    }
    if qd is not None:
        data["quality_diagnostics"] = qd
    return pa.table(data)


def _types(n, value=245):
    return np.full(n, value, dtype=np.int64)


# --------------------------------------------------------------------------
# QI resolution -- the leak this module exists to prevent
# --------------------------------------------------------------------------


def test_qifn_and_qify_differ_for_eumetsat_which_is_the_leak():
    # EUMETSAT (istype 1): GNAP 2 is qifn, GNAP 1 is qify. The promoted archive
    # scalar takes the FIRST occurrence -- GNAP 1 -- so it is the with-forecast
    # value. Resolving by GNAP must pick 91, not 42.
    table = _table(
        istype=1, qd=_qd([[(42.0, 1, None), (91.0, 2, None), (7.0, 3, None)]])
    )
    qifn, qify, _, ok, _ = qc.resolve_qi(table, np.array([1]))
    assert ok[0]
    assert qifn[0] == 91.0
    assert qify[0] == 42.0
    assert qifn[0] != qify[0]


def test_gnap1_means_opposite_things_for_eumetsat_and_nesdis():
    # The whole reason dispatch is per-family: the same code, opposite meaning.
    occ = [[(42.0, 1, None), (91.0, 2, None), (7.0, 3, None)]]
    eumetsat, _, _, _, _ = qc.resolve_qi(_table(istype=1, qd=_qd(occ)), np.array([1]))
    nesdis, _, _, _, _ = qc.resolve_qi(_table(istype=7, qd=_qd(occ)), np.array([7]))
    assert eumetsat[0] == 91.0  # GNAP 2
    assert nesdis[0] == 42.0  # GNAP 1


def test_jma_uses_the_101_102_numbering():
    table = _table(istype=4, qd=_qd([[(55.0, 101, None), (88.0, 102, None)]]))
    qifn, qify, _, ok, _ = qc.resolve_qi(table, np.array([4]))
    assert ok[0] and qifn[0] == 88.0 and qify[0] == 55.0


def test_amvqic_resolves_by_gnaps_not_position():
    # EUMETSAT 2023 layout [6,5,4,2]: GNAPS 5 is qifn, GNAPS 2 is ee.
    table = _table(
        istype=2,
        qd=_qd([[(75.0, None, 6), (80.0, None, 5), (53.0, None, 4), (62.0, None, 2)]]),
    )
    qifn, qify, ee, ok, diverged = qc.resolve_qi(table, np.array([2]))
    assert ok[0] and qifn[0] == 80.0 and ee[0] == 62.0
    assert np.isnan(qify[0])  # AMVQIC carries no with-forecast variant
    assert diverged == 0  # GNAPS 5 happens to sit in GSI's slot here


def test_amvqic_ascending_layout_still_resolves():
    # METOP / GOES-R 2023 layout [4,5,7] is ASCENDING. A descending-order
    # assumption would have rejected this; semantic selection handles it.
    table = _table(
        istype=10, qd=_qd([[(30.0, None, 4), (88.0, None, 5), (12.0, None, 7)]])
    )
    qifn, _, _, ok, diverged = qc.resolve_qi(table, np.array([10]))
    assert ok[0] and qifn[0] == 88.0 and diverged == 0


def test_layout_where_gsi_would_read_the_wrong_slot_is_counted():
    # NC005091 (istype 14) is [5,7]: GNAPS 5 is at position 1, so GSI's
    # amvqic(2,2) reads GNAPS 7. We take 5 and RECORD the divergence.
    table = _table(istype=14, qd=_qd([[(91.0, None, 5), (4.0, None, 7)]]))
    qifn, _, _, ok, diverged = qc.resolve_qi(table, np.array([14]))
    assert ok[0] and qifn[0] == 91.0
    assert diverged == 1


def test_two_qifn_occurrences_is_ambiguous_and_raises():
    table = _table(istype=2, qd=_qd([[(70.0, None, 5), (80.0, None, 5)]]))
    with pytest.raises(qc.AmvqicLayoutError, match="more than one"):
        qc.resolve_qi(table, np.array([2]))


def test_amvqic_without_gnaps_is_unresolvable():
    # 2019 GOES-R carries GNAPS null; the screen must skip, not guess.
    table = _table(istype=15, qd=_qd([[(75.0, None, None), (80.0, None, None)]]))
    _, _, _, ok, _ = qc.resolve_qi(table, np.array([15]))
    assert not ok[0]


def test_family_without_a_qifn_gnap_is_unresolvable_not_substituted():
    # LEOGEO (istype 12) has no qifn GNAP. The value present must NOT be used.
    table = _table(istype=12, qd=_qd([[(99.0, 1, None)]]))
    qifn, _, _, ok, _ = qc.resolve_qi(table, np.array([12]))
    assert not ok[0] and np.isnan(qifn[0])


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------


def test_pressure_floor_and_structural_reject():
    table = _table(n=2, pressure_hpa=100.0)  # above 125 hPa
    result = qc.evaluate(table, WINDOW, _types(2))
    assert result.counts["pressure_floor"] == 2
    assert result.kept == 0
    assert result.reject[0] & int(qc.Reject.PRESSURE_FLOOR)


def test_zenith_screen_is_geo_only():
    # istype 7 is GEO -> screened. istype 8 (MODIS, LEO) -> not screened, and
    # that is what makes the signed-angle problem moot rather than needing abs().
    geo = qc.evaluate(_table(istype=7, zenith=75.0), WINDOW, _types(1))
    leo = qc.evaluate(_table(istype=8, zenith=75.0), WINDOW, _types(1))
    assert geo.counts["zenith_limb"] == 1
    assert leo.counts["zenith_limb"] == 0
    assert leo.diagnostics["zenith_not_applicable"] == 1


def test_negative_zenith_is_never_rejected_and_is_not_abs_ed():
    # LEO cross-track angles are signed; abs() would invent a screen GSI does
    # not perform. Even for a GEO family a negative value must pass a > test.
    result = qc.evaluate(_table(istype=7, zenith=-75.0), WINDOW, _types(1))
    assert result.counts["zenith_limb"] == 0


def test_layer_winds_excluded_by_default_and_toggleable():
    on = qc.evaluate(_table(swcm=5), WINDOW, _types(1))
    assert on.counts["layer_wind"] == 1 and on.kept == 0
    off = qc.evaluate(
        _table(swcm=5), WINDOW, _types(1), qc.SatwndQCConfig(exclude_layer_winds=False)
    )
    assert off.counts["layer_wind"] == 0 and off.kept == 1
    assert "layer_wind" in off.disabled and "layer_wind" not in on.disabled
    assert off.counts.get("layer_wind", 0) == 0


def test_cawv_slow_rejects_only_type_247():
    slow = qc.evaluate(_table(u=5.0, v=0.0), WINDOW, _types(1, 247))
    fast = qc.evaluate(_table(u=25.0, v=0.0), WINDOW, _types(1, 247))
    other = qc.evaluate(_table(u=5.0, v=0.0), WINDOW, _types(1, 245))
    assert slow.counts["cawv_slow"] == 1
    assert fast.counts["cawv_slow"] == 0
    assert other.counts["cawv_slow"] == 0


def test_cawv_slow_is_scoped_to_istype_7_not_every_type_247():
    # GSI's rule sits inside case(7). Type 247 also arises from istype 6 via
    # the Defect A `ihdr9 >= 4` route, and case(6) carries no speed rule --
    # so screening on the type alone rejects data the global system keeps.
    seven = qc.evaluate(_table(istype=7, u=5.0, v=0.0), WINDOW, _types(1, 247))
    six = qc.evaluate(_table(istype=6, u=5.0, v=0.0), WINDOW, _types(1, 247))
    assert seven.counts["cawv_slow"] == 1
    assert six.counts["cawv_slow"] == 0


def test_type_247_is_exempt_from_the_qi_screen():
    config = qc.SatwndQCConfig(apply_qifn=True, apply_cawv_speed=False)
    qd = _qd([[(10.0, 1, None)]])  # far below 85
    exempt = qc.evaluate(_table(istype=7, qd=qd), WINDOW, _types(1, 247), config)
    screened = qc.evaluate(_table(istype=7, qd=qd), WINDOW, _types(1, 245), config)
    assert exempt.counts["qi_low"] == 0
    assert screened.counts["qi_low"] == 1


def test_goesr_uses_80_and_an_upper_bound_not_85():
    config = qc.SatwndQCConfig(apply_qifn=True)
    # 82 passes GOES-R (>=80) but would fail the legacy 85 rule.
    ok = qc.evaluate(
        _table(istype=15, qd=_qd([[(1.0, None, 6), (82.0, None, 5)]])),
        WINDOW,
        _types(1, 245),
        config,
    )
    assert ok.counts["qi_low"] == 0
    # >100 is rejected, which the legacy rule would not catch.
    high = qc.evaluate(
        _table(istype=15, qd=_qd([[(1.0, None, 6), (127.0, None, 5)]])),
        WINDOW,
        _types(1, 245),
        config,
    )
    assert high.counts["qi_low"] == 1


def test_unevaluable_qi_is_recorded_not_silently_passed():
    config = qc.SatwndQCConfig(apply_qifn=True)
    table = _table(istype=7, qd=_qd([[(50.0, 99, None)]]))  # no qifn GNAP
    result = qc.evaluate(table, WINDOW, _types(1, 245), config)
    assert result.counts["qi_low"] == 0
    assert "qi_unavailable" not in result.counts
    assert result.diagnostics["qi_unavailable"] == 1
    assert result.kept == 1  # recorded, not rejected


def test_speed_cap_rejects_the_2005_garbage_cluster():
    result = qc.evaluate(_table(u=300.0, v=0.0), WINDOW, _types(1))
    assert result.counts["speed_implausible"] == 1


def test_duplicates_removed_keeping_one():
    # OFF by default -- thinning already collapses exact duplicates, measured
    # identical on five windows including the 2005 cycles that carry 13,444 of
    # them. The screen is retained only to COUNT, so it must be asked for.
    table = _table(n=3)
    assert qc.evaluate(table, WINDOW, _types(3)).counts["duplicate"] == 0
    config = qc.SatwndQCConfig(deduplicate=True)
    result = qc.evaluate(table, WINDOW, _types(3), config)
    assert result.counts["duplicate"] == 2 and result.kept == 1


def test_duplicate_screen_is_off_by_default_and_reports_as_disabled():
    # "rejected nothing" and "was not run" must stay distinguishable.
    result = qc.evaluate(_table(n=3), WINDOW, _types(3))
    assert "duplicate" in result.disabled


def test_out_of_window_rejected():
    late = np.datetime64("2019-08-01T20:00:00", "ns")
    result = qc.evaluate(_table(times=[late]), WINDOW, _types(1))
    assert result.counts["out_of_window"] == 1


def test_every_screen_is_individually_disableable():
    # A config with everything off must keep a row that fails many screens.
    table = _table(istype=7, pressure_hpa=50.0, swcm=5, zenith=80.0, u=300.0)
    config = qc.SatwndQCConfig(
        require_structural=False,
        require_in_window=False,
        require_valid_swcm=False,
        pressure_floor_hpa=None,
        zenith_limit_deg=None,
        apply_qifn=False,
        apply_cawv_speed=False,
        maximum_wind_speed=None,
        exclude_layer_winds=False,
        deduplicate=False,
        apply_quarantine=False,
    )
    result = qc.evaluate(table, WINDOW, _types(1), config)
    assert result.kept == 1 and result.reject[0] == 0
    # every screen reports as DISABLED, not as "rejected nothing"
    assert set(qc.SCREEN_NAMES) <= result.disabled
    assert all(v == 0 for v in result.counts.values())


def test_bitmask_composes_when_several_screens_fire():
    table = _table(istype=7, pressure_hpa=50.0, swcm=5, zenith=80.0)
    result = qc.evaluate(table, WINDOW, _types(1))
    flags = result.reject[0]
    assert flags & int(qc.Reject.PRESSURE_FLOOR)
    assert flags & int(qc.Reject.ZENITH_LIMB)
    assert flags & int(qc.Reject.LAYER_WIND)


# --------------------------------------------------------------------------
# quarantine -- identity, not magnitude
# --------------------------------------------------------------------------


def test_quarantine_catches_corrupt_rows_a_speed_cap_cannot():
    # The NC005064 lattice starts at 25.4 m/s. That is a perfectly ordinary
    # wind: the 160 m/s cap passes it, and so does every other value screen.
    # Only the identity screen removes it.
    bad_cycle = np.datetime64("2005-06-07T12:00:00", "ns")
    table = _table(
        subtype="NC005064", cycle=bad_cycle, u=25.4, v=0.0, times=[bad_cycle]
    )
    result = qc.evaluate(table, bad_cycle, _types(1, 245))
    assert result.counts["quarantined"] == 1
    assert result.counts["speed_implausible"] == 0
    assert result.kept == 0


def test_quarantine_is_scoped_to_the_affected_cycles_and_subtype():
    bad_cycle = np.datetime64("2005-06-07T12:00:00", "ns")
    good_cycle = np.datetime64("2005-06-07T00:00:00", "ns")
    same_subtype_clean_cycle = _table(
        subtype="NC005064", cycle=good_cycle, times=[good_cycle]
    )
    other_subtype_bad_cycle = _table(
        subtype="NC005030", cycle=bad_cycle, times=[bad_cycle]
    )
    assert (
        qc.evaluate(same_subtype_clean_cycle, good_cycle, _types(1, 245)).counts[
            "quarantined"
        ]
        == 0
    )
    assert (
        qc.evaluate(other_subtype_bad_cycle, bad_cycle, _types(1, 245)).counts[
            "quarantined"
        ]
        == 0
    )


def test_quarantine_refuses_to_run_blind():
    # Silently skipping is the failure this screen exists to prevent, so a
    # table that cannot be identified must raise rather than pass everything.
    table = _table().drop_columns(["cycle"])
    with pytest.raises(ValueError, match="apply_quarantine"):
        qc.evaluate(table, WINDOW, _types(1, 245))


def test_quarantine_survives_a_window_in_another_time_unit():
    """An int64 compare against a non-ns window mis-scales and skips the screen."""
    cycle = np.datetime64("2005-06-07T12:00:00", "ns")
    table = _table(subtype="NC005064", cycle=cycle)
    fine = qc.evaluate(table, cycle, _types(1))
    coarse = qc.evaluate(table, cycle.astype("datetime64[s]"), _types(1))
    assert fine.counts["quarantined"] == coarse.counts["quarantined"] == 1
