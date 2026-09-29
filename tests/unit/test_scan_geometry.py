# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan angle, scan side, and the numbering the keep ranges are written in."""

import numpy as np
import pytest

from healda.observations.preprocessing import scan_geometry as geometry

LINEAR = ("atms", "mhs", "amsua", "amsub", "airs")


@pytest.mark.parametrize("sensor", LINEAR)
def test_linear_scanners_are_symmetric_about_nadir(sensor):
    config = geometry.SCAN_GEOMETRY[sensor]
    positions = np.arange(1, config.positions + 1)
    angle = geometry.scan_angle(sensor, positions)
    assert angle[0] == pytest.approx(-angle[-1], abs=0.11)
    assert np.all(np.diff(angle) > 0)


def test_iasi_pairs_source_fovs_onto_one_scan_angle():
    fov = np.arange(1, 121)
    position = geometry.scan_position("iasi", fov)
    angle = geometry.scan_angle("iasi", fov)
    assert position[0] == 1 and position[-1] == 60
    # Both source FOVs of a pair share a position and so share a scan angle.
    assert np.array_equal(position[::2], position[1::2])
    assert angle[::2] == pytest.approx(angle[1::2])
    # Detector offset alternates about the EFOV centre, positive on odd positions.
    centre = geometry.SCAN_GEOMETRY["iasi"].start_deg
    assert angle[0] == pytest.approx(centre + geometry.IASI_DETECTOR_OFFSET_DEG)
    assert angle[2] == pytest.approx(centre - geometry.IASI_DETECTOR_OFFSET_DEG)


def test_cris_centre_detector_has_no_offset_and_the_rest_rotate():
    regard = np.arange(1, 31)
    config = geometry.SCAN_GEOMETRY["cris"]
    assert geometry.scan_angle("cris", np.full(30, 5), regard) == pytest.approx(
        config.centre(regard)
    )
    corner = geometry.scan_angle("cris", np.ones(30, int), regard) - config.centre(
        regard
    )
    assert np.abs(corner).max() == pytest.approx(
        np.rad2deg(geometry.CRIS_DETECTOR_DISTANCE_RAD[0]), abs=0.01
    )
    assert corner.min() < 0 < corner.max()


def test_signed_zenith_splits_the_swath_in_half_and_keeps_the_magnitude():
    for sensor, config in geometry.SCAN_GEOMETRY.items():
        # CrIS numbers scan position by field of regard; IASI runs four IFOVs per EFOV.
        regard = None
        if sensor == "cris":
            fov = np.full(config.positions, 5)
            regard = np.arange(1, config.positions + 1)
        else:
            fov = np.arange(1, 121 if sensor == "iasi" else config.positions + 1)
        zenith = np.linspace(1.0, 60.0, len(fov))
        signed = geometry.signed_zenith(sensor, fov, zenith, regard)
        assert (signed < 0).sum() == (signed > 0).sum(), sensor
        np.testing.assert_array_equal(np.abs(signed), zenith)


def test_keep_range_is_restated_in_the_column_that_gets_screened():
    # Only IASI numbers scan positions differently from the archive's field_of_view.
    assert geometry.keep_range_in_source_units("amsua", (4, 27)) == (4, 27)
    assert geometry.keep_range_in_source_units("iasi", (5, 56)) == (9, 112)
    fov = np.arange(1, 121)
    low, high = geometry.keep_range_in_source_units("iasi", (5, 56))
    kept = geometry.scan_position("iasi", fov[(fov >= low) & (fov <= high)])
    assert kept.min() == 5 and kept.max() == 56
