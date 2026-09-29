# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed cross-track scan angle for the cross-track sounders in the archive.

The archive carries a one-based source scan position and an unsigned satellite zenith angle,
never a scan angle. These are distinct fields and are easy to conflate:

  source scan position    one-based instrument sample the source encodes
  model scan position     what GSI writes as diagnostic Scan_Position, not always the source one
  scan centre angle       spacecraft-relative angle of the scan or field-of-regard centre
  detector offset         cross-track contribution of a detector's place in a multi-detector look
  scan angle              scan centre angle plus detector offset
  satellite zenith angle  ray angle at the footprint relative to local vertical, unsigned here
  satellite azimuth       geographic direction of that ray about the local vertical

Sign is spacecraft-relative: negative on the first half of the scan, the left side looking
forward along the flight direction, per the EUMETSAT ATOVS record order. On a northbound pass
that is roughly west and it reverses going south, so the sign does not encode west/east,
ascending/descending, day/night, or azimuth. GSI signs a copy of the zenith angle by scan side
and leaves the source magnitude alone, which is what the archive stores.

MHS and AMSU-A alone use the native instrument fractions rather than the rounded
`global_scaninfo.txt` the UFS replay pinned (e.g. 10/3 vs 3.333); every other sensor's pinned values are already exact.
The two differ by up to 8.4e-3 deg at the swath edge.

Constructions follow GSI: `read_atms.f90`, `read_bufrtovs.f90` for MHS and the AMSUs,
`read_airs.f90`, `read_iasi.f90` and `read_cris.f90`, with the two angles written separately in
`setuprad.f90`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossTrackScan:
    """Pinned scan configuration; the centre angle is `start_deg + (step - 1) * step_deg`.

    `positions` counts model scan positions and `keep` is the inclusive range of them the replay
    retains, dropping the most oblique looks. `detectors` is how many archive samples share one
    cross-track look, which is what separates the source field-of-view numbering from the model
    scan position.
    """

    positions: int
    start_deg: float
    step_deg: float
    keep: tuple[int, int]
    detectors: int = 1

    def centre(self, position) -> np.ndarray:
        return (
            self.start_deg
            + (np.asarray(position, dtype=np.float64) - 1.0) * self.step_deg
        )


SCAN_GEOMETRY = {
    "atms": CrossTrackScan(
        positions=96, start_deg=-52.725, step_deg=1.110, keep=(7, 90)
    ),
    # MHS samples 90 beams of 10/9 deg from -445/9; AMSU-A 30 of 10/3 from -48-1/3.
    "mhs": CrossTrackScan(
        positions=90, start_deg=-445 / 9, step_deg=10 / 9, keep=(10, 81)
    ),
    "amsua": CrossTrackScan(
        positions=30, start_deg=-48 - 1 / 3, step_deg=10 / 3, keep=(4, 27)
    ),
    "amsub": CrossTrackScan(
        positions=90, start_deg=-48.950, step_deg=1.100, keep=(10, 81)
    ),
    "airs": CrossTrackScan(
        positions=90, start_deg=-48.900, step_deg=1.100, keep=(10, 81)
    ),
    "iasi": CrossTrackScan(
        positions=60, start_deg=-48.330, step_deg=3.334, keep=(5, 56), detectors=2
    ),
    "cris": CrossTrackScan(
        positions=30, start_deg=-48.330, step_deg=3.3331, keep=(1, 30), detectors=9
    ),
}

# IASI reads out 30 EFOVs of four IFOVs each, which NCEP flattens into FOVN 1..120. GSI pairs
# them onto 60 model positions and gives both members of a pair the same scan angle, offset
# either side of the EFOV centre by this much, positive when the model position is odd. Fitting
# the offset against the archive's own encoded look angles gives 0.619, so this is right to
# 0.006 deg.
IASI_DETECTOR_OFFSET_DEG = 0.625

# CrIS reads out a 3x3 array, FOV 5 at the field-of-regard centre, arranged with the spacecraft
# velocity upward as 7 8 9 / 4 5 6 / 1 2 3. The array rotates across the scan, so a detector's
# cross-track contribution is distance * sin(phase - (regard - 1) * step) rather than one fixed
# offset. Corners sit 1.556 deg from the centre, edge centres 1.100 deg.
# fmt: off
CRIS_DETECTOR_DISTANCE_RAD = np.array(
    [0.0271510, 0.0191986, 0.0271510,
     0.0191986, 0.0,       0.0191986,
     0.0271510, 0.0191986, 0.0271510]
)
CRIS_DETECTOR_PHASE_RAD = np.array(
    [4.77057, 3.98517, 3.19977,
     5.55597, 0.0,     2.41437,
     0.05818, 0.84358, 1.62897]
)
# fmt: on


def scan_position(sensor: str, field_of_view, field_of_regard=None) -> np.ndarray:
    """The model scan position GSI writes as diagnostic Scan_Position."""
    if sensor == "cris":
        if field_of_regard is None:
            raise ValueError("cris scan position is field_of_regard, not field_of_view")
        return np.asarray(field_of_regard).astype(np.int64)
    field_of_view = np.asarray(field_of_view).astype(np.int64)
    if sensor == "iasi":
        return (field_of_view - 1) // 2 + 1
    return field_of_view


def detector_index(sensor: str, field_of_view) -> np.ndarray:
    """One-based place within the detector group sharing one cross-track look."""
    field_of_view = np.asarray(field_of_view).astype(np.int64)
    if sensor == "cris":
        return field_of_view
    return (field_of_view - 1) % SCAN_GEOMETRY[sensor].detectors + 1


def scan_angle(sensor: str, field_of_view, field_of_regard=None) -> np.ndarray:
    """Signed scan angle in degrees, negative on the first half of the scan."""
    geometry = SCAN_GEOMETRY[sensor]
    if sensor == "iasi":
        field_of_view = np.asarray(field_of_view).astype(np.int64)
        efov = (field_of_view - 1) // 4
        odd = scan_position(sensor, field_of_view) % 2 == 1
        offset = np.where(odd, 1.0, -1.0) * IASI_DETECTOR_OFFSET_DEG
        return geometry.start_deg + efov * geometry.step_deg + offset
    if sensor == "cris":
        regard = scan_position(sensor, field_of_view, field_of_regard)
        detector = detector_index(sensor, field_of_view) - 1
        twist = CRIS_DETECTOR_DISTANCE_RAD[detector] * np.sin(
            CRIS_DETECTOR_PHASE_RAD[detector]
            - np.deg2rad((regard - 1) * geometry.step_deg)
        )
        return geometry.centre(regard) + np.rad2deg(twist)
    return geometry.centre(scan_position(sensor, field_of_view))


def scan_side(sensor: str, field_of_view, field_of_regard=None) -> np.ndarray:
    """-1 on the first half of the scan, +1 on the second."""
    position = scan_position(sensor, field_of_view, field_of_regard)
    return np.where(position * 2 <= SCAN_GEOMETRY[sensor].positions, -1.0, 1.0)


def signed_zenith(
    sensor: str, field_of_view, zenith, field_of_regard=None
) -> np.ndarray:
    """The archive's unsigned zenith magnitude placed on the side of the swath it came from."""
    side = scan_side(sensor, field_of_view, field_of_regard)
    return side * np.abs(np.asarray(zenith, dtype=np.float64))


##### Alternative calculations for the scan angle
# Ratio Re / (Re + h) that GSI hard-codes for the CrIS geometry check, the reciprocal of the
# 1.1363987 the common ATOVS reader uses. ATMS prefers the encoded orbit height and falls back to
# 824 km; IASI uses the encoded height throughout.
CRIS_EARTH_RATIO = 0.87997285

EARTH_RADIUS_M = 6_378_137.0

# Spacecraft height per footprint, in metres, present on every footprint of every sensor. AIRS and
# IASI carry it under an elevation descriptor the source reuses, so despite the name these are
# spacecraft altitudes (about 720 km for Aqua, 823 km for Metop) and not terrain.
HEIGHT_COLUMN = {
    "atms": "orbit_height",
    "mhs": "orbit_height",
    "amsua": "orbit_height",
    "amsub": "orbit_height",
    "cris": "orbit_height",
    "airs": "station_elevation",
    "iasi": "station_elevation",
}


def spherical_off_nadir_angle_from_zenith(
    sensor: str, field_of_view, zenith, height, field_of_regard=None
) -> np.ndarray:
    """Off-nadir angle closing the Earth-satellite triangle from zenith and height, in degrees.

    Zenith and off-nadir angle are the two endpoint angles of one Earth-satellite ray, so for a
    spherical Earth the law of sines gives `asin(R / (R + h) * sin(z))`. Height is in metres and
    every sensor supplies it per footprint under `HEIGHT_COLUMN`.

    A consistency check against `scan_angle`, which is what a model should read.
    """
    ratio = EARTH_RADIUS_M / (EARTH_RADIUS_M + np.asarray(height, dtype=np.float64))
    sine = ratio * np.sin(np.deg2rad(np.abs(np.asarray(zenith, dtype=np.float64))))
    return scan_side(sensor, field_of_view, field_of_regard) * np.rad2deg(
        np.arcsin(np.clip(sine, -1.0, 1.0))
    )


def keep_range_in_source_units(sensor: str, keep: tuple[int, int]) -> tuple[int, int]:
    """A model scan position range restated against the archive's own scan-identity column.

    Model position is monotone in the source numbering, so the range stays contiguous and a
    screen can compare the raw column without reconstructing the position per row.
    """
    detectors = SCAN_GEOMETRY[sensor].detectors if sensor == "iasi" else 1
    low, high = keep
    return (low - 1) * detectors + 1, high * detectors
