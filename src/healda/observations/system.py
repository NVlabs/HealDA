# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime shape of an observing system: where rows come from, which are filtered
out, which are randomly dropped, and the named report-type groups a filter can address.

``ObsConfig`` is the flat record serialized into checkpoints; this is what the loaders
are built from.

Filters are deterministic: the recipe's, plus whatever a run adds (``extra`` to
``ObsPipeline``; ``inference_filters`` is the scoring default). Random drops are a training
regulariser and are off when not training.
"""

import csv
import dataclasses
import enum
import pathlib

import pandas as pd

from healda.config.models import ObsConfig
from healda.observations.preprocessing.ncep_report_types import observation_type_tab

__all__ = [
    "ChannelDenial",
    "FAMILY_REPORT_TYPES",
    "ObsFamily",
    "ObsFilters",
    "ObsPipeline",
    "ObsRandomDrop",
    "inference_filters",
    "load_denials",
    "report_types_for",
]


# --------------------------------------------------------------------------------------
# Named groups of PrepBUFR report types, so a filter can say "amv" not 240..260. Codes
# come from ncep_report_types, NCEP's own mnemonic table, so there is one source for
# them. A family is one mnemonic unless the merge is stated below.
# --------------------------------------------------------------------------------------


class ObsFamily(enum.StrEnum):
    RADIOSONDE = "radiosonde"  # ADPUPA
    AIRCRAFT = "aircraft"  # AIRCAR + AIRCFT, split only by reporting system
    AMV = "amv"  # SATWND
    PROFILER = "profiler"  # PROFLR
    VAD = "vad"  # VADWND
    RASS = "rass"  # RASSDA
    SURFACE_LAND = "surface_land"  # ADPSFC
    SURFACE_MARINE = "surface_marine"  # SFCSHP
    MESONET = "mesonet"  # MSONET
    SCATTEROMETER = "scatterometer"  # ASCATW + QKSWND + ERS1DA + WDSATR
    SATELLITE_RETRIEVAL = "satellite_retrieval"  # SPSSMI + GOESND + SATEMP
    GPS_PRECIPITABLE_WATER = "gps_precipitable_water"  # GPSIPW
    BOGUS = "bogus"  # SYNDAT + SFCBOG


_MNEMONIC_TO_FAMILY = {
    "ADPUPA": ObsFamily.RADIOSONDE,
    "AIRCAR": ObsFamily.AIRCRAFT,
    "AIRCFT": ObsFamily.AIRCRAFT,
    "SATWND": ObsFamily.AMV,
    "PROFLR": ObsFamily.PROFILER,
    "VADWND": ObsFamily.VAD,
    "RASSDA": ObsFamily.RASS,
    "ADPSFC": ObsFamily.SURFACE_LAND,
    "SFCSHP": ObsFamily.SURFACE_MARINE,
    "MSONET": ObsFamily.MESONET,
    "ASCATW": ObsFamily.SCATTEROMETER,
    "QKSWND": ObsFamily.SCATTEROMETER,
    "ERS1DA": ObsFamily.SCATTEROMETER,
    "WDSATR": ObsFamily.SCATTEROMETER,
    "SPSSMI": ObsFamily.SATELLITE_RETRIEVAL,
    "GOESND": ObsFamily.SATELLITE_RETRIEVAL,
    "SATEMP": ObsFamily.SATELLITE_RETRIEVAL,
    "GPSIPW": ObsFamily.GPS_PRECIPITABLE_WATER,
    "SYNDAT": ObsFamily.BOGUS,
    "SFCBOG": ObsFamily.BOGUS,
}


def _build_families() -> dict[ObsFamily, frozenset[int]]:
    codes: dict[ObsFamily, set[int]] = {family: set() for family in ObsFamily}
    table = observation_type_tab
    seen: set[int] = set()
    for code, mnemonic in zip(
        table.column("code").to_pylist(), table.column("mnemonic").to_pylist()
    ):
        if code is None:  # the table carries one such row
            continue
        family = _MNEMONIC_TO_FAMILY.get(mnemonic)
        if family is None:
            raise ValueError(f"mnemonic {mnemonic!r} (type {code}) has no family")
        # One code per row is what keeps families disjoint, and a denial unambiguous.
        if int(code) in seen:
            raise ValueError(f"report type {code} appears under two mnemonics")
        seen.add(int(code))
        codes[family].add(int(code))
    return {family: frozenset(members) for family, members in codes.items()}


FAMILY_REPORT_TYPES = _build_families()


def report_types_for(families) -> frozenset[int]:
    return frozenset().union(
        *(FAMILY_REPORT_TYPES[ObsFamily(family)] for family in families)
    )


# --------------------------------------------------------------------------------------
# Dated satellite channel denials; the packaged list is what scoring applies by default.
# --------------------------------------------------------------------------------------

DENIALS_CSV = pathlib.Path(__file__).with_name("denials.csv")


@dataclasses.dataclass(frozen=True)
class ChannelDenial:
    """Drops a (sensor, platform, raw channel)'s observations taken in [start, end).

    Times are UTC ISO strings; None leaves that side open.
    """

    sensor: str
    platform: str
    channel: int
    start: str | None = None
    end: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "channel", int(self.channel))
        start, end = (
            None if t is None else pd.Timestamp(t) for t in (self.start, self.end)
        )
        if any(t is not None and t.tzinfo is not None for t in (start, end)):
            raise ValueError(f"{self}: times are UTC without a zone suffix")
        if start is not None and end is not None and start >= end:
            raise ValueError(f"{self}: start is not before end")


def load_denials(path=DENIALS_CSV) -> tuple[ChannelDenial, ...]:
    with open(path, newline="") as handle:
        return tuple(
            ChannelDenial(
                row["sensor"],
                row["platform"],
                row["channel"],
                row["start"] or None,
                row["end"] or None,
            )
            for row in csv.DictReader(handle)
        )


# --------------------------------------------------------------------------------------
# Filters and random drops.
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ObsFilters:
    """Which observations exist at all, deterministically. Set per run, so a scoring
    run can deny more than training did."""

    # Sensor denial is not here: it has to produce the same list the model's sensor
    # vocabulary is built from, so it lives in tasks.satellite_sensors.
    channel_ids: tuple[int, ...] = ()
    report_types: tuple[int, ...] = ()
    restricted_aircraft: bool = False
    balloon_drift: bool = False
    uv_in_situ_only: bool = False
    gps_level1_only: bool = False
    # Non-GPS conv rows only. GPS-RO keeps its own floor, QCLimits.PRESSURE_MIN_GPS
    # (0.5 hPa), which is hardcoded and not configurable.
    non_gps_min_pressure_hpa: float | None = None
    # NNJA satellites: dated (sensor, platform, channel) denials, and footprints their
    # provider quality flags mark.
    channel_denials: tuple[ChannelDenial, ...] = ()
    drop_source_flagged: bool = False

    def removing_also(self, extra: "ObsFilters") -> "ObsFilters":
        """These filters plus ``extra``'s channel ids, report types, denials and flags."""
        return dataclasses.replace(
            self,
            channel_ids=tuple(sorted(set(self.channel_ids) | set(extra.channel_ids))),
            report_types=tuple(
                sorted(set(self.report_types) | set(extra.report_types))
            ),
            channel_denials=(*self.channel_denials, *extra.channel_denials),
            drop_source_flagged=self.drop_source_flagged or extra.drop_source_flagged,
        )


@dataclasses.dataclass(frozen=True)
class ObsRandomDrop:
    """Rows hidden at random, resampled per window. Training only.

    Carries no seed: the loaders draw from the worker's process-global numpy stream,
    which PyTorch seeds per (rank, worker).
    """

    wind: float = 0.0
    surface_pressure: float = 0.0
    platform_channel: tuple[tuple[str, str, int, float], ...] = ()
    # "row" or "sample"; see ObsConfig.nnja_dropout_scope.
    scope: str = "row"


def inference_filters(
    *,
    denials_csv=DENIALS_CSV,
    extra_denials=(),
    drop_source_flagged: bool = True,
    channel_ids=(),
    report_types=(),
) -> ObsFilters:
    """What scoring removes beyond the recipe: the packaged denials and provider flags
    unless asked otherwise. ``denials_csv`` None applies no file."""
    from_file = () if denials_csv is None else load_denials(denials_csv)
    return ObsFilters(
        channel_ids=tuple(channel_ids),
        report_types=tuple(report_types),
        channel_denials=(*from_file, *extra_denials),
        drop_source_flagged=drop_source_flagged,
    )


class ObsPipeline:
    def __init__(
        self,
        obs_config: ObsConfig,
        *,
        training: bool,
        extra: ObsFilters = ObsFilters(),
    ):
        # Which archive supplies each half. NNJA satellites with UFS conventional is a
        # supported pairing; the reverse is rejected by ObsConfig.
        self.nnja_sat = obs_config.use_nnja_sat
        self.nnja_conv = obs_config.use_nnja_conv
        self.gpsro_saids = obs_config.nnja_gpsro_saids
        self.surface_winds = obs_config.nnja_surface_winds
        self.pressure_height_fill = obs_config.nnja_pressure_height_fill
        self.max_quality_mark = obs_config.nnja_max_quality_mark
        # PrepBUFR AMVs are always dropped; this only adds the dedicated archive.
        self.satwnd = obs_config.use_nnja_satwnd
        self.satwnd_thin_hpx_level = obs_config.nnja_satwnd_thin_hpx_level
        self.filters = ObsFilters(
            channel_ids=tuple(obs_config.drop_obs_channel_ids or ()),
            report_types=tuple(
                sorted(
                    set(obs_config.drop_report_types)
                    | report_types_for(obs_config.drop_obs_families)
                )
            ),
            restricted_aircraft=obs_config.drop_restricted_aircraft,
            balloon_drift=obs_config.nnja_balloon_drift,
            uv_in_situ_only=obs_config.conv_uv_in_situ_only,
            gps_level1_only=obs_config.conv_gps_level1_only,
            non_gps_min_pressure_hpa=obs_config.conv_min_pressure_hpa,
        ).removing_also(extra)
        # The only place `training` is read: it turns off random drops, nothing else.
        self.random_drop = ObsRandomDrop(
            wind=obs_config.nnja_wind_dropout if training else 0.0,
            surface_pressure=(
                obs_config.nnja_surface_pressure_dropout if training else 0.0
            ),
            platform_channel=obs_config.nnja_platform_channel_dropout
            if training
            else (),
            scope=obs_config.nnja_dropout_scope,
        )
