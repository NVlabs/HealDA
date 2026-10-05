# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Earth2Studio NNJA frames as the canonical NNJA archive tables.

``earth2studio.data.NNJAObsConv`` (PrepBUFR ``u``, ``v``, ``q``, ``t``, ``pres``;
GPS-RO ``gps``, ``gps_refractivity``) and ``NNJAObsSatwnd`` (``u``, ``v``) return raw
decoded rows. These functions restate them in the column contract of healda's NNJA
PrepBUFR, GPS-RO and SATWND archive tables, deriving what the ETL derives with the
same code, so the loaders screen, thin and emit them exactly as they do archive rows.
Each returns ``{cycle: table}``, a ``CycleTableSource`` for the loaders'
``table_source`` argument.

``sat_tables`` restates ``NNJAObsSat`` radiances as the wide satellite archive's
rows; ``NNJAWideLoader`` takes them per sensor as its ``table_source``.

Rows are keyed by ``cycle_time``, the NCEP file each was decoded from, which the
Earth2Studio NNJA sources fill: an observation at the shared edge of two files' windows
is in both files, and stays in both, as in the archive.

Usage, for the cycle file around ``cycle`` (dumps cover ``[cycle - 3h, cycle + 3h]``)::

    frame = NNJAObsConv(time_tolerance=window)(cycle, ["gps", "gps_refractivity"])
    loader = NNJAGpsroLoader(table_source=gpsro_tables(frame))
    await loader.sel_time(times)  # or via NNJAConventionalLoader(gpsro_table_source=...)

    conv = NNJAObsConv(time_tolerance=window, original_event=True)
    frame = conv(cycle, ["u", "v", "q", "t", "pres"])
    NNJAConventionalLoader(prepbufr_table_source=prepbufr_tables(frame))

    frame = NNJAObsSat(time_tolerance=window)(cycle, ["atms_antenna_temperature"])
    loader = NNJAWideLoader(["atms"], table_source={"atms": sat_tables(frame, "atms")})
"""

from __future__ import annotations

import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from healda.observations.loaders.nnja_base import archive_window
from healda.observations.loaders.nnja_satwnd import ARCHIVE_HPX_LEVEL
from healda.observations.loaders.nnja_wide import (
    ARCHIVE_HPX_ORDER,
    CRIS_FOV_QUALITY_BAD,
    RADIANCE_PREFIXES,
    SAID_TO_PLATFORM_NAME,
    SCAN_POSITION_COLUMN,
    SENSOR_VALUE_PREFIX,
    _resolve_ir_channels,
)
from healda.observations.preprocessing import ir_spectral, satwnd_gsi_types
from healda.observations.preprocessing.gpsro_pressure import derive_level_coordinates
from healda.observations.preprocessing.sat_helpers import hpx_nest_pixels
from healda.observations.sensors_nnja import SENSOR_CONFIGS

GPS_VARIABLE = "gps"
REFRACTIVITY_VARIABLE = "gps_refractivity"
# e2s variable -> PrepBUFR archive value and quality columns.
PREPBUFR_VARIABLES = {
    "q": ("QOB", "QQM"),
    "t": ("TOB", "TQM"),
    "u": ("UOB", "WQM"),
    "v": ("VOB", "WQM"),
}
PREPBUFR_LEVEL_VARIABLES = (*PREPBUFR_VARIABLES, "pres")
# e2s columns that every observation of one report level shares.
PREPBUFR_LEVEL_COLUMNS = [
    "cycle_time",
    "report_time",
    "station",
    "type",
    "report_lat",
    "report_lon",
    "station_elev",
    "level_cat",
    "pres",
    "elev",
    "pressure_quality",
]

# The NNJAObsSat variable holding each sensor's archive value column
# (SENSOR_VALUE_PREFIX); every other sensor's variable is its own name. The ATMS
# archive column is antenna temperature, which e2s `atms` (brightness temperature) is not.
SAT_VARIABLE = {"atms": "atms_antenna_temperature"}

# The value encoding every CrIS spectral_radiance_code column of the archive states.
CRIS_VALUE_ENCODING = {
    "bit_width": 22,
    "descriptor": "0-14-044",
    "formula": "(code + reference) * 10**(-scale)",
    "id": "cris_srad_014044_s7_r-100000_w22_v1",
    "null_representation": "arrow_null",
    "physical_dtype": "float32",
    "physical_units": "W m^-2 sr^-1 (cm^-1)^-1",
    "reference": -100000,
    "scale": 7,
    "source_missing_code": 4194303,
    "source_units": "W M**-2 SR**-1 CM",
    "storage_dtype": "int32",
}
VALUE_ENCODING = {"cris": CRIS_VALUE_ENCODING}
VALUE_ENCODING_KEY = b"nnja.archive.value_encoding"

PLATFORM_NAME_TO_SAID = {name: said for said, name in SAID_TO_PLATFORM_NAME.items()}

# NNJAObsConv columns gpsro_tables reads.
GPSRO_COLUMNS = (
    "cycle_time",
    "variable",
    "observation",
    "time",
    "type",
    "station",
    "lat",
    "lon",
    "elev",
    "quality",
    "radius_curvature",
    "geoid_undulation",
)
# NNJAObsConv columns prepbufr_tables reads.
PREPBUFR_COLUMNS = (
    *PREPBUFR_LEVEL_COLUMNS,
    "time",
    "lat",
    "lon",
    "variable",
    "observation",
    "quality",
)
# NNJAObsSatwnd columns satwnd_tables reads; the float ones also identify a wind.
SATWND_FLOAT_KEYS = (
    "lat",
    "lon",
    "pres",
    "satellite_id",
    "wind_method",
    "wind_method_local",
    "height_method",
    "satellite_za",
    "quality",
)
SATWND_COLUMNS = (
    "cycle_time",
    "variable",
    "observation",
    "time",
    "subset",
    *SATWND_FLOAT_KEYS,
)
# NNJAObsSat columns sat_tables reads.
# Columns constant over one footprint's channel rows.
FOOTPRINT_KEYS = (
    "cycle_time",
    "time",
    "lat",
    "lon",
    "satellite",
    "scan_position",
    "scan_line",
    "detector",
    "satellite_za",
    "solza",
)
# NNJAObsSat footprint flag columns -> the archive's; null for sensors that lack them.
FOOTPRINT_FLAGS = {
    "scan_quality": "scan_quality",
    "granule_quality": "granule_quality",
    "footprint_quality": "summary_quality",
}
# NNJAObsSat columns sat_tables reads; the quality columns are optional.
SAT_COLUMNS = (
    "variable",
    "sensor_index",
    "observation",
    "quality",
    *FOOTPRINT_FLAGS,
    *FOOTPRINT_KEYS,
)


def _integer(values: np.ndarray, pa_type: pa.DataType) -> pa.Array:
    """Float values with NaN as an integer column with nulls."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    integers = np.where(valid, values, 0).astype(np.int64)
    return pa.array(integers, mask=~valid).cast(pa_type)


def gpsro_tables(frame: pd.DataFrame) -> dict[pd.Timestamp, pa.Table]:
    """GPS-RO archive rows, one per bending-angle level, keyed by cycle.

    ``frame`` holds ``gps`` rows and, for the derived vertical coordinates,
    the same occultations' ``gps_refractivity`` rows.
    """
    return _gpsro_tables(_arrow(frame, GPSRO_COLUMNS))


def _gpsro_tables(table: pa.Table) -> dict[pd.Timestamp, pa.Table]:
    variable, variables = _codes(table["variable"])
    is_gps = variable == variables.index(GPS_VARIABLE).as_py()
    is_profile = variable == variables.index(REFRACTIVITY_VARIABLE).as_py()
    radius = _floats(table["radius_curvature"])
    geoid = _floats(table["geoid_undulation"])
    elev = _floats(table["elev"])
    value = _floats(table["observation"])
    lat = _floats(table["lat"])
    lon = _floats(table["lon"])
    time = _native(table["time"])
    cycle = _native(table["cycle_time"])
    kind = _floats(table["type"])
    station, stations = _codes(table["station"])

    derived = {
        name: np.full(table.num_rows, np.nan)
        for name in (
            "impact_height_m",
            "heit_at_impact_m",
            "dry_pressure_hpa",
            "standard_atmosphere_pressure_hpa",
            "blended_pressure_5km_hpa",
        )
    }
    occ_lat = np.full(table.num_rows, np.nan)
    occ_lon = np.full(table.num_rows, np.nan)
    # An occultation is one message: its file, header time, receiver/transmitter and
    # radius of curvature.
    members = np.flatnonzero(is_gps | is_profile)
    if not members.size:
        return {}
    keys = [cycle, time, _group_key(kind), station, _group_key(radius)]
    group = _group_ids([key[members] for key in keys])[0]
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.diff(group[order])) + 1
    for positions in np.split(order, bounds):
        rows = members[positions]
        levels = rows[is_gps[rows]]
        if not levels.size:
            continue
        profile = rows[is_profile[rows]]
        # Refractivity rows sit at the occultation's reference point.
        header_lat = lat[profile[0]] if profile.size else np.nan
        header_lon = lon[profile[0]] if profile.size else np.nan
        occ_lat[levels], occ_lon[levels] = header_lat, header_lon
        elrc = radius[levels[0]]
        columns = derive_level_coordinates(
            elev[levels] + elrc,
            elev[profile],
            value[profile],
            latitude_degrees=header_lat,
            earth_radius_of_curvature_m=elrc if np.isfinite(elrc) else None,
            geoid_undulation_m=geoid[levels[0]]
            if np.isfinite(geoid[levels[0]])
            else None,
        )
        for name, values in columns.items():
            derived[name][levels] = values

    # The transmitter is the station id after its first four characters.
    names = stations.to_pylist()
    transmitter_by_station = np.array(
        [
            *pd.to_numeric(pd.Series(names, dtype=object).str[4:], errors="coerce"),
            np.nan,
        ],
        dtype=np.float64,
    )
    gps = np.flatnonzero(is_gps)
    table = {
        "satellite_id": _integer(kind[gps], pa.int64()),
        "transmitter_id": _integer(transmitter_by_station[station[gps]], pa.int64()),
        "occ_latitude": occ_lat[gps],
        "occ_longitude": occ_lon[gps],
        "tangent_latitude": lat[gps],
        "tangent_longitude": lon[gps],
        "qfro": _integer(_floats(table["quality"])[gps], pa.int64()),
        "earth_radius_curvature": radius[gps],
        "geoid_undulation": geoid[gps],
        "obs_time": pa.array(time[gps]).cast(pa.timestamp("ns")),
        "impact_parameter": elev[gps] + radius[gps],
        "bending_angle": value[gps].astype(np.float32),
        **{name: values[gps] for name, values in derived.items()},
    }
    out = pa.table(
        {
            name: values
            if isinstance(values, pa.Array)
            else pa.array(values, from_pandas=True)
            for name, values in table.items()
        }
    )
    return _split_by_cycle(out, cycle[gps].astype("datetime64[ns]"))


def satwnd_tables(frame: pd.DataFrame) -> dict[pd.Timestamp, pa.Table]:
    """SATWND archive rows, one per wind, keyed by cycle.

    ``frame`` holds ``NNJAObsSatwnd`` ``u`` and ``v`` rows; each wind's two
    components are rejoined on the wind's own columns.
    """
    return _satwnd_tables(_arrow(frame, SATWND_COLUMNS))


def _satwnd_tables(table: pa.Table) -> dict[pd.Timestamp, pa.Table]:
    variable, names = _codes(table["variable"])
    is_u = variable == names.index("u").as_py()
    u_rows, v_rows = np.flatnonzero(is_u), np.flatnonzero(~is_u)
    subset, subset_names = _codes(table["subset"])
    time = _native(table["time"])
    keys = {"cycle_time": _native(table["cycle_time"]), "time": time, "subset": subset}
    keys.update((name, _floats(table[name])) for name in SATWND_FLOAT_KEYS)
    # NNJAObsSatwnd builds a file's v rows from the same winds, in the same order, as
    # its u rows.
    if not _same_rows(keys, u_rows, v_rows):
        raise ValueError("NNJAObsSatwnd v rows are not in the u rows' wind order")
    observation = _floats(table["observation"])
    winds = {name: values[u_rows] for name, values in keys.items()}

    cycle = winds["cycle_time"].astype("datetime64[ns]")
    window = archive_window(winds["time"], cycle)
    pixels, valid_position = hpx_nest_pixels(
        winds["lat"], winds["lon"], ARCHIVE_HPX_LEVEL
    )
    subset_lookup = np.array([*subset_names.to_pylist(), None], dtype=object)
    subset_name = subset_lookup[winds["subset"]]
    swcm = winds["wind_method"]
    types = satwnd_gsi_types.resolve(
        subset_name, winds["satellite_id"], swcm, winds["wind_method_local"]
    )
    # The archive keeps SWCM as read, with CMCM only for templates lacking it.
    known = winds["subset"] >= 0
    has_swcm = np.bincount(
        winds["subset"][known],
        weights=np.isfinite(swcm[known]),
        minlength=len(subset_names),
    )
    use_swcm = np.ones(swcm.size, dtype=bool)
    use_swcm[known] = has_swcm[winds["subset"][known]] > 0
    method = np.where(use_swcm, swcm, winds["wind_method_local"])

    table = pa.table(
        {
            "time_utc": pa.array(winds["time"]).cast(pa.timestamp("ns")),
            "da_window": pa.array(window),
            "cycle": pa.array(cycle),
            "latitude": winds["lat"],
            "longitude": winds["lon"],
            "hpx4096_nest": pa.array(pixels, mask=~valid_position),
            "wind_u_derived": observation[u_rows],
            "wind_v_derived": observation[v_rows],
            "assigned_pressure": pa.array(winds["pres"], from_pandas=True),
            "satellite_id": _integer(winds["satellite_id"], pa.uint32()),
            "wind_computation_method": _integer(method, pa.uint32()),
            "ncep_dump_subtype": pa.array(subset_name, pa.string()),
            "sdmedit_wind_quality_mark": _integer(winds["quality"], pa.uint32()),
            "satellite_zenith_angle": pa.array(winds["satellite_za"], from_pandas=True),
            "gsi_observation_type": _integer(
                np.where(types.report_type >= 0, types.report_type, np.nan), pa.uint16()
            ),
            "gsi_internal_subtype": _integer(
                np.where(types.internal_subtype >= 0, types.internal_subtype, np.nan),
                pa.uint8(),
            ),
            "gsi_type_mapping_tier": pa.array(types.tier, pa.string()),
        }
    )
    return _split_by_cycle(table, cycle)


def prepbufr_tables(frame: pd.DataFrame) -> dict[pd.Timestamp, pa.Table]:
    """PrepBUFR archive rows, one per report level, keyed by cycle.

    ``frame`` holds ``u``, ``v``, ``q``, ``t`` and ``pres`` rows from
    ``NNJAObsConv(original_event=True)``. The archive holds each observation's original
    event (program code 1); the default, the latest event, is for temperature often
    the virtual temperature, with later quality marks, which the loader reads as a
    different observation. ``XOB``, ``YOB`` and ``DHR`` are the report header's
    position and time (``report_lon``, ``report_lat``, ``report_time``); ``XDR``,
    ``YDR`` and ``HRDR`` the level's (``lon``, ``lat``, ``time``), which is the
    header's except for drifting radiosonde levels. ``pres`` rows supply the levels
    that carry no other variable.

    Reports are keyed by ``cycle_time``, the file each was decoded from; one at the
    edge of two files' windows is in both, as in the archive.
    """
    return _prepbufr_tables(_arrow(frame, PREPBUFR_COLUMNS))


def _prepbufr_tables(table: pa.Table) -> dict[pd.Timestamp, pa.Table]:
    variable, variables = _codes(table["variable"])
    wanted = [variables.index(name).as_py() for name in PREPBUFR_LEVEL_VARIABLES]
    rows = np.flatnonzero(np.isin(variable, [code for code in wanted if code >= 0]))
    if not rows.size:
        return {}
    variable = variable[rows]
    station, stations = _codes(table["station"])
    levels = {
        name: _native(table[name])[rows] for name in ("cycle_time", "report_time")
    }
    levels["station"] = station[rows]
    for name in PREPBUFR_LEVEL_COLUMNS:
        if name not in levels:
            levels[name] = _floats(table[name])[rows]
    keys = [_group_key(values) for values in levels.values()]
    # A report repeated in the file gives each variable once per copy.
    occurrence = _group_ids([*keys, variable])[2]
    level, first, _ = _group_ids([*keys, occurrence])

    observation = _floats(table["observation"])[rows]
    quality = _floats(table["quality"])[rows]
    # e2s returns kg/kg and K; PrepBUFR stores QOB in whole mg/kg and TOB in 0.1 degC.
    value = observation.copy()
    code = {name: variables.index(name).as_py() for name in PREPBUFR_VARIABLES}
    is_q, is_t = variable == code["q"], variable == code["t"]
    value[is_q] = np.rint(observation[is_q] * 1e6)
    value[is_t] = np.round(observation[is_t] - 273.15, 1)
    observed = {
        name: np.full(first.size, np.nan)
        for name in ("QOB", "QQM", "TOB", "TQM", "UOB", "VOB", "WQM")
    }
    for name, (value_column, quality_column) in PREPBUFR_VARIABLES.items():
        selected = variable == code[name]
        observed[value_column][level[selected]] = value[selected]
        observed[quality_column][level[selected]] = quality[selected]

    header = {name: values[first] for name, values in levels.items()}
    cycle = header["cycle_time"].astype("datetime64[ns]")

    def hours_from_cycle(time_ns):
        return (time_ns - cycle.view(np.int64)) / 3_600_000_000_000

    station_lookup = np.array([*stations.to_pylist(), None], dtype=object)
    columns = {
        "XOB": header["report_lon"],
        "YOB": header["report_lat"],
        "DHR": hours_from_cycle(header["report_time"]),
        "XDR": _floats(table["lon"])[rows][first],
        "YDR": _floats(table["lat"])[rows][first],
        "HRDR": hours_from_cycle(_native(table["time"])[rows][first]),
        "ELV": header["station_elev"],
        "TYP": header["type"],
        "CAT": header["level_cat"],
        # e2s returns Pa; PrepBUFR stores hPa.
        "POB": header["pres"] / 100.0,
        "PQM": header["pressure_quality"],
        "ZOB": header["elev"],
        **observed,
    }
    table = pa.table(
        {
            "SID": pa.array(station_lookup[header["station"]], pa.string()),
            **{
                name: pa.array(values.astype(np.float32), from_pandas=True)
                for name, values in columns.items()
            },
        }
    )
    return _split_by_cycle(table, cycle)


def _footprint_flags(table, kept, first) -> dict:
    """The archive's footprint flag columns from those NNJAObsSat carries."""
    columns = {}
    for name, archive_name in FOOTPRINT_FLAGS.items():
        if name not in table.column_names:
            continue
        values = _native(table[name]).astype(np.float64)
        values = (values if values.size == kept.size else values[kept])[first]
        values[values < 0] = np.nan
        if np.isfinite(values).any():
            columns[archive_name] = _integer(values, pa.uint32())
    return columns


def _quality_columns(table, sensor, kept, slot, footprint, channel_count) -> dict:
    """The per-channel provider flags NNJAObsSat's quality carries, as the archive's
    columns: ATMS channel_quality, and CrIS fov_quality_bad from NFQF."""
    if "quality" not in table.column_names or sensor not in ("atms", "cris"):
        return {}
    quality = np.asarray(_native(table["quality"]), dtype=np.float64)
    quality = quality if quality.size == kept.size else quality[kept]
    flags = np.zeros((channel_count, footprint[-1] + 1), dtype=np.int64)
    flags[slot, footprint] = np.where(quality > 0, quality, 0)
    if sensor == "cris":
        bad = (flags & CRIS_FOV_QUALITY_BAD).any(axis=0)
        return {"fov_quality_bad": pa.array(bad.astype(np.uint8))}
    leaf = pa.StructArray.from_arrays(
        [pa.array(flags.T.ravel().astype(np.uint16))], ["channel_quality"]
    )
    return {"channel_auxiliary": pa.FixedSizeListArray.from_arrays(leaf, channel_count)}


def sat_tables(
    frame: pd.DataFrame,
    sensor: str,
    ir_channels: str | Mapping[str, Sequence[int]] | None = "ir32",
) -> dict[pd.Timestamp, pa.Table]:
    """Wide satellite archive rows, one per footprint, keyed by cycle.

    ``frame`` holds ``NNJAObsSat`` rows of ``sensor``'s variable (``SAT_VARIABLE``) in
    the order the source emits them: a footprint's channel rows are consecutive and
    ascending in ``sensor_index``. Rows of other variables are ignored. The channel
    columns are those ``NNJAWideLoader(ir_channels=ir_channels)`` reads, null where
    the frame has no row.
    """
    return _sat_tables(_arrow(frame, SAT_COLUMNS), sensor, ir_channels)


def _sat_tables(
    table: pa.Table, sensor: str, ir_channels
) -> dict[pd.Timestamp, pa.Table]:
    # Arrow and numpy throughout: earth2studio frames are Arrow-backed, and pandas
    # operations on them are slower for identical output.
    variable, names = _codes(table["variable"])
    code = names.index(SAT_VARIABLE.get(sensor, sensor)).as_py()
    if code < 0:
        return {}
    wanted = (_resolve_ir_channels(ir_channels) or {}).get(sensor)
    channels = np.array(
        sorted(SENSOR_CONFIGS[sensor].channels if wanted is None else wanted),
        dtype=np.int64,
    )
    channel_slot = np.full(channels.max() + 1, -1, dtype=np.int64)
    channel_slot[channels] = np.arange(channels.size)
    sensor_index = _native(table["sensor_index"])
    in_axis = (variable == code) & (sensor_index >= 0)
    in_axis &= sensor_index < channel_slot.size
    slot = np.full(sensor_index.size, -1, dtype=np.int64)
    slot[in_axis] = channel_slot[sensor_index[in_axis]]
    kept = np.flatnonzero(slot >= 0)
    if not kept.size:
        return {}
    slot = slot[kept]

    # A footprint starts where the channel sequence restarts or a footprint column changes.
    starts = np.ones(kept.size, dtype=bool)
    starts[1:] = slot[1:] <= slot[:-1]
    keys = {}
    for name in FOOTPRINT_KEYS:
        if name not in table.column_names:
            continue
        values = _native(table[name])
        values = values if values.size == kept.size else values[kept]
        keys[name] = values
        same = values[1:] == values[:-1]
        if values.dtype.kind == "f":
            same |= np.isnan(values[1:]) & np.isnan(values[:-1])
        starts[1:] |= ~same
    first = np.flatnonzero(starts)

    footprint = np.cumsum(starts) - 1
    observation = np.full((channels.size, first.size), np.nan)
    values = _native(table["observation"])
    values = values if values.size == kept.size else values[kept]
    observation[slot, footprint] = values
    prefix = SENSOR_VALUE_PREFIX[sensor]
    if prefix in RADIANCE_PREFIXES:
        wavenumber = ir_spectral.wavenumber_cm_inverse(sensor, channels)
        observation = ir_spectral.spectral_radiance_mw(observation, wavenumber[:, None])
    encoding = VALUE_ENCODING.get(sensor)
    value_type, metadata = pa.float32(), None
    if encoding is not None:
        # The inverse of `ir_spectral.radiance_mw`.
        scale = 10.0 ** (encoding["scale"] - 3)
        observation = np.rint(observation * scale - encoding["reference"])
        value_type = pa.int32()
        metadata = {VALUE_ENCODING_KEY: json.dumps(encoding, separators=(",", ":"))}

    time_ns = keys["time"][first]
    cycle = keys["cycle_time"][first].astype("datetime64[ns]")
    window = archive_window(time_ns, cycle)
    lat = keys["lat"][first].astype(np.float64)
    lon = keys["lon"][first].astype(np.float64)
    pixels, valid_position = hpx_nest_pixels(lat, lon, ARCHIVE_HPX_ORDER)
    satellite, platforms = _codes(table["satellite"])
    said_by_code = np.array(
        [PLATFORM_NAME_TO_SAID.get(name, -1) for name in platforms.to_pylist()] + [-1]
    )
    said = said_by_code[satellite[kept[first]]]
    if (said < 0).any():
        unknown = sorted(
            {str(platforms[c]) for c in np.unique(satellite[kept[first]][said < 0])}
        )
        raise ValueError(f"{sensor}: no SAID for satellites {unknown}")

    def integer(name, pa_type):
        values = keys[name][first]
        return pa.array(np.maximum(values, 0), mask=values < 0).cast(pa_type)

    scan_columns = {"field_of_view": integer("scan_position", pa.uint16())}
    if sensor in SCAN_POSITION_COLUMN:
        # The detector within the look, which the archive names field_of_view.
        if "detector" not in keys:
            raise ValueError(f"{sensor} needs the frame's detector column")
        scan_columns = {
            "field_of_view": integer("detector", pa.uint16()),
            SCAN_POSITION_COLUMN[sensor]: integer("scan_position", pa.uint16()),
        }

    utc = pa.timestamp("ns", tz="UTC")
    columns = {
        "latitude": pa.array(lat),
        "longitude": pa.array(lon),
        "time_utc": pa.array(time_ns).cast(pa.timestamp("ns")).cast(utc),
        "da_window": pa.array(window).cast(utc),
        "platform_id": pa.array(said, pa.uint16()),
        "satellite_zenith_angle": pa.array(
            keys["satellite_za"][first].astype(np.float32), from_pandas=True
        ),
        "solar_zenith_angle": pa.array(
            keys["solza"][first].astype(np.float32), from_pandas=True
        ),
        **scan_columns,
        "hpx2048_nest": pa.array(pixels, mask=~valid_position),
        **_quality_columns(table, sensor, kept, slot, footprint, channels.size),
        **_footprint_flags(table, kept, first),
    }
    fields = [pa.field(name, values.type) for name, values in columns.items()]
    arrays = list(columns.values())
    for channel, values in zip(channels, observation):
        missing = ~np.isfinite(values)
        stored = np.where(missing, 0, values).astype(value_type.to_pandas_dtype())
        arrays.append(pa.array(stored, mask=missing))
        fields.append(
            pa.field(f"{prefix}__ch_{channel:05d}", value_type, metadata=metadata)
        )
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    return _split_by_cycle(table, cycle)


def analysis_tables(
    conv: pd.DataFrame | None = None,
    satwnd: pd.DataFrame | None = None,
    sat: pd.DataFrame | None = None,
    *,
    sensors: Sequence[str],
    ir_channels: str | Mapping[str, Sequence[int]] | None,
    workers: int | None = None,
) -> dict:
    """``DAModel.run_analysis`` table arguments from Earth2Studio NNJA frames.

    ``conv`` is an ``NNJAObsConv(original_event=True)`` frame of any
    PrepBUFR and GPS-RO variables, ``satwnd`` an ``NNJAObsSatwnd`` frame and ``sat`` an
    ``NNJAObsSat`` frame of any of ``sensors``. A frame not passed gives no argument.

    Every footprint, report, occultation and wind comes from one file, so each stream
    is cut by ``cycle_time`` and the pieces adapted in parallel on ``workers`` threads (default
    the smaller of 32 and the CPU count); the result equals adapting each whole stream.
    """

    # (frame's Arrow table, stream of each row)
    streams = []
    if conv is not None:
        streams.append((_arrow(conv, conv.columns), _conv_stream))
    if satwnd is not None:
        streams.append((_arrow(satwnd, SATWND_COLUMNS), None))
    if sat is not None:
        streams.append((_arrow(sat, SAT_COLUMNS), _variable_stream))
    adapters = {
        "prepbufr": ("prepbufr_tables", None, _prepbufr_tables),
        "gps": ("gpsro_tables", None, _gpsro_tables),
        "satwnd": ("satwnd_tables", None, _satwnd_tables),
        **{
            SAT_VARIABLE.get(sensor, sensor): (
                "satellite_tables",
                sensor,
                functools.partial(_sat_tables, sensor=sensor, ir_channels=ir_channels),
            )
            for sensor in sensors
        },
    }

    with ThreadPoolExecutor(max_workers=workers or min(32, os.cpu_count())) as pool:
        jobs = []
        for table, stream in streams:
            for (name, _), piece in _pieces(table, stream, pool).items():
                if name in adapters:
                    jobs.append((*adapters[name], piece))
        results = list(pool.map(lambda job: job[2](job[3]), jobs))

    tables = {}
    if conv is not None:
        tables.update(prepbufr_tables={}, gpsro_tables={})
    if satwnd is not None:
        tables["satwnd_tables"] = {}
    if sat is not None:
        tables["satellite_tables"] = {sensor: {} for sensor in sensors}
    for (argument, sensor, _, _), result in zip(jobs, results):
        target = tables[argument] if sensor is None else tables[argument][sensor]
        target.update(result)
    return tables


def _group_key(values: np.ndarray) -> np.ndarray:
    """Integers equal exactly where pandas groups values together: floats by bit
    pattern with every NaN one value and -0.0 as 0.0."""
    if values.dtype.kind != "f":
        return values
    bits = (values.astype(np.float64) + 0.0).view(np.int64)
    return np.where(np.isnan(values), np.int64(-1), bits)


def _group_ids(keys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rows by equal ``keys`` tuples, as pandas ``groupby(sort=False)`` does.

    Returns each row's group number (by first appearance, ``ngroup``), the first row
    of each group in that order, and each row's position within its group
    (``cumcount``). One stable sort; no Python per group.
    """
    order = np.lexsort(keys[::-1])
    change = np.zeros(order.size, dtype=bool)
    change[0] = True
    for key in keys:
        ordered = key[order]
        change[1:] |= ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(change)
    run = np.cumsum(change) - 1
    first_rows = order[starts]
    rank = np.empty(starts.size, dtype=np.int64)
    rank[np.argsort(first_rows, kind="stable")] = np.arange(starts.size)
    ids = np.empty(order.size, dtype=np.int64)
    ids[order] = rank[run]
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size) - starts[run]
    return ids, np.sort(first_rows), position


def _same_rows(keys: dict, a: np.ndarray, b: np.ndarray) -> bool:
    """True when rows ``a[i]`` and ``b[i]`` agree on every key, NaN equal to NaN."""
    if a.size != b.size:
        return False
    for values in keys.values():
        x, y = values[a], values[b]
        same = x == y
        if x.dtype.kind == "f":
            same |= np.isnan(x) & np.isnan(y)
        if not same.all():
            return False
    return True


def _floats(column: pa.ChunkedArray) -> np.ndarray:
    return column.cast(pa.float64()).to_numpy()


def _split_by_cycle(table: pa.Table, cycle: np.ndarray) -> dict[pd.Timestamp, pa.Table]:
    order = np.argsort(cycle, kind="stable")
    values, first = np.unique(cycle[order], return_index=True)
    bounds = [*first, order.size]
    return {
        pd.Timestamp(value): table.take(pa.array(order[lo:hi]))
        for value, lo, hi in zip(values, bounds[:-1], bounds[1:])
    }


def _pieces(table: pa.Table, stream, pool, segments: int = 32) -> dict:
    """``table`` cut into one table per (stream, archive cycle), rows in their original
    order. ``stream(segment)`` names each row's stream (None: one stream, "satwnd").

    Segments of the table are keyed on ``pool`` in parallel; a key's runs are zero-copy
    slices, concatenated in row order, so a report straddling two segments rejoins.
    """
    if not table.num_rows:
        return {}
    size = -(-table.num_rows // segments)
    parts = [table.slice(lo, size) for lo in range(0, table.num_rows, size)]

    def runs(part):
        names = np.array(["satwnd"]) if stream is None else None
        codes = np.zeros(part.num_rows, dtype=np.int64)
        if stream is not None:
            codes, names = stream(part)
        cycle = _native(part["cycle_time"])
        change = (codes[1:] != codes[:-1]) | (cycle[1:] != cycle[:-1])
        bounds = [0, *(np.flatnonzero(change) + 1), cycle.size]
        return [
            ((str(names[codes[lo]]), int(cycle[lo])), part.slice(lo, hi - lo))
            for lo, hi in zip(bounds[:-1], bounds[1:])
        ]

    merged = {}
    for part_runs in pool.map(runs, parts):
        for key, piece in part_runs:
            merged.setdefault(key, []).append(piece)
    return {key: pa.concat_tables(slices) for key, slices in merged.items()}


def _variable_stream(part: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    codes, names = _codes(part["variable"])
    return codes, np.array([*names.to_pylist(), None], dtype=object)


def _conv_stream(part: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    variable, names = _codes(part["variable"])
    gps = np.isin(names.to_pylist(), [GPS_VARIABLE, REFRACTIVITY_VARIABLE])
    is_gps = np.append(gps, False)[variable]
    return is_gps.astype(np.int64), np.array(["prepbufr", "gps"])


def _arrow(frame: pd.DataFrame, names) -> pa.Table:
    """The frame's ``names`` present in it, as Arrow; zero-copy for Arrow-backed frames."""
    present = [name for name in names if name in frame.columns]
    # A view without attrs: Arrow cannot serialize e2s's ndarray request_time and warns.
    view = pd.DataFrame(frame, copy=False)
    return pa.Table.from_pandas(view, columns=present, preserve_index=False)


def _codes(column: pa.ChunkedArray) -> tuple[np.ndarray, pa.Array]:
    """Per-row int32 codes of a string column, -1 for null, and the code -> value table."""
    if not pa.types.is_dictionary(column.type):
        column = pc.dictionary_encode(column)
    array = column.unify_dictionaries().combine_chunks()
    return pc.fill_null(array.indices.cast(pa.int32()), -1).to_numpy(), array.dictionary


def _native(column: pa.ChunkedArray) -> np.ndarray:
    """One numpy array in the column's own dtype: floats keep NaN for null, integers
    widen to int64 with -1 for null, strings become codes, timestamps int64 ns."""
    if pa.types.is_dictionary(column.type) or pa.types.is_string(column.type):
        return _codes(column)[0]
    if pa.types.is_timestamp(column.type):
        # UTC instants either way; dropping the timezone keeps the stored integers.
        return column.cast(pa.timestamp("ns")).cast(pa.int64()).to_numpy()
    if pa.types.is_integer(column.type):
        return pc.fill_null(column.cast(pa.int64()), -1).to_numpy()
    return column.to_numpy()
