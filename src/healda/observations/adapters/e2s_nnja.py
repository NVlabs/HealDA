# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Earth2Studio NNJA frames as the canonical NNJA archive tables.

``earth2studio.data.NNJAObsConv`` (``gps``, ``gps_refractivity``) and
``NNJAObsSatwnd`` (``u``, ``v``) return raw decoded rows. These functions
restate them in the column contract of healda's NNJA GPS-RO and SATWND
parquet archives, deriving what the ETL derives with the same code, so the
archive loaders screen, thin and emit them exactly as they do archive files.
Each returns ``{cycle: table}``, a ``CycleTableSource`` for the loaders'
``table_source`` argument, or can be
written in the archive layout (``write_gpsro_archive``/``write_satwnd_archive``)
so a later run points the loaders' ``archive_root`` at it instead of decoding again.

Usage, for the cycle file around ``cycle`` (dumps cover ``[cycle - 3h, cycle + 3h)``)::

    frame = NNJAObsConv(time_tolerance=window)(cycle, ["gps", "gps_refractivity"])
    loader = NNJAGpsroLoader(table_source=gpsro_tables(frame))
    await loader.sel_time(times)  # or via NNJAConventionalLoader(gpsro_table_source=...)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from healda.observations.loaders.nnja_base import archive_cycle_and_window
from healda.observations.loaders.nnja_satwnd import ARCHIVE_HPX_LEVEL
from healda.observations.preprocessing import satwnd_gsi_types
from healda.observations.preprocessing.gpsro_pressure import derive_level_coordinates
from healda.observations.preprocessing.sat_helpers import hpx_nest_pixels

GPS_VARIABLE = "gps"
REFRACTIVITY_VARIABLE = "gps_refractivity"


def _float(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
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
    frame = _naive_utc(frame).reset_index(drop=True)
    is_gps = (frame["variable"] == GPS_VARIABLE).to_numpy()
    is_profile = (frame["variable"] == REFRACTIVITY_VARIABLE).to_numpy()
    radius = _float(frame, "radius_curvature")
    geoid = _float(frame, "geoid_undulation")
    elev = _float(frame, "elev")
    value = _float(frame, "observation")
    lat = _float(frame, "lat")
    lon = _float(frame, "lon")

    derived = {
        name: np.full(len(frame), np.nan)
        for name in (
            "impact_height_m",
            "heit_at_impact_m",
            "dry_pressure_hpa",
            "standard_atmosphere_pressure_hpa",
            "blended_pressure_5km_hpa",
        )
    }
    occ_lat = np.full(len(frame), np.nan)
    occ_lon = np.full(len(frame), np.nan)
    # An occultation is one message: its header time, receiver/transmitter and
    # radius of curvature.
    keys = frame[["time", "type", "station", "radius_curvature"]].copy()
    keys["type"] = keys["type"].astype("float64")
    members = np.flatnonzero(is_gps | is_profile)
    groups = keys.iloc[members].groupby(list(keys.columns), sort=False, dropna=False)
    for positions in groups.indices.values():
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

    gps = frame.loc[is_gps]
    station = gps["station"].astype("string")
    table = {
        "satellite_id": _integer(_float(gps, "type"), pa.int64()),
        "transmitter_id": _integer(
            pd.to_numeric(station.str[4:], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            ),
            pa.int64(),
        ),
        "occ_latitude": occ_lat[is_gps],
        "occ_longitude": occ_lon[is_gps],
        "tangent_latitude": lat[is_gps],
        "tangent_longitude": lon[is_gps],
        "qfro": _integer(_float(gps, "quality"), pa.int64()),
        "earth_radius_curvature": radius[is_gps],
        "geoid_undulation": geoid[is_gps],
        "obs_time": pa.array(gps["time"].to_numpy(dtype="datetime64[ns]")),
        "impact_parameter": elev[is_gps] + radius[is_gps],
        "bending_angle": value[is_gps].astype(np.float32),
        **{name: values[is_gps] for name, values in derived.items()},
    }
    out = pa.table(
        {
            name: values
            if isinstance(values, pa.Array)
            else pa.array(values, from_pandas=True)
            for name, values in table.items()
        }
    )
    cycle, _ = archive_cycle_and_window(gps["time"])
    return _split_by_cycle(out, cycle.to_numpy())


def satwnd_tables(frame: pd.DataFrame) -> dict[pd.Timestamp, pa.Table]:
    """SATWND archive rows, one per wind, keyed by cycle.

    ``frame`` holds ``NNJAObsSatwnd`` ``u`` and ``v`` rows; each wind's two
    components are rejoined on the wind's own columns.
    """
    frame = _naive_utc(frame)
    keys = [
        "time",
        "lat",
        "lon",
        "pres",
        "satellite_id",
        "subset",
        "wind_method",
        "wind_method_local",
        "height_method",
        "satellite_za",
        "quality",
    ]
    base = pd.DataFrame(
        {
            name: frame[name].to_numpy()
            if name in ("time", "subset")
            else _float(frame, name)
            for name in keys
        }
    )
    base["observation"] = _float(frame, "observation")
    is_u = (frame["variable"] == "u").to_numpy()
    u_rows, v_rows = base.loc[is_u].reset_index(drop=True), base.loc[~is_u]
    v_rows = v_rows.reset_index(drop=True)
    if _same_winds(u_rows, v_rows, keys):
        # The source emits the v block in the u block's wind order.
        winds = u_rows.rename(columns={"observation": "observation_u"})
        winds["observation_v"] = v_rows["observation"].to_numpy()
    else:
        for rows in (u_rows, v_rows):
            rows["occurrence"] = rows.groupby(keys, dropna=False, sort=False).cumcount()
        winds = u_rows.merge(
            v_rows,
            on=[*keys, "occurrence"],
            how="inner",
            suffixes=("_u", "_v"),
            sort=False,
        )
    cycle, window = archive_cycle_and_window(winds["time"])
    lat = winds["lat"].to_numpy(dtype=np.float64)
    lon = winds["lon"].to_numpy(dtype=np.float64)
    pixels, valid_position = hpx_nest_pixels(lat, lon, ARCHIVE_HPX_LEVEL)
    swcm = winds["wind_method"].to_numpy()
    types = satwnd_gsi_types.resolve(
        winds["subset"].to_numpy(dtype=object),
        winds["satellite_id"].to_numpy(),
        swcm,
        winds["wind_method_local"].to_numpy(),
    )
    # The archive keeps SWCM as read, with CMCM only for templates lacking it.
    method = np.where(
        winds.groupby("subset")["wind_method"].transform(lambda s: s.notna().any()),
        swcm,
        winds["wind_method_local"].to_numpy(),
    )

    table = pa.table(
        {
            "time_utc": pa.array(winds["time"].to_numpy(dtype="datetime64[ns]")),
            "da_window": pa.array(window.to_numpy(dtype="datetime64[ns]")),
            "cycle": pa.array(cycle.to_numpy(dtype="datetime64[ns]")),
            "latitude": lat,
            "longitude": lon,
            "hpx4096_nest": pa.array(pixels, mask=~valid_position),
            "wind_u_derived": winds["observation_u"].to_numpy(),
            "wind_v_derived": winds["observation_v"].to_numpy(),
            "assigned_pressure": pa.array(
                winds["pres"].to_numpy(dtype=np.float64), from_pandas=True
            ),
            "satellite_id": _integer(winds["satellite_id"].to_numpy(), pa.uint32()),
            "wind_computation_method": _integer(method, pa.uint32()),
            "ncep_dump_subtype": pa.array(
                winds["subset"].to_numpy(dtype=object), pa.string()
            ),
            "sdmedit_wind_quality_mark": _integer(
                winds["quality"].to_numpy(), pa.uint32()
            ),
            "satellite_zenith_angle": pa.array(
                winds["satellite_za"].to_numpy(dtype=np.float64), from_pandas=True
            ),
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
    return _split_by_cycle(table, cycle.to_numpy())


def _same_winds(u: pd.DataFrame, v: pd.DataFrame, keys: list[str]) -> bool:
    """True when row i of ``u`` and of ``v`` are the same wind, NaN equal to NaN."""
    if len(u) != len(v):
        return False
    for name in keys:
        a, b = u[name].to_numpy(), v[name].to_numpy()
        same = a == b
        if a.dtype.kind == "f":
            same |= np.isnan(a) & np.isnan(b)
        if not same.all():
            return False
    return True


def _naive_utc(frame: pd.DataFrame) -> pd.DataFrame:
    # Loaders look cycles up with naive UTC timestamps; a tz-aware key would never match.
    if getattr(frame["time"].dtype, "tz", None) is None:
        return frame
    return frame.assign(time=frame["time"].dt.tz_convert("UTC").dt.tz_localize(None))


def _split_by_cycle(table: pa.Table, cycle: np.ndarray) -> dict[pd.Timestamp, pa.Table]:
    out = {}
    for value in np.unique(cycle):
        rows = np.flatnonzero(cycle == value)
        out[pd.Timestamp(value)] = table.take(pa.array(rows))
    return out


def write_gpsro_archive(tables: dict[pd.Timestamp, pa.Table], root: str) -> None:
    """Write ``gpsro_tables`` output as GPS-RO archive cycle files under ``root``."""
    for cycle, table in tables.items():
        directory = os.path.join(root, f"{cycle:%Y}")
        os.makedirs(directory, exist_ok=True)
        name = f"gdas.{cycle:%Y%m%d}.t{cycle:%H}z.gpsro.tm00.bufr_d.parquet"
        pq.write_table(table, os.path.join(directory, name))


def write_satwnd_archive(tables: dict[pd.Timestamp, pa.Table], root: str) -> None:
    """Write ``satwnd_tables`` output as SATWND archive day files under ``root``.

    Day files hold one row group per DA window. Windows already in a day file
    are kept unless ``tables`` supplies them again.
    """
    by_day: dict[pd.Timestamp, dict[pd.Timestamp, pa.Table]] = {}
    for table in tables.values():
        window = pd.to_datetime(table["da_window"].to_numpy(zero_copy_only=False))
        for value in np.unique(window):
            rows = np.flatnonzero(window == value)
            stamp = pd.Timestamp(value)
            by_day.setdefault(stamp.normalize(), {})[stamp] = table.take(rows)
    os.makedirs(root, exist_ok=True)
    for day, windows in by_day.items():
        path = os.path.join(root, f"{day:%Y%m%d}.parquet")
        if os.path.exists(path):
            existing = pq.ParquetFile(path)
            for group in range(existing.num_row_groups):
                part = existing.read_row_group(group)
                stamp = pd.Timestamp(part["da_window"][0].as_py()).tz_localize(None)
                windows.setdefault(stamp, part)
        ordered = [windows[stamp] for stamp in sorted(windows)]
        schema = ordered[0].schema
        with pq.ParquetWriter(path, schema) as writer:
            for part in ordered:
                writer.write_table(part.cast(schema), row_group_size=part.num_rows)
