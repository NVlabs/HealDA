# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
import fsspec

import healda.utils.storage
from healda.config import environment as config
from healda.utils.cache import AtomicFileCache


logger = logging.getLogger(__name__)
STANDARD_GRAVITY = 9.80665
ERA5_INVARIANT_ROOT = "s3://nsf-ncar-era5/e5.oper.invariant/197901"
ERA5_STATIC_FILES = {
    "land_surface": (
        f"{ERA5_INVARIANT_ROOT}/"
        "e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"
    ),
    "orog": (
        f"{ERA5_INVARIANT_ROOT}/"
        "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
    ),
}
ERA5_STATIC_RAW_NAMES = {"land_surface": "lsm", "orog": "z"}


FILE_PATTERN = re.compile(
    r"""
(?P<path>[^\n]*?
    (
        (?P<file>e5\.oper\.an\.(sfc|pl)\.
        (?P<table>\d+)_
        (?P<paramID>\d+)_
        (?P<variable_name>[\w\d]+)\.
        ll025(sc|uv)\.
        (?P<start_date>\d{10})_
        (?P<end_date>\d{10})\.
        nc)|
        (?P<precipFile>e5\.accumulated_tp_1h\.(?P<precipMonth>\d{6})\.nc)
    )
)
""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Era5SourceRequest:
    time: pd.Timestamp
    raw_name: str
    level: int | None = None


def _parse_time(s: str) -> pd.Timestamp:
    return pd.Timestamp(
        year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]), hour=int(s[8:10])
    )


def _precip_month_to_range(month_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    year = int(month_str[:4])
    month = int(month_str[4:6])
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    return start, end


def _index_from_files(files: list[str]) -> pd.DataFrame:
    records = []
    for match in FILE_PATTERN.findall("\n".join(files)):
        values = {k: match[i - 1] for k, i in FILE_PATTERN.groupindex.items()}
        record = {"file": values["path"]}
        if values["precipFile"]:
            record["name"] = "tp"
            record["start_date"], record["end_date"] = _precip_month_to_range(
                values["precipMonth"]
            )
        else:
            record["name"] = values["variable_name"]
            record["start_date"] = _parse_time(values["start_date"])
            record["end_date"] = _parse_time(values["end_date"])
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=["file", "name", "start_date", "end_date"],
    )


class Era5FileIndex:
    def __init__(
        self,
        roots: Sequence[str],
        *,
        profile: str | None = None,
        index_cache_path: str | Path | None = None,
    ):
        self.roots = list(roots)
        self.profile = profile
        self.index_cache_path = Path(
            index_cache_path or os.path.join(config.CACHE_DIR, "era5_index.csv")
        )
        self.df = self._load_or_build()
        self.by_name = {
            name: df.reset_index(drop=True) for name, df in self.df.groupby("name")
        }

    def resolve_file(self, raw_name: str, time: pd.Timestamp) -> str:
        if raw_name not in self.by_name:
            raise KeyError(f"No ERA5 files indexed for {raw_name}")
        df = self.by_name[raw_name]
        timestamp = pd.Timestamp(time)
        idx = (
            np.searchsorted(
                df["start_date"].values, np.datetime64(timestamp), side="right"
            )
            - 1
        )
        if idx < 0:
            raise FileNotFoundError(f"No file for {raw_name} at {time}")
        row = df.iloc[int(idx)]
        if not (row.start_date <= timestamp <= row.end_date):
            raise FileNotFoundError(f"No file for {raw_name} at {time}")
        return row.file

    def _load_or_build(self) -> pd.DataFrame:
        if self.index_cache_path.exists():
            return pd.read_csv(
                self.index_cache_path, parse_dates=["start_date", "end_date"]
            )
        files = self._list_files()
        df = _index_from_files(files).sort_values(["name", "start_date"])
        self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        df.to_csv(tmp_path, index=False)
        try:
            os.replace(tmp_path, self.index_cache_path)
        except FileExistsError:
            tmp_path.unlink(missing_ok=True)
        return df.reset_index(drop=True)

    def _list_files(self) -> list[str]:
        storage_options = _storage_options(self.profile)
        files = []
        for root in self.roots:
            fs = (
                fsspec.filesystem("s3", **storage_options)
                if root.startswith("s3://")
                else fsspec.filesystem("file")
            )
            try:
                paths = fs.ls(root)
            except FileNotFoundError:
                logger.warning("Skipping missing ERA5 source root %s", root)
                continue
            for path in paths:
                if isinstance(path, dict):
                    path = path["name"]
                if root.startswith("s3://") and not path.startswith("s3://"):
                    path = f"s3://{path}"
                files.append(path)
        return files


def _storage_options(profile: str | None) -> dict:
    if profile is None:
        # Public NCAR ERA5 access should be anonymous even if the caller's shell
        # has an unrelated AWS_PROFILE set for internal buckets.
        os.environ.pop("AWS_PROFILE", None)
        os.environ.pop("AWS_DEFAULT_PROFILE", None)
        return {"anon": True, "client_kwargs": {"region_name": "us-east-1"}}
    return healda.utils.storage.get_storage_options(profile) or {}


def _level_index(da: xr.DataArray, levels: Sequence[int], source_path: str) -> dict:
    if not levels:
        return {}
    index = {int(value): i for i, value in enumerate(da["level"].values)}
    missing = [level for level in levels if level not in index]
    if missing:
        raise KeyError(f"{source_path} has no levels {missing}")
    return index


def read_era5_source_group(
    *,
    file_index: Era5FileIndex,
    raw_name: str,
    times: Sequence[pd.Timestamp],
    levels: Sequence[int] = (),
    cache_dir: str | Path | None = None,
    profile: str | None = None,
) -> dict[int | None, np.ndarray]:
    times = [pd.Timestamp(time) for time in times]
    storage_options = _storage_options(profile)
    cache = AtomicFileCache(cache_dir) if cache_dir else None

    output_levels: tuple[int | None, ...]
    output_levels = tuple(levels) if levels else (None,)
    out = {
        level: np.full((len(times), 721, 1440), np.nan, dtype=np.float32)
        for level in output_levels
    }

    by_file: dict[str, list[tuple[int, pd.Timestamp]]] = {}
    for time_idx, time in enumerate(times):
        path = file_index.resolve_file(raw_name, time)
        by_file.setdefault(path, []).append((time_idx, time))

    for source_path, requests in by_file.items():
        read_path = (
            str(cache.ensure(source_path, storage_options)) if cache else source_path
        )
        open_kwargs = {}
        if not cache and storage_options:
            open_kwargs["storage_options"] = storage_options
        with xr.open_dataset(
            read_path, engine="h5netcdf", chunks=None, **open_kwargs
        ) as ds:
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            short_name_map = {
                ds[var].attrs.get("short_name"): var
                for var in ds.data_vars
                if "short_name" in ds[var].attrs
            }
            var_name = short_name_map.get(raw_name)
            if var_name is None:
                raise KeyError(
                    f"No variable with short_name={raw_name!r} in {source_path}. "
                    f"Available: {sorted(short_name_map)}"
                )

            da = ds[var_name].rename({lat_name: "lat", lon_name: "lon"})
            level_index = _level_index(da, levels, source_path)
            for time_idx, time in requests:
                selected = da.sel(time=time, method="nearest")
                selected_time = pd.Timestamp(selected.time.values)
                if abs(selected_time - time) > pd.Timedelta(minutes=1):
                    raise ValueError(
                        f"Requested {time}, got {selected_time} from {source_path}"
                    )
                if levels:
                    # The HDF5 chunk spans all levels of a timestep, so selecting levels
                    # one at a time re-inflates the whole chunk per level, measured at
                    # 11x. Materialize the chunk once and index it in memory.
                    values = np.asarray(
                        selected.transpose("level", "lat", "lon").values,
                        dtype=np.float32,
                    )
                    for level in levels:
                        out[level][time_idx] = values[level_index[level]]
                else:
                    out[None][time_idx] = selected.squeeze().values.astype(np.float32)

    return out


def read_era5_static_fields(
    *,
    cache_dir: str | Path | None = None,
    profile: str | None = None,
) -> dict[str, np.ndarray]:
    storage_options = _storage_options(profile)
    cache = AtomicFileCache(cache_dir) if cache_dir else None
    fields = {}
    for output_name, source_path in ERA5_STATIC_FILES.items():
        read_path = (
            str(cache.ensure(source_path, storage_options)) if cache else source_path
        )
        open_kwargs = {}
        if not cache and storage_options:
            open_kwargs["storage_options"] = storage_options
        with xr.open_dataset(
            read_path, engine="h5netcdf", chunks=None, **open_kwargs
        ) as ds:
            short_name = ERA5_STATIC_RAW_NAMES[output_name]
            short_name_map = {
                ds[var].attrs.get("short_name"): var
                for var in ds.data_vars
                if "short_name" in ds[var].attrs
            }
            var_name = short_name_map.get(short_name)
            if var_name is None:
                raise KeyError(
                    f"No variable with short_name={short_name!r} in {source_path}. "
                    f"Available: {sorted(short_name_map)}"
                )
            values = ds[var_name].squeeze().values.astype(np.float32)
        if output_name == "orog":
            values = values / np.float32(STANDARD_GRAVITY)
        fields[output_name] = values
    return fields
