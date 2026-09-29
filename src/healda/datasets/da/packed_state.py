# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-store frame reader for a packed ``(time, variable, x)`` state store.

``PackedStateReader`` reads channels, frames, static fields, and per-frame
validity for one packed HPX64 store, in its own store time.
``PackedDADataset`` pairs readers with query offsets and composes a sample.
"""

import numpy as np
import pandas as pd

from healda.datasets.catalog import _Zarr
from healda.config.variables import resolve_store_channels

SOURCE_STATUS_WRITTEN_VALID = 1


def _decode(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _locate_channel(store_names: list[str], channel: str) -> int:
    """Position of ``channel`` in the store, falling back to a case-insensitive match.

    Stores disagree on capitalisation: the packed HPX64 zarr holds `W1000` but
    `u10`/`v10`/`t10`/`z10`, while the model asks for `U10`. Exact match wins, so
    a store that is already consistent is unaffected. Remove the fallback once the
    store's `variable` coordinate is renamed.
    """
    try:
        return store_names.index(channel)
    except ValueError:
        pass

    folded = [
        i for i, name in enumerate(store_names) if name.lower() == channel.lower()
    ]
    if len(folded) == 1:
        return folded[0]
    if not folded:
        raise KeyError(
            f"channel {channel!r} is not in the store; it has {len(store_names)}"
            f" channels starting {store_names[:4]}"
        )
    raise KeyError(
        f"channel {channel!r} matches {len(folded)} store channels case-insensitively"
        f" ({[store_names[i] for i in folded]}); rename them to disambiguate"
    )


class PackedStateReader:
    def __init__(
        self,
        entry: _Zarr,
        channels: list[str],
        *,
        source: str = "",
        time_slice: slice | None = None,
    ):
        dataset = entry.to_xarray(chunks=None)
        data = dataset["data"]
        if time_slice is not None:
            data = data.sel(time=time_slice)
        store_channels = resolve_store_channels(list(channels), source)
        # Positions of the requested channels in the store's channel axis,
        # applied in NumPy after a read (see _read_frames).
        store_names = _decode(data["variable"].values)
        self._channel_pos = np.array(
            [_locate_channel(store_names, c) for c in store_channels]
        )
        self.channels = list(channels)
        self.data = data  # full channel axis retained
        self._dataset = dataset

    @property
    def times(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.data["time"].values)

    def restrict_times(self, mask) -> "PackedStateReader":
        self.data = self.data.isel(time=mask)
        return self

    def _read_frames(self, time_indices) -> np.ndarray:
        # zarr basic indexing (a slice or scalar) is ~5x faster than fancy
        # orthogonal selection (an integer-array time index, or a non-contiguous
        # channel subset) for identical bytes. So read contiguous time as one
        # slice (else per-frame scalar reads, no over-read for spread frames) and
        # subset channels in NumPy -- never fancy-index either axis.
        t = np.asarray(time_indices)
        if t.size and np.array_equal(t, np.arange(t[0], t[0] + t.size)):
            block = self.data.isel(time=slice(int(t[0]), int(t[-1]) + 1)).values
        else:
            block = np.stack([self.data.isel(time=int(i)).values for i in t])
        return block[..., self._channel_pos, :]

    def values_at(self, coords):
        return self._read_frames(np.asarray(coords["time"]))

    def read(self, store_idx) -> np.ndarray:
        """Read frames at positional indices into this reader's time axis."""
        return self._read_frames(np.asarray(store_idx))

    def index_of(self, times) -> np.ndarray:
        """Positional indices of ``times`` in this reader's time axis (raises if absent)."""
        idx = self.data.indexes["time"].get_indexer(pd.DatetimeIndex(times))
        if (idx == -1).any():
            raise KeyError("PackedStateReader missing requested times")
        return idx

    def static(self, names: list[str]) -> np.ndarray:
        """Read static fields (e.g. orog, land_surface) as a ``(len(names), x)`` array."""
        store_names = _decode(self._dataset["static_variable"].values)
        index = [store_names.index(name) for name in names]
        static_data = np.asarray(self._dataset["static_data"].values, dtype=np.float32)
        return static_data[index]

    def valid_store_mask(self, times) -> np.ndarray:
        """Mask over ``times``, True where every channel was written. Absent
        times are False; read against the full axis to ignore prior subsetting."""
        times = pd.DatetimeIndex(times)
        full_times = pd.to_datetime(self._dataset["data"]["time"].values)
        idx = full_times.get_indexer(times)
        mask = np.zeros(len(times), dtype=bool)
        present = idx != -1
        mask[present] = self._valid_mask(idx[present]).all(axis=1)
        return mask

    def _valid_mask(self, time_idx: np.ndarray) -> np.ndarray:
        # (len(time_idx), len(self._channel_pos)) validity over the requested
        # (aliased) channels, read from the store's is_valid / source_status arrays.
        time_idx = np.asarray(time_idx)
        valid = np.ones((len(time_idx), len(self._channel_pos)), dtype=bool)
        if "is_valid" in self._dataset:
            is_valid = self._dataset["is_valid"]
            arr = np.asarray(is_valid.values, dtype=bool)
            if is_valid.ndim == 1:
                valid &= arr[time_idx][:, None]
            else:
                valid &= arr[np.ix_(time_idx, self._channel_pos)]
        if "source_status" in self._dataset:
            status = np.asarray(self._dataset["source_status"].values, dtype=np.uint8)
            valid &= (
                status[np.ix_(time_idx, self._channel_pos)]
                == SOURCE_STATUS_WRITTEN_VALID
            )
        return valid
