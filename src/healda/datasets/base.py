# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Protocol, Any
from enum import Enum
import numpy as np
from healda.domain import Domain
import torch
import dataclasses
import json
from datetime import timedelta


def _channel_tensor(values, x: torch.Tensor) -> torch.Tensor:
    """``values`` broadcast against ``x``'s channel axis.

    Uncached on purpose. After ``NormalizationStats.to(device)`` the values are
    already float32 on the right device, so ``.to`` returns them unchanged and
    ``.view`` is free -- no host->device copy and no sync. Without that hoist this
    is a pageable H2D per call, which is why it must be hoisted, not memoized.

    The broadcast shape follows ``x``'s rank: spatial is ``X`` on HPX and ``nlat, nlon``
    on lat/lon, so a fixed four-dim view would line C up against T on the latter.
    """
    if x.ndim < 4:
        raise ValueError(
            f"NormalizationStats expects at least 4 dims (B, C, T, ...), got {x.ndim}"
        )
    return torch.as_tensor(values).to(x).view((1, -1) + (1,) * (x.ndim - 2))


@dataclasses.dataclass(frozen=True)
class NormalizationStats:
    """Per-channel affine normalization for channel-first tensors.

    Training batches are ``(B, C, T, ...)``: a packed HEALPix map is ``(B, C, T, X)``,
    a 0.25 degree target is ``(B, C, T, nlat, nlon)``. Only the channel axis matters.

    Call ``to(device)`` once at setup; ``normalize``/``denormalize`` then cost no
    host->device traffic. Uncalled, they still work from host arrays.
    """

    center: Any
    scales: Any

    def to(self, device) -> "NormalizationStats":
        """This instance with center/scales as float32 tensors on ``device``.

        Stored flat; ``_channel_tensor`` views them to the caller's rank per call.
        """

        def as_dev(values):
            return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(
                -1
            )

        return NormalizationStats(
            center=as_dev(self.center), scales=as_dev(self.scales)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return (x - _channel_tensor(self.center, x)) / _channel_tensor(self.scales, x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # Physical-unit math should be fp32, particularly for denormalizing full-state fields.
        # As they sit on huge means (e.g. Z100 ~1.6e5) where the bf16 mantissa step exceeds
        # the dynamic range, resulting in heavy qunatization.
        x = x.float()
        return x * _channel_tensor(self.scales, x) + _channel_tensor(self.center, x)


class TimeUnit(Enum):
    """Time units supported by the dataset.
    Values are the pandas frequency strings (offset aliases)"""

    HOUR = "h"
    DAY = "D"
    MINUTE = "min"
    SECOND = "s"

    def to_timedelta(self, steps: float) -> timedelta:
        return {
            TimeUnit.HOUR: timedelta(hours=steps),
            TimeUnit.DAY: timedelta(days=steps),
            TimeUnit.MINUTE: timedelta(minutes=steps),
            TimeUnit.SECOND: timedelta(seconds=steps),
        }[self]


@dataclasses.dataclass
class BatchInfo:
    channels: list[str]
    time_step: int = 1  # Time (in units `time_unit`) between consecutive frames
    time_unit: TimeUnit = TimeUnit.HOUR
    scales: Any | None = None
    center: Any | None = None

    def __post_init__(self):
        if isinstance(self.time_unit, str):
            raise ValueError("Time unit is an str. Should be a TimeUnit.")

    @staticmethod
    def loads(s):
        kw = json.loads(s)

        if "time_unit" in kw:
            kw["time_unit"] = TimeUnit(kw["time_unit"])

        # Ignore deprecated residual normalization field if present
        kw.pop("residual_normalization", None)

        return BatchInfo(**kw)

    def asdict(self):
        """Return a dictionary representation of the BatchInfo, suitable for JSON serialization."""
        out = {}
        out["channels"] = self.channels
        out["time_step"] = self.time_step
        out["time_unit"] = self.time_unit.value  # TimeUnit is always a TimeUnit enum
        # Convert numpy arrays to lists for JSON compatibility
        if self.scales is not None:
            out["scales"] = np.asarray(self.scales).tolist()
        else:
            out["scales"] = None
        if self.center is not None:
            out["center"] = np.asarray(self.center).tolist()
        else:
            out["center"] = None
        return out

    def sel_channels(self, channels: list[str]):
        channels = list(channels)
        index = np.array([self.channels.index(ch) for ch in channels])
        scales = None
        if self.scales is not None:
            scales = np.asarray(self.scales)[index]

        center = None
        if self.center is not None:
            center = np.asarray(self.center)[index]

        return BatchInfo(
            time_step=self.time_step,
            time_unit=self.time_unit,
            channels=channels,
            scales=scales,
            center=center,
        )

    @property
    def normalization(self) -> NormalizationStats:
        return NormalizationStats(center=self.center, scales=self.scales)

    def denormalize(self, x):
        return self.normalization.denormalize(x)

    def get_time_delta(self, t: int) -> timedelta:
        """Gets time offset of the t-th frame in a frame sequence."""
        total_steps = t * self.time_step
        return self.time_unit.to_timedelta(total_steps)


@dataclasses.dataclass
class DatasetMetadata:
    name: str
    start: str
    end: str
    time_step: int  # time between successive data points in `time_unit`
    time_unit: TimeUnit

    @property
    def freq(self) -> str:
        return f"{self.time_step}{self.time_unit.value}"


@dataclasses.dataclass(frozen=True)
class VariableConfig:
    name: str
    variables_2d: list[str]
    variables_3d: list[str]
    levels: list[int]
    variables_static: list[str] = dataclasses.field(default_factory=list)
    #: Levels beyond ``levels``, for a 3D variable carried at levels the rest of the
    #: family isn't -- e.g. ERA5 has U/V/T/Z at 10 hPa but not Q or W.
    extra_levels_by_var: dict[str, list[int]] = dataclasses.field(default_factory=dict)


class SpatioTemporalDataset(Protocol):
    @property
    def domain(self) -> Domain:
        pass

    def __len__(self) -> int:
        pass

    @property
    def num_channels(self) -> int:
        pass

    @property
    def condition_channels(self) -> int:
        pass

    @property
    def augment_channels(self) -> int:
        return 0

    @property
    def label_dim(self) -> int:
        return 0

    @property
    def time_length(self) -> int:
        pass

    @property
    def batch_info(self) -> BatchInfo:
        return BatchInfo(
            channels=[str(i) for i in range(self.num_channels)],
        )

    def metadata(self) -> Any:
        """Unstructured metadata about the dataset and the values it yields

        Can be used to save normalization constants, timestamps, channel names,
        config values, etc. The training code will avoid looking into this, but
        could be useful for inference.

        """
        return {}

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Returns:
            image: shaped (num_channels, time_length, x)
            labels: shaped (label_dim,)
            condition: shaped (condition_channels, time_length, x)
        """
        pass
