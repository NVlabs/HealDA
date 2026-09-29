# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical <-> model space per channel, applied outside the model.

Two things a channel may need that the network should not know about:

  residual   `tcw` is 99.977% correlated with `tcwv`, so predicting it directly
             spends a channel re-learning column vapour -- the condensate is
             0.05% of its variance. The model predicts `tcw - tcwv` instead.

  fill       masked channels are unconstrained outside their domain (sst over
             land is not scored), so the loader's fill must be re-imposed on
             output or the values there are whatever the network happened to
             emit.

`to_model_space` runs on the target at load; `to_physical_space` runs on the
prediction. Both take the full channel tensor because a transform may read a
sibling channel, and both are exact inverses where a transform is invertible.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import numpy as np
import torch

Array = torch.Tensor | np.ndarray


@dataclasses.dataclass(frozen=True)
class ChannelTransform:
    """Invertible model-space mapping for one channel.

    ``needs`` names sibling channels the transform reads; they are passed
    positionally after the channel's own values.
    """

    channel: str
    forward: Callable[..., Array]
    inverse: Callable[..., Array]
    needs: tuple[str, ...] = ()


TRANSFORMS: dict[str, dict[str, ChannelTransform]] = {
    "era5_104ch": {
        "tcw": ChannelTransform(
            channel="tcw",
            needs=("tcwv",),
            forward=lambda tcw, tcwv: tcw - tcwv,
            inverse=lambda residual, tcwv: residual + tcwv,
        ),
    },
}


def _index(channels: Sequence[str], name: str) -> int:
    try:
        return list(channels).index(name)
    except ValueError as err:
        raise KeyError(f"{name!r} not among the {len(channels)} channels") from err


def _apply(
    state: Array,
    channels: Sequence[str],
    config_name: str,
    direction: str,
    channel_axis: int,
) -> Array:
    transforms = TRANSFORMS.get(config_name)
    if not transforms:
        return state
    out = state.clone() if isinstance(state, torch.Tensor) else state.copy()
    for transform in transforms.values():
        take = [slice(None)] * out.ndim
        take[channel_axis] = _index(channels, transform.channel)
        siblings = []
        for name in transform.needs:
            grab = [slice(None)] * out.ndim
            grab[channel_axis] = _index(channels, name)
            siblings.append(state[tuple(grab)])
        fn = transform.forward if direction == "forward" else transform.inverse
        out[tuple(take)] = fn(state[tuple(take)], *siblings)
    return out


def to_model_space(
    state: Array, channels: Sequence[str], config_name: str, channel_axis: int = -3
) -> Array:
    """Physical values -> what the model predicts."""
    return _apply(state, channels, config_name, "forward", channel_axis)


def to_physical_space(
    state: Array, channels: Sequence[str], config_name: str, channel_axis: int = -3
) -> Array:
    """Model output -> physical values.

    Siblings are read from the model's own output, so a channel a transform
    depends on must not itself be transformed -- `tcwv` is predicted directly.
    """
    return _apply(state, channels, config_name, "inverse", channel_axis)


def restore_fill(
    state: Array,
    channels: Sequence[str],
    mask_file: str,
    pixel_order: str | None = None,
    channel_axis: int = -3,
) -> Array:
    """Re-impose the loader's fill outside each masked channel's domain.

    A masked channel is unconstrained where it was not scored, so without this
    the region carries whatever the network emitted. The mask broadcasts over
    the trailing spatial dims, so this works for (..., C, N) and (..., C, H, W).

    ``pixel_order`` must be the run's order: model output has been reordered out
    of the store's NEST, and applying a NEST mask to it masks the wrong cells
    with no error.
    """
    from healda.datasets.da.state_masks import CHANNEL_DOMAIN, domain_fill, state_mask

    out = state.clone() if isinstance(state, torch.Tensor) else state.copy()
    for channel, domain in CHANNEL_DOMAIN.items():
        if channel not in channels:
            continue
        valid = state_mask(domain, mask_file, pixel_order)
        if isinstance(out, torch.Tensor):
            valid = torch.as_tensor(valid, device=out.device)
        take = [slice(None)] * out.ndim
        take[channel_axis] = _index(channels, channel)
        sub = out[tuple(take)]
        sub[..., ~valid] = domain_fill(channel)
        out[tuple(take)] = sub
    return out
