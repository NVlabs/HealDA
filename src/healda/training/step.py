# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Literal

import torch

from healda.datasets.base import NormalizationStats


@dataclasses.dataclass
class StepInput:
    target: torch.Tensor
    condition: torch.Tensor
    second_of_day: torch.Tensor
    day_of_year: torch.Tensor
    timestamp: torch.Tensor
    unified_obs: object | None
    labels: torch.Tensor | None
    background: torch.Tensor | None = None
    subdomain: object | None = None


@dataclasses.dataclass
class StepOutput:
    # Tensor compared to model output for loss. Full mode: normalized target.
    # Residual mode: normalized target-background increment.
    loss_target: torch.Tensor
    model_output: torch.Tensor
    mse: torch.Tensor
    huber_loss: torch.Tensor
    loss: torch.Tensor
    # Full-state tensors in physical units, used for metrics/visualization.
    physical_prediction: torch.Tensor
    physical_target: torch.Tensor
    unified_obs: object | None


def prepare_loss_target_and_physical_fields(
    *,
    model_output: torch.Tensor,
    target: torch.Tensor,
    background: torch.Tensor | None,
    prediction_mode: Literal["full", "residual"],
    target_stats: NormalizationStats,
    residual_stats: NormalizationStats | None = None,
    background_stats: NormalizationStats | None = None,
    residual_base_index: torch.Tensor | None = None,
    channels: list[str] | None = None,
    config_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``channels``/``config_name`` invert TransformV2's state_transforms.to_model_space
    (e.g. tcw's residual) so physical_target/physical_prediction are true physical
    values, not the model-space quantity. None (the default) is a no-op, matching every
    caller that predates this -- as is a config_name absent from state_transforms.TRANSFORMS.
    """

    def _to_physical_space(state):
        if channels is None or config_name is None:
            return state
        from healda.datasets.da import state_transforms

        # channel_axis=1 explicitly: the default -3 is the time axis here.
        return state_transforms.to_physical_space(
            state, channels, config_name, channel_axis=1
        )

    physical_target = _to_physical_space(target_stats.denormalize(target))

    if prediction_mode == "full":
        return (
            target,
            _to_physical_space(target_stats.denormalize(model_output)),
            physical_target,
        )

    if prediction_mode != "residual":
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")

    if background is None:
        raise ValueError("background is required when prediction_mode='residual'")
    if residual_stats is None:
        raise ValueError("residual_stats is required when prediction_mode='residual'")

    background_stats = target_stats if background_stats is None else background_stats
    physical_base = background_stats.denormalize(background)
    if residual_base_index is not None:
        physical_base = physical_base.index_select(
            1, residual_base_index.to(physical_base.device)
        )
    # physical_target was already inverted above, so residual_stats characterizes the
    # delta between two already-physical quantities -- this sum is physical directly,
    # NOT model-space, and must not go through _to_physical_space a second time.
    physical_prediction = physical_base + residual_stats.denormalize(model_output)

    return (
        residual_stats.normalize(physical_target - physical_base),
        physical_prediction,
        physical_target,
    )


def _validate_step_input(
    batch: StepInput,
    prediction_mode: str = "full",
    residual_base_index: torch.Tensor | None = None,
):
    # A latlon target is (b, c, t, nlat, nlon) while the condition stays on HPX cells, so
    # there only batch and time are comparable. An HPX target shares the cell axis too.
    b, _, t, *spatial = batch.target.shape
    cond = tuple(batch.condition.shape)
    shares_cells = len(spatial) == 1
    if cond[0] != b or cond[2] != t or (shares_cells and cond[3] != spatial[0]):
        raise ValueError(
            f"condition shape {cond} must match target {tuple(batch.target.shape)} in "
            f"batch and time{' and cells' if shares_cells else ''}"
        )
    if batch.background is not None:
        bb, bc, bt, *bspatial = batch.background.shape
        if (bb, bt, bspatial) != (b, t, spatial):
            raise ValueError(
                f"background shape {tuple(batch.background.shape)} must match target "
                f"{tuple(batch.target.shape)} in batch, time and space"
            )
        if prediction_mode == "residual":
            n = batch.target.shape[1]
            if residual_base_index is None and bc != n:
                raise ValueError(
                    f"residual mode needs background C={bc} == target C={n} "
                    "or a residual_base_index"
                )
            if residual_base_index is not None and (
                residual_base_index.numel() != n
                or int(residual_base_index.min()) < 0
                or int(residual_base_index.max()) >= bc
            ):
                raise ValueError(
                    f"residual_base_index must select {n} in-range channels "
                    f"from background C={bc}"
                )


def temporal_loss_weights(
    time_length: int,
    *,
    weight_start: float,
    power: float = 1.0,
    device=None,
) -> torch.Tensor:
    """Per-frame weights ``weight_start + (1 - weight_start) * s ** power`` over
    ``s = i / (T - 1) in [0, 1]``, normalized to mean 1.

    ``weight_start`` is the first:last frame weight ratio (1.0 = uniform);
    ``power`` shapes the interior (1 = linear, <1 = steep-early then flattens).
    """
    if time_length <= 1 or weight_start >= 1.0:
        return torch.ones(max(time_length, 1), device=device)

    s = torch.linspace(0.0, 1.0, time_length, device=device)
    weights = weight_start + (1.0 - weight_start) * s**power
    return weights / weights.mean()


def _temporal_weight_vector(
    local_time_length: int,
    *,
    weight_start: float,
    time_size: int,
    time_rank: int,
    power: float = 1.0,
    device=None,
) -> torch.Tensor | None:
    """The (local_time_length,) slice of temporal weights for this time rank, or
    None when temporal weighting is off / degenerate."""
    if weight_start >= 1.0 or local_time_length * time_size <= 1:
        return None
    weights = temporal_loss_weights(
        local_time_length * time_size,
        weight_start=weight_start,
        power=power,
        device=device,
    )
    start = time_rank * local_time_length
    return weights[start : start + local_time_length]


def build_loss_weights(
    n_frames: int,
    *,
    channel_loss_weights: torch.Tensor | None,
    weight_start: float,
    time_size: int,
    time_rank: int,
    power: float,
    device,
) -> torch.Tensor | None:
    # Combined per-frame and per-channel loss weight (C, T), or None if neither is active.
    # Shaped for the reduced error, which is rank-generic where a (1, C, T, 1) view is not.
    temporal_weight = _temporal_weight_vector(
        n_frames,
        weight_start=weight_start,
        time_size=time_size,
        time_rank=time_rank,
        power=power,
        device=device,
    )
    weight = None
    if temporal_weight is not None:
        weight = temporal_weight.to(torch.float32).view(1, -1)
    if channel_loss_weights is not None:
        channel_weight = channel_loss_weights.to(device, torch.float32).view(-1, 1)
        weight = channel_weight if weight is None else weight * channel_weight
    return weight


def _check_spatial(spatial: torch.Tensor | None) -> None:
    """Kept out of the reduction so the assert never lands inside a compiled region."""
    if spatial is None:
        return
    mean = spatial.mean()
    if not torch.isclose(mean, torch.ones_like(mean), atol=1e-4):
        raise ValueError(
            f"spatial weight must be normalized to mean 1 over latitude, got {float(mean)}"
        )


def _reduce_weighted(error: torch.Tensor, spatial: torch.Tensor | None) -> torch.Tensor:
    """Area-weighted error as ``(C, T, W)``, W being the axis the weight varies along:
    latitude on lat/lon, a singleton on equal-area HEALPix.

    The smallest tensor the loss and every metric are still exactly derivable from.
    """
    reduced = error.mean(dim=(0, error.ndim - 1), dtype=torch.float32)
    if error.ndim != 5:
        reduced = reduced.unsqueeze(-1)
    return reduced if spatial is None else reduced * spatial.reshape(1, 1, -1)


def last_frame_rmse_per_channel(
    squared_error: torch.Tensor, spatial: torch.Tensor | None
) -> torch.Tensor:
    """``(C,)`` RMSE of the last frame, weighted like the loss.

    Derived from the same reduction rather than hardcoded axes, which are rank-specific.
    """
    _check_spatial(spatial)
    return torch.sqrt(_reduce_weighted(squared_error, spatial)[:, -1].mean(dim=-1))


def run_model_step(
    *,
    batch: StepInput,
    prediction_mode: Literal["full", "residual"],
    target_stats: NormalizationStats,
    residual_stats: NormalizationStats | None,
    background_stats: NormalizationStats,
    residual_base_index: torch.Tensor | None = None,
    is_causal: bool,
    huber_delta: float,
    loss_type: str,
    loss_weights: torch.Tensor | None,
    forward_fn,
    channel_mask: torch.Tensor | None = None,
    physical_channels: list[str] | None = None,
    physical_config_name: str | None = None,
    spatial_loss_weight: torch.Tensor | None = None,
) -> StepOutput:
    """``channel_mask``: an optional (1, C, 1, *spatial) multiplier from
    ``state_masks.loss_weight_mask``/``loss_rescale_mask``/``hpx_loss_rescale_mask``,
    already reordered to this run's pixel_order. Masked channels (e.g. sst over land)
    are unconstrained there, so without this the model gets gradient from -- and every
    logged per-channel metric is diluted by -- a region that was never meant to be
    scored. None (the default) is a no-op.

    ``physical_channels``/``physical_config_name``: forwarded to
    ``prepare_loss_target_and_physical_fields`` to invert a model-space transform
    (e.g. tcw's residual) for physical_prediction/physical_target. None is a no-op.
    """
    _validate_step_input(batch, prediction_mode, residual_base_index)

    unified_obs = batch.unified_obs

    condition = batch.condition
    if batch.background is not None:
        condition = torch.cat([condition, batch.background], dim=1)

    noise_labels = torch.zeros([batch.target.shape[0]], device=batch.target.device)
    prediction = forward_fn(
        condition,
        noise_labels=noise_labels,
        class_labels=batch.labels,
        second_of_day=batch.second_of_day,
        day_of_year=batch.day_of_year,
        unified_obs=unified_obs,
        timestamp=batch.timestamp,
        is_causal=is_causal,
        subdomain=batch.subdomain,
    )
    model_output = prediction.out

    loss_target, physical_prediction, physical_target = (
        prepare_loss_target_and_physical_fields(
            model_output=model_output,
            target=batch.target,
            background=batch.background,
            prediction_mode=prediction_mode,
            target_stats=target_stats,
            residual_stats=residual_stats,
            background_stats=background_stats,
            residual_base_index=residual_base_index,
            channels=physical_channels,
            config_name=physical_config_name,
        )
    )
    mse = (loss_target - model_output) ** 2
    huber_loss = torch.nn.functional.huber_loss(
        loss_target,
        model_output,
        reduction="none",
        delta=huber_delta,
    )
    if channel_mask is not None:
        channel_mask = channel_mask.to(mse.device, mse.dtype)
        mse = mse * channel_mask
        huber_loss = huber_loss * channel_mask
    if loss_type == "mse":
        loss = mse
    elif loss_type == "huber":
        loss = huber_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Reduce first, then weight: the weight varies on no reduced axis, so the result is
    # identical and the reduced form broadcasts at any input rank. mse/huber_loss are
    # returned unweighted so logged per-channel metrics stay comparable across runs.
    _check_spatial(spatial_loss_weight)
    train_reduced = _reduce_weighted(loss, spatial_loss_weight)
    if loss_weights is not None:
        train_reduced = train_reduced * loss_weights.unsqueeze(-1)
    loss = train_reduced.mean()

    return StepOutput(
        loss_target=loss_target,
        model_output=model_output,
        mse=mse,
        huber_loss=huber_loss,
        loss=loss,
        physical_prediction=physical_prediction,
        physical_target=physical_target,
        unified_obs=unified_obs,
    )
