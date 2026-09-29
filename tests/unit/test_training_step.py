# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from healda.datasets.base import NormalizationStats
from healda.training import step


def test_normalization_stats_requires_batched_packed_tensors():
    stats = NormalizationStats(center=[10.0, -2.0], scales=[2.0, 4.0])

    batched = torch.zeros(1, 2, 1, 1)
    torch.testing.assert_close(
        stats.denormalize(batched), torch.tensor([[[[10.0]], [[-2.0]]]])
    )

    with pytest.raises(ValueError, match="expects at least 4 dims"):
        stats.denormalize(torch.zeros(2, 1, 1))


def test_prepare_loss_target_and_physical_fields_full_mode_uses_target_stats():
    stats = NormalizationStats(center=[10.0, -2.0], scales=[2.0, 4.0])
    target = torch.zeros(1, 2, 1, 1)
    model_output = torch.ones(1, 2, 1, 1)

    loss_target, physical_prediction, physical_target = (
        step.prepare_loss_target_and_physical_fields(
            model_output=model_output,
            target=target,
            background=None,
            prediction_mode="full",
            target_stats=stats,
        )
    )

    torch.testing.assert_close(loss_target, target)
    torch.testing.assert_close(
        physical_target,
        torch.tensor([[[[10.0]], [[-2.0]]]]),
    )
    torch.testing.assert_close(
        physical_prediction,
        torch.tensor([[[[12.0]], [[2.0]]]]),
    )


def test_prepare_loss_target_and_physical_fields_residual_mode_reconstructs_full_state():
    target_stats = NormalizationStats(center=[10.0], scales=[2.0])
    background_stats = NormalizationStats(center=[8.0], scales=[4.0])
    residual_stats = NormalizationStats(center=[1.0], scales=[0.5])

    target = torch.tensor([[[[2.0]]]])  # physical 14
    background = torch.tensor([[[[0.5]]]])  # physical 10
    model_output = torch.tensor([[[[4.0]]]])  # residual physical 3

    loss_target, physical_prediction, physical_target = (
        step.prepare_loss_target_and_physical_fields(
            model_output=model_output,
            target=target,
            background=background,
            prediction_mode="residual",
            target_stats=target_stats,
            residual_stats=residual_stats,
            background_stats=background_stats,
        )
    )

    # target residual physical = 14 - 10 = 4, normalized as (4 - 1) / 0.5 = 6
    torch.testing.assert_close(loss_target, torch.tensor([[[[6.0]]]]))
    torch.testing.assert_close(physical_target, torch.tensor([[[[14.0]]]]))
    torch.testing.assert_close(physical_prediction, torch.tensor([[[[13.0]]]]))


def test_residual_mode_indexes_target_aligned_background_channels():
    # Background channel 0 is context; channel 1 is the residual base selected
    # by residual_base_index.
    target_stats = NormalizationStats(center=[10.0], scales=[2.0])
    background_stats = NormalizationStats(center=[0.0, 8.0], scales=[1.0, 4.0])
    residual_stats = NormalizationStats(center=[1.0], scales=[0.5])

    target = torch.tensor([[[[2.0]]]])  # physical 14
    background = torch.tensor([[[[99.0]], [[0.5]]]])  # ch1 physical 10, ch0 context
    model_output = torch.tensor([[[[4.0]]]])  # residual physical 3

    loss_target, physical_prediction, physical_target = (
        step.prepare_loss_target_and_physical_fields(
            model_output=model_output,
            target=target,
            background=background,
            prediction_mode="residual",
            target_stats=target_stats,
            residual_stats=residual_stats,
            background_stats=background_stats,
            residual_base_index=torch.tensor([1], dtype=torch.long),
        )
    )

    # base physical 10: residual target (14 - 10 = 4) normalized (4 - 1) / 0.5 = 6.
    torch.testing.assert_close(loss_target, torch.tensor([[[[6.0]]]]))
    torch.testing.assert_close(physical_target, torch.tensor([[[[14.0]]]]))
    torch.testing.assert_close(physical_prediction, torch.tensor([[[[13.0]]]]))


def test_prepare_loss_target_and_physical_fields_residual_mode_requires_background_and_stats():
    target_stats = NormalizationStats(center=[0.0], scales=[1.0])
    tensor = torch.zeros(1, 1, 1, 1)

    with pytest.raises(ValueError, match="background is required"):
        step.prepare_loss_target_and_physical_fields(
            model_output=tensor,
            target=tensor,
            background=None,
            prediction_mode="residual",
            target_stats=target_stats,
            residual_stats=target_stats,
        )

    with pytest.raises(ValueError, match="residual_stats is required"):
        step.prepare_loss_target_and_physical_fields(
            model_output=tensor,
            target=tensor,
            background=tensor,
            prediction_mode="residual",
            target_stats=target_stats,
        )


def test_prepare_loss_target_and_physical_fields_config_without_transform_is_unaffected():
    # config_name given but absent from state_transforms.TRANSFORMS -- must behave
    # exactly like the channels=None/config_name=None default (regression safety for
    # every non-era5_104ch caller, including ones that now pass their real config_name).
    stats = NormalizationStats(center=[10.0, -2.0], scales=[2.0, 4.0])
    target = torch.zeros(1, 2, 1, 1)
    model_output = torch.ones(1, 2, 1, 1)

    loss_target, physical_prediction, physical_target = (
        step.prepare_loss_target_and_physical_fields(
            model_output=model_output,
            target=target,
            background=None,
            prediction_mode="full",
            target_stats=stats,
            channels=["a", "b"],
            config_name="era5_99ch",
        )
    )

    torch.testing.assert_close(loss_target, target)
    torch.testing.assert_close(physical_target, torch.tensor([[[[10.0]], [[-2.0]]]]))
    torch.testing.assert_close(physical_prediction, torch.tensor([[[[12.0]], [[2.0]]]]))


def test_prepare_loss_target_and_physical_fields_inverts_tcw_residual():
    # target_stats denormalizes to the MODEL-SPACE values TransformV2 loaded: tcwv
    # physical (20.0) and tcw's residual physical (0.5, i.e. tcw - tcwv). Inverting
    # must reconstruct true physical tcw = 20.0 + 0.5 = 20.5, tcwv untouched.
    stats = NormalizationStats(center=[20.0, 0.5], scales=[1.0, 1.0])
    target = torch.zeros(1, 2, 1, 1)
    model_output = torch.zeros(1, 2, 1, 1)

    _, physical_prediction, physical_target = (
        step.prepare_loss_target_and_physical_fields(
            model_output=model_output,
            target=target,
            background=None,
            prediction_mode="full",
            target_stats=stats,
            channels=["tcwv", "tcw"],
            config_name="era5_104ch",
        )
    )

    torch.testing.assert_close(physical_target, torch.tensor([[[[20.0]], [[20.5]]]]))
    torch.testing.assert_close(
        physical_prediction, torch.tensor([[[[20.0]], [[20.5]]]])
    )


def test_run_model_step_channel_mask_zeroes_and_rescales_loss():
    stats = NormalizationStats(center=[0.0], scales=[1.0])
    # 1 channel, 4 pixels: model is wrong everywhere by the same amount (error=2 -> mse=4),
    # but only pixels 0 and 1 are inside this channel's valid domain.
    batch = step.StepInput(
        target=torch.zeros(1, 1, 1, 4),
        condition=torch.ones(1, 1, 1, 4),
        second_of_day=torch.zeros(1, 1),
        day_of_year=torch.zeros(1, 1),
        timestamp=torch.zeros(1, 1, dtype=torch.long),
        unified_obs=None,
        labels=torch.empty(1, 0),
    )
    valid = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 1, 1, 4)
    rescale = 2.0  # valid_fraction = 0.5 -> 1/0.5
    channel_mask = valid * rescale

    def forward_fn(condition, **kwargs):
        class Prediction:
            out = torch.full_like(batch.target, 2.0)

        return Prediction()

    out = step.run_model_step(
        batch=batch,
        prediction_mode="full",
        target_stats=stats,
        residual_stats=None,
        background_stats=stats,
        is_causal=True,
        huber_delta=0.1,
        loss_type="mse",
        loss_weights=None,
        forward_fn=forward_fn,
        channel_mask=channel_mask,
    )

    # Masked pixels are zero; valid pixels carry the raw mse (2**2) scaled by
    # 1/valid_fraction. loss is the mean over the valid domain alone.
    torch.testing.assert_close(out.loss, torch.tensor(4.0))
    torch.testing.assert_close(out.mse, torch.tensor([[[[8.0, 8.0, 0.0, 0.0]]]]))


def test_run_model_step_appends_background_to_condition():
    stats = NormalizationStats(center=[0.0], scales=[1.0])
    batch = step.StepInput(
        target=torch.zeros(1, 1, 1, 2),
        condition=torch.ones(1, 2, 1, 2),
        second_of_day=torch.zeros(1, 1),
        day_of_year=torch.zeros(1, 1),
        timestamp=torch.zeros(1, 1, dtype=torch.long),
        unified_obs=None,
        labels=torch.empty(1, 0),
        background=torch.full((1, 1, 1, 2), 3.0),
    )
    captured = {}

    class Prediction:
        def __init__(self, out):
            self.out = out

    def forward_fn(condition, **kwargs):
        captured["condition"] = condition
        return Prediction(torch.zeros_like(batch.target))

    out = step.run_model_step(
        batch=batch,
        prediction_mode="full",
        target_stats=stats,
        residual_stats=None,
        background_stats=stats,
        is_causal=True,
        huber_delta=0.1,
        loss_type="mse",
        loss_weights=None,
        forward_fn=forward_fn,
    )

    expected_condition = torch.cat([batch.condition, batch.background], dim=1)
    torch.testing.assert_close(captured["condition"], expected_condition)
    torch.testing.assert_close(out.loss, torch.tensor(0.0))


def test_temporal_loss_weights_single_rank_multiple_frames():
    weight = step.build_loss_weights(
        4,
        channel_loss_weights=None,
        weight_start=0.25,
        time_size=1,
        time_rank=0,
        power=1.0,
        device=None,
    )

    expected_weights = torch.linspace(0.25, 1.0, 4)
    expected_weights = expected_weights / expected_weights.mean()
    torch.testing.assert_close(weight, expected_weights.view(1, 4))


def test_temporal_loss_weights_slices_rank_frames_when_local_time_is_multiple_frames():
    weight = step.build_loss_weights(
        2,
        channel_loss_weights=None,
        weight_start=0.5,
        time_size=3,
        time_rank=1,
        power=1.0,
        device=None,
    )

    expected_weights = torch.linspace(0.5, 1.0, 6)
    expected_weights = expected_weights / expected_weights.mean()
    torch.testing.assert_close(weight, expected_weights[2:4].view(1, 2))


def test_loss_weights_combines_temporal_and_channel():
    channel = torch.tensor([2.0, 0.5])
    weight = step.build_loss_weights(
        4,
        channel_loss_weights=channel,
        weight_start=0.25,
        time_size=1,
        time_rank=0,
        power=1.0,
        device=None,
    )

    temporal = torch.linspace(0.25, 1.0, 4)
    temporal = temporal / temporal.mean()
    expected = temporal.view(1, 4) * channel.view(2, 1)
    torch.testing.assert_close(weight, expected)


def test_loss_weights_none_when_inactive():
    assert (
        step.build_loss_weights(
            4,
            channel_loss_weights=None,
            weight_start=1.0,
            time_size=1,
            time_rank=0,
            power=1.0,
            device=None,
        )
        is None
    )
