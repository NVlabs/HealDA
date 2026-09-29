# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.distributed as dist

from healda.config.models import ModelSensorConfig, SensorEmbedderConfig
from healda.observations.types import UnifiedObservation
from healda.utils.distributed import init
from healda.models import dit
from healda.models.sharding import shard_t, shard_x
from healda.training.metrics import _StepMetricBuffer
from tests.unit.utils.obs_test_utils import create_unified_observation


def _is_multi_rank_launch():
    """Check if launched with torchrun/mpirun with multiple ranks."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


requires_multi_gpu = pytest.mark.skipif(
    not _is_multi_rank_launch(),
    reason="Requires torchrun with >=2 ranks (torchrun --nproc_per_node=2)",
)


@requires_multi_gpu
def test_step_metrics_use_global_sum_and_count():
    if not torch.distributed.is_initialized():
        init()

    rank = dist.get_rank()
    world = dist.get_world_size()
    count = rank + 1
    values = torch.full((count,), float(count), device="cuda")
    buffer = _StepMetricBuffer()
    buffer.update("loss", values)
    buffer.end_step(cur_nimg=1)

    keys, means, steps = buffer.drain()

    expected = sum(i * i for i in range(1, world + 1)) / sum(range(1, world + 1))
    assert keys == ["loss"] and steps == [1]
    torch.testing.assert_close(means[0, 0], torch.tensor(expected, device="cuda"))


@requires_multi_gpu
def test_sharding_routines():
    if not torch.distributed.is_initialized():
        init()

    group_size = 2
    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)

    b, c, t, x = 1, 3, 2, 8

    tensor = torch.arange(b * c * t * x).view(b, t, x, c).cuda()
    out = shard_t(tensor, group)

    assert out.shape == (b, t // 2, x * 2, c)

    # back
    roundtrip = shard_x(out, group)
    assert torch.all(roundtrip == tensor)


@requires_multi_gpu
@pytest.mark.parametrize("method", ["separable", "context"])
def test_sharding_dit(method):
    if not torch.distributed.is_initialized():
        init()

    group_size = 2

    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)
    level_model = 4  # HPX16

    b, c, t = 1, 3, 2

    model = dit.DiT(
        num_attention_heads=1,
        in_channels=c,
        out_channels=c,
        num_layers=7,
        level_model=level_model,
        temporal_attention=True,
        time_length=2 * group_size,
    )
    if method == "separable":
        model.set_time_parallel_group(group)
    elif method == "context":
        model.set_domain_parallel_group(group)

    x = 12 * 4**model._level_in

    model.cuda()

    tensor = torch.arange(b * c * t * x).view(b, c, t, x).cuda().float()
    noise_labels = torch.ones(b).cuda()
    class_labels = torch.empty([b, 0], device=tensor.device)

    # Transformer engine does not have support for fp32 attention on my desktop
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=method == "context"
    ):
        out = model(
            tensor,
            noise_labels=noise_labels,
            class_labels=class_labels,
            second_of_day=torch.zeros([b, t]).cuda(),
            day_of_year=torch.zeros([b, t]).cuda(),
            timestamp=torch.zeros(1).cuda(),
        )

    assert out.out.shape == tensor.shape


@requires_multi_gpu
@pytest.mark.parametrize(
    "backend,dtype",
    [
        ("diffusers", torch.float32),
        ("te", torch.bfloat16),  # TE's fused attention kernels run in bf16/fp16
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_time_parallel_temporal_attention_invariance(
    backend, dtype, is_causal, monkeypatch
):
    """Time-parallel (separable) sharding must reproduce the single-rank forward.

    Runs the full unsharded forward and the time-parallel sharded forward with
    identical weights and inputs; a logic bug in the temporal-attention / shard_x
    path (frame ordering, emb desync, causal mask) would diverge here.
    """
    # The fp32 case asserts rtol=1e-4, tighter than TF32's own ~3e-4 error; bf16 is unaffected.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    linear_attention = True
    if not torch.distributed.is_initialized():
        init()

    group_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [1, group_size])
    group = mesh.get_group(1)

    level_model = 4  # HPX16
    b, c, t_local = 1, 3, 2
    time_length = t_local * group_size

    torch.manual_seed(0)  # identical init on every rank
    model = dit.DiT(
        num_attention_heads=2,
        in_channels=c,
        out_channels=c,
        num_layers=4,
        level_model=level_model,
        temporal_attention=True,
        linear_temporal_attention=linear_attention,
        time_length=time_length,
        attention_backend=backend,
        compile_dit=False,
    ).cuda()
    model.eval()

    x = 12 * 4**model._level_in
    rank = dist.get_rank(group)

    torch.manual_seed(1234)  # identical inputs on every rank
    full_input = torch.randn(b, c, time_length, x).cuda()
    # Non-trivial, frame-dependent calendar inputs to exercise per-frame encode.
    sod = torch.linspace(0, 1, time_length).view(1, time_length).expand(b, -1).cuda()
    doy = (
        torch.linspace(0.3, 0.7, time_length).view(1, time_length).expand(b, -1).cuda()
    )
    noise_labels = torch.zeros(b).cuda()
    class_labels = torch.empty([b, 0], device=full_input.device)

    def run(net, hs, second, day):
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=dtype, enabled=dtype != torch.float32
            ),
        ):
            return net(
                hs,
                noise_labels=noise_labels,
                class_labels=class_labels,
                second_of_day=second,
                day_of_year=day,
                timestamp=torch.zeros([b, hs.shape[2]]).cuda(),
                is_causal=is_causal,
            ).out

    # Reference: full unsharded forward (no parallel group).
    out_full = run(model, full_input, sod, doy)

    # Time-parallel forward on this rank's frame shard, then gather.
    model.set_time_parallel_group(group)
    sl = slice(rank * t_local, (rank + 1) * t_local)
    out_shard = run(model, full_input[:, :, sl], sod[:, sl], doy[:, sl])

    gathered = [torch.empty_like(out_shard) for _ in range(group_size)]
    dist.all_gather(gathered, out_shard.contiguous(), group=group)
    out_tp = torch.cat(gathered, dim=2)

    out_full = out_full.float()
    out_tp = out_tp.float()
    abs_diff = (out_full - out_tp).abs()
    rel = abs_diff.max() / out_full.abs().max()
    if rank == 0:
        print(
            f"\n[tp-invariance backend={backend} dtype={dtype} causal={is_causal}] "
            f"max_abs={abs_diff.max():.3e} max_rel={rel:.3e} "
            f"mean_abs={abs_diff.mean():.3e}"
        )
    if dtype == torch.float32:
        torch.testing.assert_close(out_tp, out_full, rtol=1e-4, atol=1e-4)
    else:
        assert (
            rel < 0.1
        ), f"bf16 tp divergence too large ({rel:.3e}); suspect a logic bug"


@requires_multi_gpu
def test_dit_with_observations_domain_parallel():
    """Test DiT model with unified observations under context parallelism."""
    if not torch.distributed.is_initialized():
        init()

    spatial_parallelism = 2
    time_parallelism = 2
    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh(
        "cuda",
        [
            world_size // spatial_parallelism // time_parallelism,
            spatial_parallelism,
            time_parallelism,
        ],
    )
    level_model = 4  # HPX16
    sp_rank = mesh.get_local_rank(2)

    b, c, t = 1, 3, 2 // time_parallelism

    # Create sensor config
    sensor_config = {
        "sensor_1": ModelSensorConfig(
            sensor_id=1,
            nchannel=8,
            platform_ids=tuple(range(10)),
        ),
    }

    sensor_embedder_config = SensorEmbedderConfig(
        embed_dim=16,
    )
    meta_dim = sensor_embedder_config.meta_dim

    # Create unified observations
    if sp_rank == 0:
        obs = create_unified_observation(
            nobs=1000,
            batch_size=b,
            time_steps=t,
            meta_dim=meta_dim,
            hpx_level=6,
            n_embed=1024,
            device="cuda",
            sensor_config=sensor_config,
        )
    else:
        obs = UnifiedObservation.empty(device="cuda", hpx_level=6)

    model = dit.DiT(
        embed_v2=True,
        embed_v2_meta_dim=meta_dim,
        embed_v2_n_embed=1024,
        obs_hpx_level=obs.hpx_level,
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        num_attention_heads=1,
        in_channels=c,
        out_channels=c,
        num_layers=2,
        level_model=level_model,
        level_in=6,
        temporal_attention=True,
        time_length=t * time_parallelism,
        compile_dit=False,
    )

    model.set_time_parallel_group(mesh.get_group(1))
    model.set_domain_parallel_group(mesh.get_group(2))

    x = 12 * 4**model._level_in // spatial_parallelism

    model.cuda()

    if sp_rank == 0:
        tensor = torch.arange(b * c * t * x).view(b, c, t, x).cuda().float()
    else:
        tensor = torch.zeros(b, c, t, x).cuda().float()

    subdomain = dit.my_subdomain(mesh.get_group(2), model._level_in, "cuda", b)

    noise_labels = torch.ones(b).cuda()
    class_labels = torch.empty([b, 0], device=tensor.device)

    # Transformer engine does not have support for fp32 attention on my desktop
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        out = model(
            tensor,
            noise_labels=noise_labels,
            class_labels=class_labels,
            second_of_day=torch.zeros([b, t]).cuda(),
            day_of_year=torch.zeros([b, t]).cuda(),
            timestamp=torch.zeros(1).cuda(),
            unified_obs=obs,
            subdomain=subdomain,
        )

    assert out.out.shape == tensor.shape


@requires_multi_gpu
def test_fsdp_global_grad_norm_matches_unsharded():
    """_global_scalar(get_total_norm(grads)) under fully_shard must match the
    unsharded grad-norm: fully_shard makes grads DTensors, so full_tensor() has
    to all-reduce the sharded norm to the true global value before float().
    """
    import math

    from torch.distributed.fsdp import fully_shard

    from healda.training.loop import _global_scalar

    if not torch.distributed.is_initialized():
        init()

    def make_model():
        torch.manual_seed(1234)  # identical weights on every rank
        return torch.nn.Sequential(
            torch.nn.Linear(64, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 64),
        ).cuda()

    torch.manual_seed(4321)  # identical input on every rank
    x = torch.randn(8, 64, device="cuda")

    # Reference: unsharded grad-norm (plain tensors).
    ref = make_model()
    ref(x).pow(2).mean().backward()
    ref_grads = [p.grad for p in ref.parameters() if p.grad is not None]
    ref_norm = float(torch.nn.utils.get_total_norm(ref_grads))

    # fully_shard -> params/grads become DTensors.
    model = make_model()
    fully_shard(model)
    model(x).pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(
        hasattr(g, "full_tensor") for g in grads
    ), "expected DTensor grads under fully_shard; FSDP path not exercised"

    norm_value = _global_scalar(torch.nn.utils.get_total_norm(grads))
    assert math.isfinite(norm_value)
    torch.testing.assert_close(norm_value, ref_norm, rtol=1e-3, atol=1e-3)
