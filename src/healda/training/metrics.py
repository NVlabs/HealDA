# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import time

import torch
import torch.utils.tensorboard
import torchmetrics

from healda.utils import distributed as dist
from healda import training_stats

try:
    import wandb
except ImportError:
    wandb = None


class _StepMetricBuffer:
    """Buffer per-step sums and counts until the publish boundary."""

    def __init__(self):
        self._sum = {}
        self._count = {}
        self._pending_sum = {}
        self._pending_count = {}
        self._steps = []

    def update(self, key, value):
        if not torch.is_tensor(value):
            raise TypeError("step metrics must be tensors")
        value = value.detach().float()
        n = value.numel()
        total = value.sum() if n != 1 else value.reshape(())
        prev = self._sum.get(key)
        self._sum[key] = total if prev is None else prev + total
        self._count[key] = self._count.get(key, 0) + n

    def end_step(self, cur_nimg):
        """Close one step without host traffic."""
        if not self._sum:
            return
        for key, total in self._sum.items():
            self._pending_sum.setdefault(key, []).append(total)
            self._pending_count.setdefault(key, []).append(self._count[key])
        self._sum.clear()
        self._count.clear()
        self._steps.append(cur_nimg)

    @property
    def n_pending(self):
        return len(self._steps)

    def drain(self):
        """Return globally reduced per-step means as ``(keys, values, steps)``."""
        steps = self._steps
        n = len(steps)
        keys = sorted(self._pending_sum)
        distributed = torch.distributed.is_initialized() and dist.get_world_size() > 1
        if distributed:
            # Every rank must reach the all_reduce below, even with nothing
            # buffered; an early return hangs its peers. Agreeing on (keys, n) first
            # turns a diverged key set into an error instead of a hang.
            world = dist.get_world_size()
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, (keys, n))
            if all(g[1] == 0 for g in gathered):
                return None
            if any(g != (keys, n) for g in gathered):
                raise RuntimeError(
                    "step metrics diverged across ranks: rank "
                    f"{dist.get_rank()} has {len(keys)} keys over {n} steps, ranks "
                    f"report {[(len(g[0]), g[1]) for g in gathered]}. Every rank must "
                    "log the same step metrics the same number of times."
                )
        elif not steps:
            return None
        # end_step records a step only when at least one metric is pending.
        device = next(iter(self._pending_sum.values()))[0].device
        total_rows = []
        count_rows = []
        for key in keys:
            vals = self._pending_sum[key]
            if len(vals) != n:
                raise RuntimeError(
                    f"step metric {key!r} has {len(vals)} values for {n} steps"
                )
            total_rows.append(torch.stack(vals).float())
            count_rows.append(self._pending_count[key])
        totals = torch.stack(total_rows)
        counts = torch.tensor(count_rows, dtype=torch.float32, device=device)
        reduced = torch.stack((totals, counts))
        if distributed:
            torch.distributed.all_reduce(reduced)
        stacked = reduced[0] / reduced[1].clamp_min(1)
        self._pending_sum = {}
        self._pending_count = {}
        self._steps = []
        return keys, stacked, steps


class VectorMeanMetric(torchmetrics.Metric):
    """Metric that computes mean across updates while preserving vector structure.

    Unlike MeanMetric which reduces to a scalar, this preserves the shape of input vectors
    (e.g., for logging per-channel metrics without tiny kernel launches).
    """

    def __init__(self, n):
        super().__init__()
        self.add_state(
            "total", default=torch.zeros(1, n).double(), dist_reduce_fx="sum"
        )
        self.add_state("n_calls", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor, enabled: bool = True) -> None:
        """Update metric with a new vector of values."""
        assert values.ndim == 1
        if enabled and values.numel() > 0:
            self.total += values.detach().double().unsqueeze(0)
            self.n_calls += 1

    def compute(self) -> torch.Tensor:
        """Compute the mean across all updates."""
        return self.total.squeeze(0) / self.n_calls.clamp_min(1.0)


class Logger:
    def __init__(self, run_dir, do_wandb=False):
        self._multi_metrics = {}
        self._metrics_to_print = set()
        self.writer = (
            torch.utils.tensorboard.SummaryWriter(run_dir)
            if dist.get_rank() == 0
            else None
        )
        self.run_dir = run_dir
        self.do_wandb = do_wandb
        self._step_buffer = _StepMetricBuffer()
        self._closed = False

    def log_metric(self, key, value, should_print=True, frequency="tick"):
        """Log a metric. Will be averaged over all calls within a tick

        Args:
            should_print: if True then print the metric to the console at the end of the tick

        """
        if frequency == "tick":
            training_stats.report(key, value)
        elif frequency == "step":
            self._step_buffer.update(key, value)

        if should_print:
            self._metrics_to_print.add(key)

    def log_multiple_metrics(
        self,
        keys: list[str],
        values: torch.Tensor,
        should_print=True,
        disable=False,
    ):
        """Log multiple metrics. Used to avoid tiny kernel launches e.g. logging rmse for every channel"""
        key = tuple(keys)
        if key not in self._multi_metrics:
            self._multi_metrics[key] = VectorMeanMetric(len(key)).to(values.device)

        metric = self._multi_metrics[key]
        metric.update(values, enabled=not disable)
        if should_print:
            self._metrics_to_print.update(keys)

    def _get_multi_metrics(self):
        out = {}
        for keys, metric in self._multi_metrics.items():
            values = metric.compute().cpu().numpy().tolist()
            for key, mean in zip(keys, values, strict=True):
                out[key] = {"mean": mean}
        return out

    def flush_step_metrics(self, cur_nimg):
        self._step_buffer.end_step(cur_nimg)

    def publish_step_metrics(self):
        """Reduce and write every buffered step and key."""
        drained = self._step_buffer.drain()
        if drained is None:
            return
        keys, stacked, steps = drained
        if self.writer is None:
            return
        values = stacked.cpu().tolist()
        for key, row in zip(keys, values, strict=True):
            for step, value in zip(steps, row, strict=True):
                self.writer.add_scalar(key, value, global_step=step)

    def close(self):
        if self._closed:
            return
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self._closed = True

    def _reset_multi_metrics(self):
        for metric in self._multi_metrics.values():
            metric.reset()

    def flush_training_stats(self, cur_nimg=None):
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Begin flushing training stats.")
        # Collective: every rank must reach it, so it goes before the rank-0 work.
        self.publish_step_metrics()
        training_stats.default_collector.update()
        info = training_stats.default_collector.as_dict()
        info.update(self._get_multi_metrics())
        nimg = cur_nimg
        if dist.get_rank() == 0:
            try:
                nimg = info["Progress/kimg"]["mean"] * 1000
            except KeyError:
                nimg = cur_nimg

            for k, v in info.items():
                for moment in v:
                    self.writer.add_scalar(f"{k}/{moment}", v[moment], global_step=nimg)

            stats_path = os.path.join(self.run_dir, "stats.jsonl")
            with open(stats_path, "at") as f:
                stats = info
                for stat in stats:
                    mean = stats[stat]["mean"]
                    if stat in self._metrics_to_print:
                        print(f"{stat} = {mean:4g}")
                f.write(json.dumps(dict(info, timestamp=time.time())) + "\n")

            if self.do_wandb:
                metrics = {name: info[name]["mean"] for name in info}
                wandb.log(metrics, step=cur_nimg)

        self._reset_multi_metrics()
