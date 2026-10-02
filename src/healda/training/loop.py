# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import dataclasses
from typing import Optional
import json
import math
import gc
import os
import time
import itertools
import shutil
import warnings
import glob
from functools import partial

from typing import Iterable, Union
import re

import healda.config.models
import healda.models
import healda.training.checkpoint
import healda.training.distributed_checkpoint
import healda.utils.profiling
import numpy as np
import psutil
import torch
import torch.utils.tensorboard
from healda.utils import distributed as dist
import signal
from healda.utils.signals import finish_before_quitting, QuitEarly, handler
from healda import training_stats
from healda.training import utils as misc
from healda.training.metrics import Logger
from healda.datasets.base import SpatioTemporalDataset, BatchInfo
import logging

try:
    import wandb
except ImportError:
    wandb = None

DATASET_METADATA_FILENAME = "dataset-metadata.pth"
TRAINER_METADATA_FILENAME = "loop.json"


logger = logging.getLogger(__name__)


def _global_tensor(t):
    """Materialize a global scalar, staying on device. No host sync."""
    # DTensor.full_tensor() all-reduces FSDP shard-local scalar norms.
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    if torch.is_tensor(t) and t.numel() != 1:
        raise ValueError(f"expected scalar tensor, got shape {tuple(t.shape)}")
    return t


def _global_scalar(t):
    return float(_global_tensor(t))


def _to_batch(x, device, non_blocking=True):
    if isinstance(x, dict):
        return {
            k: _to_batch(v, device, non_blocking=non_blocking) for k, v in x.items()
        }
    elif isinstance(x, list):
        return [_to_batch(i, device, non_blocking=non_blocking) for i in x]
    elif torch.is_tensor(x):
        if torch.is_floating_point(x):
            x = x.float()
        return x.to(device, non_blocking=non_blocking)
    elif hasattr(x, "to") and callable(getattr(x, "to")):
        # custom object with a 'to' method
        return x.to(device, non_blocking=non_blocking)
    elif dataclasses.is_dataclass(x):
        return x.__class__(
            **{
                field.name: _to_batch(
                    getattr(x, field.name), device, non_blocking=non_blocking
                )
                for field in dataclasses.fields(x)
            }
        )
    else:
        raise NotImplementedError(x)


def _format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(
            s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60
        )


class CheckpointHandler:
    def __init__(self, run_dir, filename: str = "training-state-{}.checkpoint"):
        self.filename = filename
        self.run_dir = run_dir

    def get_filename(self, nimg):
        return self.filename.format("%09d" % nimg)

    def get_path(self, nimg):
        return os.path.join(self.run_dir, self.get_filename(nimg))

    def nimg_from_path(self, path) -> int | None:
        m = re.match(self.filename.format(r"(\d{9})"), os.path.basename(path))
        return int(m.group(1)) if m else None

    def list_checkpoints(self, run_dir=None):
        run_dir = run_dir or self.run_dir
        files = glob.glob(self.filename.format("*"), root_dir=run_dir)
        pattern = self.filename.format(r"(\d{9})")
        files = sorted(files)
        for file in files:
            m = re.match(pattern, file)
            if m:
                nimg = int(m.group(1))
                yield os.path.join(run_dir, file), nimg


@dataclasses.dataclass
class TrainingLoopBase(abc.ABC):
    """Abstract base class for diffusion trainings loops

    Implementations should define
    - get_data_loaders
    - get_network

    """

    run_dir: str = "."  # Output directory.
    seed: int = 0  # Global random seed.
    # Seed from the data rank, so time-parallel ranks sharding one sample share a
    # numpy stream. Required by obs dropout_scope='sample'.
    fix_time_parallel_rng: bool = False
    batch_size: int = 512  # Total batch size for one training iteration.
    batch_gpu: Optional[int] = None  # Limit batch size per GPU, None = no limit.
    enable_ema: bool = False
    ema_halflife_kimg: int = (
        500  # Half-life of the exponential moving average (EMA) of model weights.
    )
    ema_rampup_ratio: float = 0.05  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_img: int = 10_000  # Learning rate ramp-up duration.
    flat_imgs: int = 1_500_000 - 10_000
    decay_imgs: int = 1_500_000
    lr_min: float = 1e-6
    lr: float = 1e-4

    gradient_clip_max_norm: Optional[float] = None
    total_ticks: int = 10
    print_steps: int = 50
    steps_per_tick: int = 1024
    snapshot_ticks: int | None = (
        50  # How often to save network snapshots, None = disable.
    )
    state_dump_ticks: int | None = (
        500  # How often to dump training state, None = disable.
    )
    save_final_state: bool = True

    test_with_single_batch: bool = False
    """Only load a single batch of data for testing and profiling purposes"""

    setup_datasets: bool = True
    """Build the train/valid loaders in setup(). Inference builds its own and skips these."""

    # Performance optimizations
    # Mixed precision and performance options
    cudnn_benchmark: bool = True  # Enable torch.backends.cudnn.benchmark?
    tf32: bool = True
    bf16: bool = True
    compile_optimizer: bool = False  # if true wrap the optimizer with torch compile

    # wandb
    wandb_id: str | None = None  # will be read from checkpoint if not provided
    wandb_enabled: bool = True
    wandb_project: str = "healda"
    wandb_entity: str | None = None

    # logging
    log_parameter_norm: bool = False
    log_parameter_grad_norm: bool = False
    description: str = ""

    device: torch.device | None = None

    def __post_init__(self):
        if self.steps_per_tick <= 0:
            raise ValueError(
                f"steps_per_tick must be positive, got {self.steps_per_tick}"
            )

        self.ema: torch.nn.Module | None = None
        self.iteration = 0
        self.do_wandb = False
        self._wandb_run = None

    @abc.abstractmethod
    def get_data_loaders(
        self, batch_gpu: int
    ) -> tuple[SpatioTemporalDataset, Iterable, Iterable]:
        pass

    def get_network(self) -> torch.nn.Module:
        return healda.models.get_model(self.model_config)

    @abc.abstractmethod
    def get_optimizer(self, parameters):
        pass

    @abc.abstractmethod
    def get_loss_fn(self):
        pass

    @property
    def model_config(self) -> healda.config.models.ModelConfigV1 | None:
        """Model configuration used for the network. This is used for checkpointing.

        If you are overriding get_network, then be sure to make this consistent.
        """
        return None

    def _setup_datasets(self):
        if not self.setup_datasets:
            self.dataset_obj = self.train_loader = self.valid_loader = None
            return
        self.dataset_obj, self.train_loader, self.valid_loader = self.get_data_loaders(
            self.batch_gpu
        )
        if self.test_with_single_batch:
            self.train_loader = itertools.repeat(next(iter(self.train_loader)))
            self.valid_loader = [next(iter(self.valid_loader))]

    def _setup_networks(self):
        self.ddp = self.net = self.get_network()
        self.net.train().requires_grad_(True).to(self.device)
        if dist.get_world_size() > 1:
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.device],
                broadcast_buffers=False,
            )

    @healda.utils.profiling.nvtx
    def log_tick(
        self,
        maintenance_time,
        tick_start_time,
        tick_end_time,
        start_time,
        cur_tick,
        cur_nimg,
    ):
        # Print status line, accumulating the same information in training_stats.
        images_per_tick = self.steps_per_tick * self.batch_size
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {_format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / images_per_tick * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(self.device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(self.device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

    def setup_logs(self):
        if dist.get_rank() != 0:
            logger.setLevel(logging.CRITICAL)

        self.logger = Logger(self.run_dir, do_wandb=self.do_wandb)
        # back-compat: some callsites write scalars directly to the writer
        self.writer = self.logger.writer

    @property
    def batch_gpu_total(self) -> int:
        world_size: int = dist.get_world_size()
        return self.batch_size // world_size

    def setup_batching(self):
        # Select batch size per GPU.
        if self.batch_gpu is None or self.batch_gpu > self.batch_gpu_total:
            self.batch_gpu = self.batch_gpu_total

        if self.batch_gpu_total % self.batch_gpu != 0:
            raise ValueError()

        num_accumulation_rounds = self.batch_gpu_total // self.batch_gpu
        self.num_accumulation_rounds = num_accumulation_rounds

    @staticmethod
    def print_network_info(net, device):
        pass

    def _load_iterator_state(self, checkpoint):
        with checkpoint.open("iterator_state.json") as f:
            iterator_state = json.loads(f.read())
            if iterator_state:
                self.epoch_idx = iterator_state["epoch_idx"]
                self.samples_processed_this_epoch_per_rank = iterator_state[
                    "samples_processed_this_epoch_per_rank"
                ]

    def resume_from_state(
        self,
        resume_state_dump,
        optimizer=True,
        require_all=True,
        wandb=False,
        iterator_state=True,
    ):
        dist.print0(f'Loading training state from "{resume_state_dump}"...')

        with healda.training.checkpoint.Checkpoint(
            resume_state_dump, "r"
        ) as checkpoint:
            self._load_net_state(checkpoint, require_all)
            gc.collect()
            if optimizer and self.optimizer is not None:
                self._load_optimizer_state(checkpoint)

            loop_json = checkpoint.read_loop_json()
            if loop_json is None:
                raise ValueError(f"{resume_state_dump} has no loop.json")
            old_loop = self.loads(loop_json)

            # Restore iterator state if available (for backward compatibility)
            if iterator_state:
                try:
                    self._load_iterator_state(checkpoint)
                except FileNotFoundError as e:
                    logger.warning(
                        f"Iterator state not found in checkpoint (backward compatibility): {e}. "
                        "Using defaults (epoch_idx=0, samples_processed_this_epoch_per_rank=0)"
                    )

            # handle wandb
            if wandb:
                self.wandb_id = old_loop.wandb_id

    def _load_net_state(self, checkpoint, require_all):
        with checkpoint.open("net_state.pth", "r") as f:
            net_state = torch.load(f, weights_only=True, map_location="cpu")
            self.net.load_state_dict(net_state, strict=require_all)

    def _load_optimizer_state(self, checkpoint):
        with checkpoint.open("optimizer_state.pth", "r") as f:
            # load to cpu to avoid copies in gpu memory
            optimizer_state = torch.load(f, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)

    def train_step(
        self, *, condition=None, target, labels, augment_labels=None, **kwargs
    ):
        return self.loss_fn(
            net=partial(
                self.ddp,
                condition=condition,
                class_labels=labels,
                augment_labels=augment_labels,
                **kwargs,
            ),
            images=target,
        )

    def _stage_tuple_batch(self, batch):
        indict = {}
        images, labels, condition = batch[:3]
        assert images.ndim == 4
        indict["target"] = images.to(self.device)
        indict["condition"] = condition.to(self.device)
        indict["labels"] = labels.to(self.device)

        if len(batch) == 4:
            augment_labels = batch[3]
            if augment_labels is not None:
                augment_labels.to(self.device).float()
            indict["augment_labels"] = batch[3]
        return indict

    def _stage_dict_batch(self, batch):
        return _to_batch(batch, self.device)

    @healda.utils.profiling.nvtx
    def backward_batch(self, dataset_iterator):
        self.ddp.train()
        with healda.utils.profiling.nvtx_range("step.zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        time_start = time.time()
        for round_idx in range(self.num_accumulation_rounds):
            with misc.ddp_sync(
                self.ddp, (round_idx == self.num_accumulation_rounds - 1)
            ):
                with healda.utils.profiling.nvtx_range("load data"):
                    with healda.utils.profiling.cpu_timing_range(
                        "step.dataload", group="step"
                    ):
                        batch = next(dataset_iterator)

                    if isinstance(batch, dict):
                        with healda.utils.profiling.cpu_timing_range(
                            "step.stage", group="step"
                        ):
                            indict = self._stage_dict_batch(batch)
                    else:
                        warnings.warn(
                            DeprecationWarning(
                                "tuple based dataloaders will be removed soon. please refactor to use dicts."
                            )
                        )
                        indict = self._stage_tuple_batch(batch)

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
                ):
                    with healda.utils.profiling.nvtx_range("step.train_step"):
                        loss = self.train_step(**indict)
                    self.log_metric("Loss/loss", loss, should_print=True)

                    # run_model_step already returns a 0-dim loss, so nothing is left to
                    # reduce here.
                    loss_mean = loss / self.num_accumulation_rounds

                with healda.utils.profiling.nvtx_range("training_loop:backward"):
                    loss_mean.backward()

                total_loss += loss_mean.detach()
        time_end = time.time()
        if self.should_log_debug:
            self.log_debug(f"Final Loss: {total_loss.item()}")
            # time_end precedes .item(), so this reports CPU enqueue time only.
            self.log_debug(
                f"Enqueue time for {self.num_accumulation_rounds} accumulation "
                f"rounds (CPU, pre-sync): {time_end - time_start}"
            )

    def _log_parameter_and_gradient_norms(self):
        # Optional, off by default. The global grad_norm is computed once in
        # step_optimizer and reused for clipping (see below), so it is not
        # recomputed here.
        if not (self.log_parameter_norm or self.log_parameter_grad_norm):
            return
        for name, param in self.net.named_parameters():
            if self.log_parameter_norm:
                self.log_metric(
                    f"param_norm/{name}",
                    param.data.norm(2),
                    frequency="tick",
                    should_print=False,
                )
            if self.log_parameter_grad_norm and param.grad is not None:
                self.log_metric(f"grad_norm/{name}", param.grad.norm(2))

    @healda.utils.profiling.nvtx
    @finish_before_quitting
    def step_optimizer(self, cur_nimg):
        with healda.utils.profiling.nvtx_range("training_loop:step"):
            warmup_imgs = self.lr_rampup_img
            flat_imgs = self.flat_imgs
            decay_imgs = self.decay_imgs
            total_imgs = warmup_imgs + flat_imgs + decay_imgs

            def lr_lambda(cur_nimg):
                import math

                base_lr = self.lr
                min_lr = self.lr_min

                min_factor = min_lr / base_lr
                if cur_nimg < warmup_imgs:
                    # linear ramp from 0 → 1
                    return float(cur_nimg) / warmup_imgs
                elif cur_nimg < warmup_imgs + flat_imgs:
                    return 1.0
                elif cur_nimg < total_imgs:
                    # cosine decay from factor=1 → factor=min_factor
                    progress = float(cur_nimg - warmup_imgs - flat_imgs) / decay_imgs
                    # standard cosine schedule:
                    return min_factor + 0.5 * (1.0 - min_factor) * (
                        1.0 + math.cos(math.pi * progress)
                    )
                else:
                    return min_factor

            def default_scale(cur_nimg):
                return min(cur_nimg / max(self.lr_rampup_img, 1e-8), 1)

            use_lr_lambda = True
            scale_fn = lr_lambda if use_lr_lambda else default_scale

            scale = scale_fn(self.cur_nimg)
            self._apply_lr(scale_fn, scale, default_scale)
            self._log_parameter_and_gradient_norms()

            with healda.utils.profiling.nvtx_range("step.gradnorm_clip"):
                # Global grad norm, computed once and reused for logging and
                # clipping. See _clip_grads for the policy note.
                reported_norm = self._clip_grads()

            # Logged as a device tensor: both sinks accumulate device-side, so this
            # adds no host sync of its own. _clip_grads returns the pre-sanitization
            # norm, so NaN/Inf spikes stay visible.
            self.log_metric(
                "grad_norm", reported_norm, frequency="tick", should_print=True
            )
            self.log_metric("grad_norm", reported_norm, frequency="step")

            with (
                healda.utils.profiling.nvtx_range("step.optimizer"),
                healda.utils.profiling.cpu_timing_range("step.optimizer", group="step"),
            ):
                self._step_optimizer()
            # increment the number of images processed within the current epoch
            self.samples_processed_this_epoch_per_rank += self.batch_gpu or 1

        self._flush_step_metrics()

    def _apply_lr(self, scale_fn, scale, default_scale):
        for g in self.optimizer.param_groups:
            if "base_lr" not in g:
                if "lr" in g:
                    g["base_lr"] = g["lr"]  # lazy init from existing LR
                else:
                    g["base_lr"] = self.optimizer.defaults["lr"]
            lr = g["base_lr"] * scale
            if self.should_log_debug:
                self.log_debug(
                    f"Learning rate: {lr} from base: {g['base_lr']} with scale "
                    f"factor: {scale} (would normally be "
                    f"{default_scale(self.cur_nimg)})"
                )

            g["lr"] = lr
            # Rank-gated: this ran on every rank every step, and only rank 0's
            # writer goes anywhere.
            if dist.get_rank() == 0:
                self.writer.add_scalar("lr", lr, global_step=self.cur_nimg)

    def _clip_grads(self):
        grads = [p.grad for p in self.net.parameters() if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(grads)
        # Reported separately from the clipping norm: sanitizing below rewrites
        # total_norm to a finite value, which would hide the spike from the logs.
        reported_norm = _global_tensor(total_norm)
        # A non-finite norm has to sanitize the gradients, not just the clip
        # coefficient: NaN * 0 is still NaN, and AdamW with zero grads still steps.
        # Hence the host branch, and its sync.
        norm_value = float(reported_norm)
        if not math.isfinite(norm_value):
            torch._foreach_clamp_min_(grads, -1e5)
            torch._foreach_clamp_max_(grads, 1e5)
            for g in grads:
                torch.nan_to_num(g, nan=0.0, out=g)
            total_norm = torch.nn.utils.get_total_norm(grads)
            norm_value = _global_scalar(total_norm)
        if (
            self.gradient_clip_max_norm is not None
            and norm_value > self.gradient_clip_max_norm
        ):
            torch._foreach_mul_(
                grads, self.gradient_clip_max_norm / (norm_value + 1e-6)
            )
        return reported_norm

    def _report_step_timing(self, cur_tick):
        # Every rank prints, so a cost on one rank reads differently from a
        # collective wait on all of them. CPU only; use nsys for kernel attribution.
        if not healda.utils.profiling.CPU_TIMING_ENABLED:
            return
        print(
            f"[tick {cur_tick} rank {dist.get_rank()}] "
            f"{healda.utils.profiling.cpu_timing_report('step')}",
            flush=True,
        )
        healda.utils.profiling.cpu_timing_reset("step")

    def on_tick(self):
        pass

    @healda.utils.profiling.nvtx
    def validate(self, net):
        loss_key = "Loss/test_loss"

        with torch.no_grad():
            for batch in self.valid_loader:
                if len(batch) == 4:
                    images, labels, condition, augment_labels = batch
                else:
                    images, labels, condition = batch
                    augment_labels = None

                assert images.ndim == 4

                images = images.to(self.device).to(torch.float32)
                condition = condition.to(self.device).to(torch.float32)
                labels = labels.to(self.device)

                loss = self.train_step(
                    condition=condition,
                    target=images,
                    labels=labels,
                    augment_labels=augment_labels,
                )
                training_stats.report(loss_key, loss)

    def log_metric(self, key, value, should_print=True, frequency="tick"):
        """Log a metric, averaged over all calls within a tick.

        Args:
            should_print: if True, print the metric to the console at the end of the tick.
        """
        self.logger.log_metric(
            key, value, should_print=should_print, frequency=frequency
        )

    def log_multiple_metrics(self, keys, values, should_print=True, disable=False):
        """Log a vector of metrics in one shot (e.g. per-channel rmse) to avoid tiny kernel launches."""
        self.logger.log_multiple_metrics(
            keys,
            values,
            should_print=should_print,
            disable=disable,
        )

    def _flush_step_metrics(self):
        self.logger.flush_step_metrics(self.cur_nimg)

    @property
    def batch_info(self) -> None | BatchInfo:
        return None

    @healda.utils.profiling.nvtx
    @finish_before_quitting
    def _save_checkpoint(self, path, optimizer: bool):
        # ensure that file updates are atomic to avoid faulty
        # restart files
        tmppath = path + ".tmp" + str(os.getpid())
        with healda.training.checkpoint.Checkpoint(tmppath, "w") as checkpoint:
            checkpoint.write_model(self.net)
            if self.batch_info is not None:
                checkpoint.write_batch_info(self.batch_info)

            if optimizer:
                with checkpoint.open("optimizer_state.pth", "w") as f:
                    torch.save(self.optimizer.state_dict(), f)

            checkpoint.write_loop_json(self.dumps())

            # Save iterator state for resuming
            with checkpoint.open("iterator_state.json", "w") as f:
                iterator_state = {
                    "epoch_idx": self.epoch_idx,
                    "samples_processed_this_epoch_per_rank": self.samples_processed_this_epoch_per_rank,
                }
                f.write(json.dumps(iterator_state).encode())

            if self.model_config is not None:
                checkpoint.write_model_config(self.model_config)
        shutil.move(tmppath, path)

    @healda.utils.profiling.nvtx
    def save_training_state(self, cur_nimg):
        if dist.get_rank() != 0:
            return

        state_filename = self._state_checkpoint_handler.get_path(cur_nimg)
        dist.print0(f"Saving checkpoint to {state_filename}")
        self._save_checkpoint(state_filename, optimizer=True)
        dist.print0(f"Checkpoint saved to {state_filename}")

    @healda.utils.profiling.nvtx
    def save_network_snapshot(self, cur_nimg):
        if dist.get_rank() != 0:
            return

        filename = self._snapshot_checkpoint_handler.get_path(cur_nimg)
        dist.print0(f"Saving network snapshot to {filename}")
        self._save_checkpoint(filename, optimizer=False)

    def flush_training_stats(self):
        # setup_wandb() may flip do_wandb after the logger is created; keep them in sync.
        self.logger.do_wandb = self.do_wandb
        self.logger.flush_training_stats(self.cur_nimg)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.loads(f.read())

    @classmethod
    def from_rundir(cls, run_dir):
        path = os.path.join(run_dir, TRAINER_METADATA_FILENAME)
        loop = cls.from_json(path)
        loop.run_dir = run_dir
        return loop

    def dumps(self):
        fields = dataclasses.asdict(self)
        fields.pop("device", None)
        return json.dumps(fields)

    @classmethod
    def loads(cls, s):
        return cls(**json.loads(s))

    def save_metadata(self):
        with open(os.path.join(self.run_dir, TRAINER_METADATA_FILENAME), "w") as f:
            fields = dataclasses.asdict(self)
            fields.pop("device", None)
            f.write(self.dumps())

    def setup(self):
        # device must be set before setup_logs(): the metrics Logger pins step
        # metrics to it so the distributed all_gather runs on the NCCL group.
        self.device = self.device or torch.device("cuda", torch.cuda.current_device())
        # Every rank writes loop.json, so every rank needs the directory. Do not rely on
        # SummaryWriter creating it: that only happens on rank 0.
        os.makedirs(self.run_dir, exist_ok=True)
        self.setup_logs()
        self.save_metadata()
        self.loss_fn = self.get_loss_fn()

        # iterators
        # used to restore the sampler state during restarts
        self.cur_nimg = 0
        self.epoch_idx = 0
        self.samples_processed_this_epoch_per_rank = 0

        # Resolve batch_gpu (None -> batch_gpu_total) before building loaders, so
        # the DataLoader gets a real batch_size and keeps auto-collation.
        self.setup_batching()
        self._setup_datasets()
        self._setup_networks()
        self.print_network_info(self.net, self.device)
        self._setup_optimizer()
        self._state_checkpoint_handler = CheckpointHandler(self.run_dir)
        self._snapshot_checkpoint_handler = CheckpointHandler(
            self.run_dir, "network-snapshot-{}.checkpoint"
        )

    def _setup_optimizer(self):
        self.optimizer = self.get_optimizer(self.net.named_parameters())
        if self.compile_optimizer:
            self._step_optimizer = torch.compile(self.optimizer.step)
        else:
            self._step_optimizer = self.optimizer.step

    def setup_wandb(self, **kwargs):
        if not self.wandb_enabled:
            return
        try:
            if wandb is not None and dist.get_rank() == 0:
                os.environ["WANDB_API_KEY"]
                run = wandb.init(
                    id=self.wandb_id,
                    config=json.loads(self.dumps()),
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    notes=self.description,
                    **kwargs,
                )
                self.wandb_id = run.id
                self.do_wandb = True
                self._wandb_run = run
        except KeyError:
            # cannot init wandb
            dist.print0("WANDB_API_KEY not set. Cannot use wandb")
            pass

    def _is_checkpoint_valid(self, path):
        if os.path.isdir(path):
            return healda.training.distributed_checkpoint.is_valid(path)
        return healda.training.checkpoint.Checkpoint.is_valid(path)

    def resume_from_rundir(self, run_dir=None, require_all=True):
        checkpoints = list(self._state_checkpoint_handler.list_checkpoints(run_dir))
        if not checkpoints:
            raise FileNotFoundError("No checkpoint file found.")

        for path, nimg in reversed(checkpoints):
            if not self._is_checkpoint_valid(path):
                dist.print0(f"WARNING: Skipping invalid checkpoint: {path}")
                continue
            self.cur_nimg = nimg
            self.resume_from_state(path, require_all=require_all, wandb=True)
            return

        raise FileNotFoundError(
            "No valid checkpoint found — all checkpoints appear incomplete."
        )

    @property
    def should_log_debug(self):
        return dist.get_rank() == 0 and self.iteration % self.print_steps == 0

    def log_debug(self, msg):
        if not self.should_log_debug:
            return

        logger.debug(msg)

    def train(self):
        # signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        try:
            self._train()
        except QuitEarly as e:
            dist.print0(f"Caught {e}. Quitting early.")
            # Save the checkpoint and nothing else: there are only KillWait seconds before
            # SIGKILL, and anything else here (a metrics flush) can eat them and lose it.
            self.save_training_state(self.cur_nimg)
            try:
                del self.train_loader
                del self.valid_loader
            except AttributeError:
                pass
        finally:
            self.logger.close()

    def _batch_iterator(self):
        while True:
            for batch in self.train_loader:
                yield batch

            self.epoch_idx += 1
            self.samples_processed_this_epoch_per_rank = 0
            self.iteration = 0

    @property
    def data_rank(self) -> int:
        """Rank along the data-parallel axis. Subclasses with a device mesh override."""
        return dist.get_rank()

    @property
    def data_world_size(self) -> int:
        return dist.get_world_size()

    def _train(self):
        dist.print0("Loss function", self.loss_fn)
        start_time = time.time()
        # Ranks sharding same sample (time parallel ranks) should have same random state.
        rng_rank = self.data_rank if self.fix_time_parallel_rng else dist.get_rank()
        rng_world = (
            self.data_world_size
            if self.fix_time_parallel_rng
            else dist.get_world_size()
        )
        np.random.seed((self.seed * rng_world + rng_rank + self.cur_nimg) % (1 << 31))
        torch.manual_seed(np.random.randint(1 << 31))
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = self.tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            not self.tf32
        )

        # Train.
        tick_start_time = time.time()
        maintenance_time = tick_start_time - start_time
        dist.update_progress(0, self.total_ticks)
        dataset_iterator = self._batch_iterator()
        top_time = time.time()
        steps = 0
        dist.print0(
            f"Starting training loop: {self.total_ticks} ticks x "
            f"{self.steps_per_tick} steps/tick, batch_size={self.batch_size}, "
            f"cur_nimg={self.cur_nimg}. Loading first batch..."
        )
        for cur_tick in range(self.total_ticks):
            for _ in range(self.steps_per_tick):
                step_start = time.time()
                self.backward_batch(dataset_iterator)
                self.cur_nimg += self.batch_size
                self.step_optimizer(self.cur_nimg)
                with healda.utils.profiling.nvtx_range("step.loop_tail"):
                    step_end = time.time()
                    self.log_debug(
                        f"Step {steps} time: {step_end - step_start}. Avg time: {(step_end - top_time) / (steps + 1)}"
                    )
                    self.log_debug(
                        f"CPU Memory: {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}GB"
                    )
                    steps += 1
                    self.iteration = steps
            tick_end_time = time.time()
            self._report_step_timing(cur_tick)
            self.log_tick(
                maintenance_time,
                tick_start_time,
                tick_end_time,
                start_time,
                cur_tick,
                self.cur_nimg,
            )

            # Save network snapshot.
            if (self.snapshot_ticks is not None) and (
                cur_tick % self.snapshot_ticks == 0
            ):
                self.save_network_snapshot(self.cur_nimg)
            if (self.state_dump_ticks is not None) and (
                cur_tick % self.state_dump_ticks == 0
            ):
                self.save_training_state(self.cur_nimg)

            self.net.eval()
            logger.info("Validating...")
            val_start_time = time.time()
            self.validate(self.net)
            val_time = time.time() - val_start_time
            logger.info(f"Validation time: {val_time:.2f}s.")
            self.net.train()

            # Update logs.
            self.flush_training_stats()
            dist.update_progress(cur_tick, self.total_ticks)

            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

        # Done.
        if self.save_final_state:
            self.save_training_state(self.cur_nimg)
        dist.print0("Exiting...")
