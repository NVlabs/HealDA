# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from healda.training import loop
import torch
from healda.datasets.base import BatchInfo, TimeUnit


class MockLoop(loop.TrainingLoopBase):
    def get_data_loaders(self, batch_gpu: int):
        class MockDataset:
            def __init__(self):
                self.batch_info = BatchInfo(
                    channels=["channel_0", "channel_1"],
                    time_step=1,
                    time_unit=TimeUnit.HOUR,
                    scales=[1.0, 1.0],
                    center=[0.0, 0.0],
                )

        dataset = MockDataset()
        return dataset, None, None

    def get_network(self) -> torch.nn.Module:
        return torch.nn.Linear(10, 10)

    def get_optimizer(self, parameters):
        return torch.optim.Adam(parameters)

    def get_loss_fn(self):
        return torch.nn.MSELoss()


def test_loop_save_load(tmp_path):
    rundir = tmp_path
    loop = MockLoop(rundir.as_posix(), batch_gpu=1)
    loop.setup()
    loop.save_training_state(0)

    loop = MockLoop.loads((rundir / "loop.json").read_text())
    loop.setup()
    loop.resume_from_rundir(rundir)


class StepLoop(MockLoop):
    def _batch_iterator(self):
        while True:
            yield None

    def backward_batch(self, dataset_iterator):
        next(dataset_iterator)

    def step_optimizer(self, cur_nimg):
        pass

    def validate(self, net):
        pass

    def log_tick(self, *args):
        pass

    def flush_training_stats(self):
        pass


def test_loop_stops_at_stop_nimg(tmp_path):
    loop = StepLoop(
        tmp_path.as_posix(),
        batch_gpu=1,
        batch_size=1,
        total_ticks=10,
        steps_per_tick=5,
        state_dump_ticks=None,
        save_final_state=False,
        stop_nimg=7,
    )
    loop.setup()
    loop._train()
    assert loop.cur_nimg == 7
    assert (tmp_path / "training-state-000000007.checkpoint").exists()

    # A launch that resumes at stop_nimg exits without training.
    loop._train()
    assert loop.cur_nimg == 7
