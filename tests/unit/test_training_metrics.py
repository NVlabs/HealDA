# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import warnings

import pytest
import torch

from healda.training import metrics as metrics_module
from healda.training.metrics import Logger, VectorMeanMetric


class TestVectorMeanMetric:
    def test_single_update(self):
        metric = VectorMeanMetric(3)
        values = torch.tensor([1.0, 2.0, 3.0])
        metric.update(values)
        result = metric.compute()
        assert result.shape == (3,)
        torch.testing.assert_close(result.float(), values)

    def test_mean_across_updates(self):
        metric = VectorMeanMetric(2)
        metric.update(torch.tensor([1.0, 2.0]))
        metric.update(torch.tensor([3.0, 4.0]))
        result = metric.compute()
        torch.testing.assert_close(result.float(), torch.tensor([2.0, 3.0]))

    def test_reset(self):
        metric = VectorMeanMetric(2)
        metric.update(torch.tensor([1.0, 2.0]))
        metric.reset()
        metric.update(torch.tensor([5.0, 6.0]))
        result = metric.compute()
        torch.testing.assert_close(result.float(), torch.tensor([5.0, 6.0]))

    def test_preserves_vector_shape(self):
        n = 10
        metric = VectorMeanMetric(n)
        metric.update(torch.ones(n))
        result = metric.compute()
        assert result.shape == (n,)

    def test_rejects_non_1d_input(self):
        metric = VectorMeanMetric(3)
        with pytest.raises(AssertionError):
            metric.update(torch.ones(2, 3))


class TestLogger:
    def test_log_metric_tick_adds_to_print_set(self, tmp_path):
        logger = Logger(str(tmp_path))
        logger.log_metric(
            "Loss/loss", torch.tensor(0.5), should_print=True, frequency="tick"
        )
        assert "Loss/loss" in logger._metrics_to_print

    def test_log_metric_step_accumulates(self, tmp_path):
        logger = Logger(str(tmp_path))
        logger.log_metric(
            "lr", torch.tensor(0.001), should_print=False, frequency="step"
        )
        logger.log_metric(
            "lr", torch.tensor(0.002), should_print=False, frequency="step"
        )
        logger.flush_step_metrics(cur_nimg=1)
        keys, stacked, steps = logger._step_buffer.drain()
        assert keys == ["lr"] and steps == [1]
        torch.testing.assert_close(stacked[0, 0], torch.tensor(0.0015))

    def test_step_metrics_require_tensors(self, tmp_path):
        logger = Logger(str(tmp_path))
        with pytest.raises(TypeError, match="step metrics must be tensors"):
            logger.log_metric("lr", 0.001, frequency="step")

    def test_log_multiple_metrics(self, tmp_path):
        logger = Logger(str(tmp_path))
        keys = ["ch0", "ch1", "ch2"]
        values = torch.tensor([1.0, 2.0, 3.0])
        logger.log_multiple_metrics(keys, values)
        assert set(keys).issubset(logger._metrics_to_print)
        key_tuple = tuple(keys)
        assert key_tuple in logger._multi_metrics

    def test_log_multiple_metrics_disable_skips_update(self, tmp_path):
        logger = Logger(str(tmp_path))
        keys = ["a", "b"]
        values = torch.tensor([10.0, 20.0])
        logger.log_multiple_metrics(keys, values, disable=True)
        # metric is created but never updated, so n_calls == 0
        metric = logger._multi_metrics[tuple(keys)]
        assert metric.n_calls == 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metric.compute()
        assert not caught

    def test_flush_step_metrics_defers_the_write_to_publish(self, tmp_path):
        logger = Logger(str(tmp_path))
        written = []
        logger.writer.add_scalar = lambda k, v, global_step: written.append(
            (k, v, global_step)
        )

        for i, value in enumerate([0.01, 0.03]):
            logger.log_metric(
                "lr", torch.tensor(value), should_print=False, frequency="step"
            )
            logger.flush_step_metrics(cur_nimg=1000 + i)
        # Nothing has crossed to the host yet: a step costs no D2H.
        assert written == []
        assert logger._step_buffer.n_pending == 2

        logger.publish_step_metrics()
        logger.close()
        assert logger._step_buffer.n_pending == 0
        assert [(k, s) for k, _, s in written] == [("lr", 1000), ("lr", 1001)]
        assert [round(v, 4) for _, v, _ in written] == [0.01, 0.03]

    def test_step_metrics_match_mean_metric(self, tmp_path):
        """The device-resident buffer must reproduce MeanMetric's semantics."""
        import torchmetrics

        logger = Logger(str(tmp_path))
        reference = torchmetrics.MeanMetric()
        expected = []
        # Ragged shapes on purpose: equal-sized updates cannot distinguish
        # element-weighted from update-weighted averaging, which is the way the
        # buffer previously diverged from MeanMetric.
        shapes = [
            [(2, 3, 1, 5), (2, 3, 1, 5)],
            [(2, 3, 1, 5), (1, 3, 1, 5)],
            [(4, 3, 2, 5), (1, 3, 1, 1)],
        ]
        for step in range(3):
            for value in (torch.randn(*s) for s in shapes[step]):
                logger.log_metric("loss", value, should_print=False, frequency="step")
                reference.update(value)
            expected.append(reference.compute().item())
            reference.reset()
            logger.flush_step_metrics(cur_nimg=step)

        _, stacked, _ = logger._step_buffer.drain()
        torch.testing.assert_close(
            stacked[0], torch.tensor(expected), rtol=1e-5, atol=1e-6
        )

    def test_step_metrics_reduce_global_sum_and_count(self, tmp_path, monkeypatch):
        logger = Logger(str(tmp_path))
        logger.log_metric(
            "loss", torch.tensor([1.0, 1.0]), should_print=False, frequency="step"
        )
        logger.flush_step_metrics(cur_nimg=1)

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(metrics_module.dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(metrics_module.dist, "get_rank", lambda: 0)

        def gather_metadata(gathered, value):
            gathered[:] = [value, value]

        def add_remote_rank(reduced):
            reduced[0, 0, 0] += 9.0
            reduced[1, 0, 0] += 1.0

        monkeypatch.setattr(torch.distributed, "all_gather_object", gather_metadata)
        monkeypatch.setattr(torch.distributed, "all_reduce", add_remote_rank)

        _, stacked, _ = logger._step_buffer.drain()
        torch.testing.assert_close(stacked[0, 0], torch.tensor(11.0 / 3.0))

    def test_flush_training_stats_writes_jsonl(self, tmp_path):
        logger = Logger(str(tmp_path))
        logger.flush_training_stats(cur_nimg=0)
        stats_path = tmp_path / "stats.jsonl"
        assert stats_path.exists()
        line = stats_path.read_text().strip()
        if line:
            data = json.loads(line)
            assert "timestamp" in data
