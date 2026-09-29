# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


from healda.training.loop import CheckpointHandler


def test_checkpoint_handler_basic_functionality(tmp_path):
    """Test core CheckpointHandler functionality."""
    handler = CheckpointHandler(str(tmp_path))

    # Test get_path
    assert handler.get_path(123) == f"{tmp_path}/training-state-000000123.checkpoint"

    # Test list_checkpoints with mock files
    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = ["training-state-000000001.checkpoint", "invalid.txt"]
        checkpoints = list(handler.list_checkpoints())
        assert checkpoints == [(f"{tmp_path}/training-state-000000001.checkpoint", 1)]
