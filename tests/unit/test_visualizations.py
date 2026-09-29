# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from healda.utils.visualization import visualize


def test_visualize():
    """Test visualization with reasonable default settings."""
    # Create a simple 1D input array
    x = np.random.rand(12)  # HEALPix level 1 has 12 pixels

    # Test with reasonable defaults
    im = visualize(
        x,
        region="Robinson",
        title="Test Visualization",
        cmap="viridis",
        add_colorbar=True,
    )
    assert im is not None

    # 2D is a lat/lon field, drawn directly rather than read as a HEALPix map.
    assert visualize(np.random.rand(8, 16)) is not None

    # Anything else is neither, so it still raises.
    with pytest.raises(ValueError, match="Expected 1D HEALPix or 2D lat/lon"):
        visualize(np.random.rand(2, 4, 3))
