# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from healda.models.attention import mask_causal


@pytest.mark.parametrize("tq,tk", [(4, 4), (2, 5)])
@pytest.mark.parametrize("linear", [True, False])
def test_causal_mask(tq, tk, linear):
    attn = torch.ones(1, tq, tk, 1, 1)
    masked = mask_causal(attn, linear=linear)
    mat = masked[0, :, :, 0, 0]

    expected = torch.ones(tq, tk)
    fill = 0.0 if linear else float("-inf")
    for q in range(tq):
        for k in range(tk):
            if k > q:
                expected[q, k] = fill

    assert (mat == expected).all()


def test_causal_mask_sliding_window():
    # tq = tk = 5, window = 3: each query attends to itself + 2 previous frames.
    # 1 = kept (weight 1), 0 = masked. Checkable against: k <= q and q - k < 3.
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.float,
    )
    attn = torch.ones(1, 5, 5, 1, 1)
    masked = mask_causal(attn, linear=True, window=3)[0, :, :, 0, 0]
    assert (masked == expected).all()
