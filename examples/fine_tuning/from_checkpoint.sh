#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Experimental only: this workflow has not been scientifically validated.
# The HealDA v2 configuration runs 8-way time parallel: 8 GPUs, about 150 GB each (README,
# Hardware).
: "${ERA5_HOURLY_ROOT:?set the 0.25 degree ERA5 HDF5 root}"
: "${ERA5_HOURLY_EXT_ROOT:?set the ERA5 extra-variable root}"
: "${NNJA_ROOT:?set the NNJA observation archive root}"
: "${BASE_CHECKPOINT:?set the v2 training-state checkpoint}"

OUTPUT_DIR="${OUTPUT_DIR:-./fine-tuning-runs}"
RUN_NAME="${RUN_NAME:-v2-nnja-latlon-dense}"

torchrun --standalone --nproc-per-node=8 \
  -m healda.cli.train \
  --name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --finetune_from "${BASE_CHECKPOINT}"

# If OUTPUT_DIR/RUN_NAME already contains a valid checkpoint, normal resume
# takes precedence and BASE_CHECKPOINT is not loaded.
