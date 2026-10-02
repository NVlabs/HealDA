#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Experimental only: this workflow has not been scientifically validated.
: "${ERA5_HPX64_104CH_ZARR:?set the packed state Zarr path}"
: "${NNJA_ROOT:?set the NNJA observation archive root}"
: "${BASE_CHECKPOINT:?set a compatible training-state checkpoint}"

OUTPUT_DIR="${OUTPUT_DIR:-./fine-tuning-runs}"
RUN_NAME="${RUN_NAME:-debug-single-gpu}"

torchrun --standalone --nproc-per-node=1 \
  -m healda.cli.train \
  --name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --finetune_from "${BASE_CHECKPOINT}"

# If OUTPUT_DIR/RUN_NAME already contains a valid checkpoint, normal resume
# takes precedence and BASE_CHECKPOINT is not loaded.
