#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Experimental only: this workflow has not been scientifically validated.
: "${ERA5_HOURLY_ROOT:?set the ERA5 0.25 degree hourly HDF5 root}"
: "${ERA5_HOURLY_EXT_ROOT:?set the ERA5 0.25 degree extra-variable HDF5 root}"
: "${NNJA_ROOT:?set the NNJA observation archive root}"
: "${BASE_CHECKPOINT:?set a compatible training-state checkpoint}"
: "${RDZV_ENDPOINT:?set host:port of the first node}"

OUTPUT_DIR="${OUTPUT_DIR:-./fine-tuning-runs}"
RUN_NAME="${RUN_NAME:-v2-nnja-latlon-final}"

# The recipe uses time_parallel=8, so the GPU count must be a multiple of 8. It was
# trained on two 4-GPU GB300 nodes. Run this on every node.
torchrun --nnodes "${NNODES:-2}" --nproc-per-node "${NPROC_PER_NODE:-4}" \
  --rdzv-backend c10d --rdzv-endpoint "${RDZV_ENDPOINT}" \
  -m healda.cli.train \
  --name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --finetune_from "${BASE_CHECKPOINT}"

# If OUTPUT_DIR/RUN_NAME already contains a valid checkpoint, normal resume
# takes precedence and BASE_CHECKPOINT is not loaded.
