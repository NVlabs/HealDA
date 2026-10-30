#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Quick check: the HealDA v2 model at reduced depth and time length 1, on one GPU.
# Needs the ERA5 and NNJA archives prepared.
: "${ERA5_HOURLY_ROOT:?set the 0.25 degree ERA5 HDF5 root}"
: "${ERA5_HOURLY_EXT_ROOT:?set the ERA5 extra-variable root}"
: "${NNJA_ROOT:?set the NNJA observation archive root}"

OUTPUT_DIR="${OUTPUT_DIR:-./training-runs}"

torchrun --standalone --nproc-per-node=1 \
  -m healda.cli.train \
  --name debug-single-gpu \
  --output_dir "${OUTPUT_DIR}"

# Run this command again with the same name and output directory to resume the
# newest valid training-state checkpoint automatically.
