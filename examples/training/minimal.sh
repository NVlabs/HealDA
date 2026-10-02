#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# The smoke preset is short but still requires compatible prepared assets.
: "${ERA5_HPX64_104CH_ZARR:?set the packed state Zarr path}"
: "${NNJA_ROOT:?set the NNJA observation archive root}"

OUTPUT_DIR="${OUTPUT_DIR:-./training-runs}"

torchrun --standalone --nproc-per-node=1 \
  -m healda.cli.train \
  --name debug-single-gpu \
  --output_dir "${OUTPUT_DIR}"

# Run this command again with the same name and output directory to resume the
# newest valid training-state checkpoint automatically.
