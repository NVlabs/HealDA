#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run a HealDA checkpoint over a date range of locally ETL'd data and write the analyses
# to a Zarr store. Needs the ERA5 and NNJA data ETL'd (docs/etl.md) and a
# training-state checkpoint, e.g. HealDA v2 from https://huggingface.co/nvidia/healda-v2.
: "${ERA5_HOURLY_ROOT:?set the 0.25 degree ERA5 HDF5 root}"
: "${ERA5_HOURLY_EXT_ROOT:?set the ERA5 extra-variable root}"
: "${NNJA_ROOT:?set the NNJA observation archive root}"
: "${CHECKPOINT:?set the training-state checkpoint}"

# --time-parallel splits each window's frames across the GPUs, so it equals the GPU count.
GPUS="${GPUS:-4}"
# Each analysis time ends an 8-frame, 6-hourly window inside the same year, so the first
# analysis time of a year is 18 UTC on 2 January.
START_DATE="${START_DATE:-2025-01-08-00}"
END_DATE="${END_DATE:-2025-01-14-18}"
OUTPUT_PATH="${OUTPUT_PATH:-./healda-analysis.zarr}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc-per-node="${GPUS}" \
  -m healda.cli.inference_local \
  --checkpoint-path "${CHECKPOINT}" \
  --time-parallel "${GPUS}" \
  --batch-gpu 1 \
  --time-frequency 6h \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --output-path "${OUTPUT_PATH}"
