# HealDA

![Maturity: L2 Hardened](https://img.shields.io/badge/Maturity-L2%20Hardened-5C6BC0?style=flat-square)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)](LICENSE)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?style=flat-square)
[![Docs](https://img.shields.io/badge/docs-nvlabs.github.io%2FHealDA-blue?style=flat-square)](https://nvlabs.github.io/HealDA/)

[![pre-commit](https://img.shields.io/github/actions/workflow/status/NVlabs/HealDA/pre-commit.yml?branch=main&label=pre-commit&style=flat-square)](https://github.com/NVlabs/HealDA/actions/workflows/pre-commit.yml)
[![license-check](https://img.shields.io/github/actions/workflow/status/NVlabs/HealDA/license-check.yml?branch=main&label=license-check&style=flat-square)](https://github.com/NVlabs/HealDA/actions/workflows/license-check.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/NVlabs/HealDA/docs.yml?branch=main&label=docs&style=flat-square)](https://github.com/NVlabs/HealDA/actions/workflows/docs.yml)

HealDA is an experimental package for observation-conditioned, HEALPix-grid
atmospheric data assimilation. It contains the model architecture, distributed
training, inference, and data loaders for prepared observations (Parquet) and
target analysis states, with the observation filtering, thinning and
normalization applied on load. It does not include raw-data ETL.

> This code is provided for research and development purposes only.

## Install

HealDA requires Python 3.11 or newer, PyTorch with CUDA support, and a compatible
GPU software stack.

```bash
uv sync
uv run healda-train --help
```

For an editable environment:

```bash
uv pip install -e .
healda-train --help
```

For training on Blackwell GPUs (compute capability 10.0 and newer), install the CuTe DSL
attention kernels. They are optional, but a significant speed-up (2-3x on the attention
kernels), and the 0.25 degree recipe requests them:

```bash
uv sync --extra cutedsl        # or: uv pip install -e '.[cutedsl]'
python -c "import cutlass.cute"
```

Without them, or on older GPUs, training falls back to Triton (pixel attention) and einsum
(temporal attention) kernels and logs one warning for each at start-up.

HealDA is licensed under the [Apache License 2.0](LICENSE).

## Training

`healda-train` is the packaged entry point. Training recipes are named presets
in `healda.cli.train`; nested `--loop.*` options expose the underlying
dataclasses. Option names retain underscores.

Set paths through environment variables or a `.env` file:

```bash
export ERA5_HPX64_104CH_ZARR=/data/era5_hpx64_104ch.zarr
export NNJA_ROOT=/data/nnja/
export CHECKPOINT_ROOT=/runs

torchrun --standalone --nproc-per-node=1 \
  -m healda.cli.train --name debug-single-gpu --output_dir /runs
```

This smoke recipe is minimal in duration, not in data requirements. It expects
the prepared state Zarr and NNJA observation archive used by the selected
recipe. See [`examples/training/minimal.sh`](examples/training/minimal.sh).

### Resume and fine-tune

Reusing the same `--output_dir` and `--name` automatically resumes the newest
valid `training-state-*.checkpoint` in that run directory. An alternate run can
be searched with `--resume_dir`:

```bash
healda-train --name debug-single-gpu --output_dir /runs \
  --resume_dir /runs/previous-run
```

`--finetune_from` initializes model weights from a checkpoint only when no
resumable checkpoint is found. Optimizer and iterator state are not restored by
the CLI fine-tuning path, and the sample count starts at zero, so the preset's
learning-rate schedule (warm-up and decay) runs from the start: give a fine-tuning
preset its own schedule. Resuming the original run instead continues it at the end of
its schedule. See
[`examples/fine_tuning/from_checkpoint.sh`](examples/fine_tuning/from_checkpoint.sh),
which uses the HEALPix smoke recipe and does not load the 0.25 degree checkpoints.

## Configuration structure

Configuration has three layers:

1. Environment variables locate state stores, prepared observations, output
   directories, and optional remote profiles.
2. `--name` selects a built-in `TrainingLoop` preset.
3. Top-level CLI fields (`--output_dir`, `--resume_dir`, `--finetune_from`) and
   nested `--loop.*` flags configure a run without a preset.

Run `healda-train --help` against the installed version for the authoritative
flag list. If `--name` does not match a preset, the CLI uses the `--loop.*`
configuration. If it does match, the preset is used as defined and `--loop.*` flags are
ignored: a run with different settings needs its own preset.

## Distributed execution

Launch with `torchrun`, or with a scheduler that supplies the standard rank
environment. HealDA supports data parallelism, optional FSDP over the data
dimension, and explicit time and space activation sharding. The world size must
be divisible by `time_parallel * space_parallel`; the video length must also be
compatible with time sharding. FSDP with `space_parallel > 1` is not wired.
Multi-process and multi-node configurations require CUDA/NCCL.

## Hardware

The checkpoints come from the 0.25 degree recipe `v2-nnja-latlon-final` in
`healda.cli.train`. It has been trained on GB300 nodes with `time_parallel=8`,
eight GPUs per video sample. `time_parallel=4` also trains without changes.
Training it on H100-class GPUs needs further memory savings, such as more
activation checkpointing, and has not been validated.
`v2-nnja-latlon-dense` is that recipe with denser observations, observation position features
and corrected radiosonde placement; its settings are listed beside it in
`healda.cli.train`. It reads more observations per sample, so inference with it needs
`healda-inference-local --prefetch-factor 2` to stay within host memory.
As a reference for a new setup, that recipe trains at about 126 seconds per thousand
samples (`Timing/sec_per_kimg` in the log) on 32 GB300 GPUs (batch 4, `time_parallel=8`),
about 110k samples in a 4-hour job. A much slower run usually lacks the CuTe DSL kernels or
reads data from a slow filesystem.
Inference currently needs about 40 GB of GPU memory and has been run on H100 and
Blackwell GPUs.

## Prepared observation input

Training reads prepared Parquet observations from the NNJA archive (`NNJA_ROOT`),
the source the checkpoints are trained on. The older UFS replay archive
(`UFS_OBS_PATH`) is still readable for earlier runs. Each has its own layout, schema
and channel vocabulary, and a checkpoint reads only the source it was trained on.
See [`docs/observations.md`](docs/observations.md), including what data
must satisfy to match a trained checkpoint.

## Inference

`healda.inference.load_da_model` rebuilds a trained network from a checkpoint's
`loop.json`, and `DAModel.run_analysis` produces analyses from in-memory observation
cycle tables. `healda.observations.adapters.e2s_nnja.analysis_tables` builds those
tables from Earth2Studio NNJA observation frames.

## Contributing

This project is currently not accepting contributions.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development guidance.

## Disclaimer

This project will download and install additional third-party open-source
software. Review the license terms of those projects before use.
