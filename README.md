# HealDA

HealDA is an experimental package for observation-conditioned, HEALPix-grid
atmospheric data assimilation. It contains the model architecture, distributed
training, inference, and data loaders for observations and target analysis states
already converted to Parquet and Zarr respectively, with the observation
filtering, thinning and normalization applied on load. It does not include
raw-data ETL.

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
the CLI fine-tuning path. See
[`examples/fine_tuning/from_checkpoint.sh`](examples/fine_tuning/from_checkpoint.sh).

## Configuration structure

Configuration has three layers:

1. Environment variables locate state stores, prepared observations, output
   directories, and optional remote profiles.
2. `--name` selects a built-in `TrainingLoop` preset.
3. Top-level CLI fields (`--output_dir`, `--resume_dir`, `--finetune_from`) and
   nested `--loop.*` flags configure a run without a preset.

Run `healda-train --help` against the installed version for the authoritative
flag list. If `--name` does not match a preset, the CLI uses the `--loop.*`
configuration.

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
Inference currently needs about 40 GB of GPU memory and has been run on H100 and
Blackwell GPUs.

## Prepared observation input

Training reads prepared Parquet observations from the NNJA archive (`NNJA_ROOT`),
the source the checkpoints are trained on. The older UFS replay archive
(`UFS_OBS_PATH`) is still readable for earlier runs. Each has its own layout, schema
and channel vocabulary, and a checkpoint reads only the source it was trained on.
See [`docs/observation-input.md`](docs/observation-input.md), including what data
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
