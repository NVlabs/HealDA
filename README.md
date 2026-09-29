# HealDA

HealDA is an experimental training package for observation-conditioned,
HEALPix-grid atmospheric data assimilation. The public package boundary covers
training, checkpointing, supported state datasets, prepared-observation loaders,
model components, and distributed execution. It does not include raw-data ETL,
prepared datasets, checkpoints, scientific benchmark claims, or an end-user
inference application.

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
export UFS_OBS_PATH=/data/prepared_observations/
export CHECKPOINT_ROOT=/runs

torchrun --standalone --nproc-per-node=1 \
  -m healda.cli.train --name debug-single-gpu --output_dir /runs
```

This smoke recipe is minimal in duration, not in data requirements. It expects
the prepared state, observation, channel-table, and normalization assets used by
the selected recipe. See [`examples/training/minimal.sh`](examples/training/minimal.sh).

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
the CLI fine-tuning path. Fine-tuning is experimental and has not been
scientifically validated; do not treat the example as a supported scientific
recipe. See
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

## Prepared observation input

Training consumes prepared Parquet, not raw BUFR or NetCDF. The supported
loader-facing contract is documented in
[`docs/observation-input.md`](docs/observation-input.md). In brief, archives must
use the expected daily or cycle file layout, exact column names and compatible
Arrow types, UTC timestamps, stable sensor/platform/channel vocabularies, and a
matching `channel_table.parquet`. Observation creation and archive migration are
outside the package boundary.

## Inference boundary

HealDA owns training and its checkpoint/model interfaces. Earth2Studio is the
intended integration boundary for future inference workflows, but this
repository does not currently expose a supported Earth2Studio model wrapper or
inference CLI. Consumers should not assume a training checkpoint is directly
loadable by Earth2Studio until a versioned adapter and checkpoint compatibility
tests are published.

## Release model

The project is pre-release software. Releases are expected to use semantic
versioning while the Python API, recipes, data contracts, and checkpoint format
remain subject to change. Compatibility guarantees begin only when explicitly
stated in release notes.

## Known limitations and blockers

- No prepared datasets, checkpoints, or turnkey data-preparation pipeline are
  distributed.
- Training requires CUDA; supported hardware/software combinations have not
  been published.
- Built-in recipes include research configurations and are not all validated
  release recipes.
- Fine-tuning and checkpoint portability across configuration changes are not
  scientifically validated.
- A supported Earth2Studio inference adapter is not yet available.
- Checkpoint publication decisions and container-based validation remain
  release blockers.

## Contributing

This project is currently not accepting contributions.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development guidance.

## Disclaimer

This project will download and install additional third-party open-source
software. Review the license terms of those projects before use.
