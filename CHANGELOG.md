# Changelog

All notable changes are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Training presets as YAML: `src/healda/configs/loops/<name>.yaml`, one file per preset
  naming the preset it `extends`, the fields it changes and its aliases, loaded into
  `healda.cli.train.LOOPS` by `healda.training.presets`. `healda-train --config` takes a
  preset name or a file, which may extend a shipped preset; `--name` defaults to it, and
  `--verbose` prints the resolved loop as preset YAML. The presets were previously Python constants in
  `healda.cli.train`. `parse_dict` now reads an int into a float field and rejects a float
  YAML read as a string (`1e-4`, `1.0e5`) with a hint.
- MkDocs documentation site with an API reference, built by the `docs` workflow and
  published to GitHub Pages with mike; `make docs` and `make docs-serve`.
- A release tag `X.Y.Z` creates the GitHub release from its `CHANGELOG.md` section and
  publishes the documentation as version `X.Y.Z`, aliased `latest`.
- README badges and `CITATION.cff`.
- `scripts/etl/era5/`: the ERA5 hourly HDF5 year-file ETL (build, verify, thin, split, ARCO
  ERA5T), run as `python -m scripts.etl.era5.<step>` from the repository root;
  `docs/etl_era5.md`.
- `scripts/etl/nnja/`: the NNJA observation ETL (pull, decode PrepBUFR, GPS-RO, satellite
  winds and sounders, thin the IR sounders, radiosonde splice), run as
  `python -m scripts.etl.nnja.<step>` from the repository root; `docs/etl_nnja.md`. The
  decoder, earth2bufr, is in the `etl` extra.
- Documentation: "Building the training data" (`docs/etl.md`), an overview of both ETLs
  linking their pages.
- `healda.observations.schema.footprint_fields(sensor)` and `channel_field(sensor, channel)`:
  the satellite archive's columns, which the NNJA ETL writes the archive with;
  `channel_field(..., kelvin=False)` gives IASI and CrIS channels as stored (radiance).
- Training recipes: the `recipe` and `recipe_args` fields of `TrainingLoop` name a provider
  in `healda.training.providers.RECIPE_PROVIDERS` that supplies the optimizer, the network
  initialisation for training and the lr schedule. Without a recipe (the default) training
  uses AdamW with the warmup and cosine schedule, unchanged.
- `TrainingLoopBase.lr_scale(cur_nimg)`: the lr schedule as a method a subclass or recipe can
  override.
- `stop_nimg`: training ends at that sample count, saving training state there, so a branch
  such as a cooldown stops on the checkpoint it is meant to produce; a launch resuming at or
  past it exits at once.
- `finetune_keep_nimg`: a run started from a `finetune_from` checkpoint keeps that
  checkpoint's sample count, so its lr schedule continues from where the parent stopped.

- Installable `healda` package and `healda-train` console entry point.
- Public training, checkpointing, observation-loading, and distributed
  execution package boundary.
- Minimal training and experimental fine-tuning examples; `examples/inference/local.sh` runs
  `healda-inference-local` over a date range of ETL'd data.
- Public documentation for prepared observation inputs and release status.
- GitHub issue and pull request templates; `[project.urls]` metadata.
- `healda.inference`: `load_da_model` rebuilds a trained lat/lon network from a
  checkpoint's `loop.json`, and `DAModel.run_analysis` produces physical-space
  analyses from in-memory PrepBUFR, GPS-RO, SATWND and satellite cycle tables,
  with dated channel denials (`DENIALS`). Supporting `TransformV2.transform_observations`
  and `build_obs_loader` table sources.
- `healda-inference-local` (`healda.cli.inference_local`): score a checkpoint over a date
  range of local observations and targets, writing the analysis zarr with per-time
  metrics; `healda.inference_output` holds its asynchronous writer and store.
- `healda.observations.adapters.e2s_nnja`: Earth2Studio NNJA frames as NNJA archive
  tables, all streams at once (`analysis_tables` for wide frames, `analysis_tables_long`
  for long ones), keyed by each row's source file (`cycle_time`). `analysis_tables` takes
  no `layout` or `sensors` argument: the sensors are the keys of `sat`.
- NNJA loaders take in-memory cycle tables (`table_source`, `gpsro_table_source`,
  `satwnd_table_source`, `prepbufr_table_source`) in place of the archive.
- `v2-nnja-latlon-dense` training preset, the recipe of the released HealDA v2
  checkpoint: `v2-nnja-latlon-beta` with denser
  observations, observation position features and corrected radiosonde placement. Its
  settings are listed beside it in `healda.cli.train`; its input differences in
  `docs/observations.md`.
- `ObsConfig.nnja_balloon_drift` places radiosonde levels at their drifted position
  and time (off by default), and `ObsConfig.deny_platform_channels` drops platform
  channels at probability 1.
- GitHub Actions `license-check` workflow running `ci/check_licenses.py` and
  `pre-commit` workflow running the ruff hooks.
- `.license-info/` third-party dependency license inventory.
- `healda-inference-local --times-file PATH`: scores an explicit list of times, one per line
  on the `--time-frequency` grid, in place of `--start-date` / `--end-date`.
- `fused_qk_norm` (training loop and model config): the disk-attention q/k RMSNorm and
  the split into flex_attention's layout run as one CuTe DSL kernel.
- `SensorEmbedderConfig.film_bwd_impl=2`: CuTe DSL backward for the fused FiLM tokenizer.
- `healda-inference-local --final-frame-obs-hours N`: keeps only the observations of the first
  N hours in the final frame of each window, emulating a cycle run before late observations
  arrive; earlier frames keep all of theirs.

### Changed

- The default wandb project is `ufs-da`, the project every HealDA run so far logged to.
- `healda.observations.schema` describes the NNJA archive as written: `hpx2048_nest` is
  nullable and source quality flags have per-sensor types (CrIS `scan_quality` uint16).
- `v2-nnja-latlon-final` is renamed `v2-nnja-latlon-beta`; the old name remains an alias.
- Training is launched through the packaged CLI.
- Under time parallelism only the temporal-attention branch is all-to-all'd, in bf16;
  the residual stream stays time-sharded.
- The DiT block stack is compiled with a dynamic observation count, so it no longer
  recompiles when the count first changes.
- The fused FiLM tokenizer indexes rows in int64; int32 row addresses overflowed past
  2**31 / feature-dim rows, which one unsharded analysis window exceeds.
- earth2grid is pinned to a git revision instead of the `main` tarball.
- `ObservationBatch["obs_v2"]` holds, per analysis time, a long table or an
  `Observations`: satellite footprint blocks (`healda.observations.footprints`) plus a
  long table. `NNJAWideLoader` and `CombinedObsLoader` return `Observations`;
  `as_table` and `Observations.to_table` give the long table. `TransformV2` expands
  the blocks in torch straight into the packed tensors, on the GPU in
  `DAModel.run_analysis`. Channel denials and platform/channel dropout clear cells of
  the block.
- Active configuration, dataset, observation, model, training, checkpoint, and
  utility modules are organized under the package namespace.

### Removed

- `healda.utils.storage.get_duckdb_connection`; `duckdb` is no longer a runtime
  dependency.
