# Changelog

All notable changes will be documented in this file. The project intends to
follow semantic versioning starting with its first tagged release.

## Unreleased

### Added

- MkDocs documentation site with an API reference, built by the `docs` workflow and
  published to GitHub Pages with mike; `make docs` and `make docs-serve`.
- README badges and `CITATION.cff`.

- Installable `healda` package and `healda-train` console entry point.
- Public training, checkpointing, observation-loading, and distributed
  execution package boundary.
- Minimal training and experimental fine-tuning examples.
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
  tables (`analysis_tables` for all streams at once), keyed by each row's source file
  (`cycle_time`).
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
- `fused_qk_norm` (training loop and model config): the disk-attention q/k RMSNorm and
  the split into flex_attention's layout run as one CuTe DSL kernel.
- `SensorEmbedderConfig.film_bwd_impl=2`: CuTe DSL backward for the fused FiLM tokenizer.

### Changed

- The default wandb project is `ufs-da`, the project every HealDA run so far logged to.
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
  the block. `_device_transform_unified_obs` empties the dict it is given.
- Active configuration, dataset, observation, model, training, checkpoint, and
  utility modules are organized under the package namespace.

### Removed

- `healda.utils.storage.get_duckdb_connection`; `duckdb` is no longer a runtime
  dependency.
