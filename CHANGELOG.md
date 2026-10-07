# Changelog

All notable changes will be documented in this file. The project intends to
follow semantic versioning starting with its first tagged release.

## Unreleased

### Added

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
- `ObsConfig.nnja_balloon_drift` places radiosonde levels at their drifted position
  and time (off by default), and `ObsConfig.deny_platform_channels` drops platform
  channels at probability 1.
- GitHub Actions `license-check` workflow running `ci/check_licenses.py` and
  `pre-commit` workflow running the ruff hooks.
- `.license-info/` third-party dependency license inventory.

### Changed

- Training is launched through the packaged CLI.
- The fused FiLM tokenizer indexes rows in int64; int32 row addresses overflowed past
  2**31 / feature-dim rows, which one unsharded analysis window exceeds.
- earth2grid is pinned to a git revision instead of the `main` tarball.
- Active configuration, dataset, observation, model, training, checkpoint, and
  utility modules are organized under the package namespace.

### Removed

- `healda.utils.storage.get_duckdb_connection`; `duckdb` is no longer a runtime
  dependency.
