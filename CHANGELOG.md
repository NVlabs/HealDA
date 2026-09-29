# Changelog

All notable changes will be documented in this file. The project intends to
follow semantic versioning after its release process is approved.

## Unreleased

### Added

- Installable `healda` package and `healda-train` console entry point.
- Public training, checkpointing, observation-loading, and distributed
  execution package boundary.
- Minimal training and experimental fine-tuning examples.
- Public documentation for prepared observation inputs and release status.
- GitHub issue and pull request templates; `[project.urls]` metadata.
- `healda.inference`: `load_analysis_model` rebuilds a trained network from a
  checkpoint's `loop.json`, and `AnalysisModel.analyze` produces physical-space
  analyses from in-memory GPS-RO and SATWND cycle tables. Supporting
  `TransformV2.transform_observations`, `build_conventional_loader` with table
  sources, and `combined.rebase_conventional`.

### Changed

- Training is launched through the packaged CLI.
- Active configuration, dataset, observation, model, training, checkpoint, and
  utility modules are organized under the package namespace.

### Removed

- Non-release experiments, legacy workflows, data-preparation implementations,
  site-specific launch configuration, and campaign analysis from the intended
  release boundary.

### Known limitations

- `uv.lock` does not match `pyproject.toml`, so there is no reproducible
  install of the pinned environment.
- Release approvals and publishable model assets are outstanding.
- The training recipes have not been scientifically reviewed.
