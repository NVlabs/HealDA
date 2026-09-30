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
- GitHub Actions `license-check` workflow running `ci/check_licenses.py` and
  `pre-commit` workflow running the ruff hooks.
- `.license-info/` third-party dependency license inventory.

### Changed

- Training is launched through the packaged CLI.
- Active configuration, dataset, observation, model, training, checkpoint, and
  utility modules are organized under the package namespace.

### Removed

- Non-release experiments, legacy workflows, data-preparation implementations,
  site-specific launch configuration, and campaign analysis from the intended
  release boundary.

### Known limitations

- Checkpoint publication decisions and container-based validation remain
  release blockers; no model assets are distributed.
- The training recipes have not been scientifically reviewed.
