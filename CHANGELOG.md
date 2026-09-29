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
- `SECURITY.md` with the NVIDIA PSIRT reporting process and project threat
  model; `.security-triage.yaml` exposure classifications.
- GitHub Actions `license-check` workflow running `ci/check_licenses.py`,
  replacing the GitLab CI configuration, and a `pre-commit` workflow running
  the ruff hooks.
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

- `uv.lock` does not match `pyproject.toml`, so there is no reproducible
  install of the pinned environment.
- Release approvals and publishable model assets are outstanding.
- The training recipes have not been scientifically reviewed.
