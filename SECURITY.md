# Security Policy: HealDA

NVIDIA is dedicated to the security and trust of our software products and
services, including all source code repositories managed through our
organization.

If you need to report a security issue, please use the appropriate contact
points outlined below. **Please do not report security vulnerabilities through
GitHub issues or pull requests.**

## Reporting a Vulnerability

To report a potential security vulnerability in HealDA:

* **Web (preferred):** [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
* **E-Mail:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage you to use the following PGP key for secure email communication:
    [NVIDIA public PGP Key](https://www.nvidia.com/en-us/security/pgp-key)
* **GitHub:** Use this repository's **Security** tab > **Report a vulnerability**
  to submit a report privately.

If a vulnerability is reported through public channels (issues, pull requests, or
discussions), maintainers may limit public discussion and redirect the reporter
to the private channels above.

### What to Include

- Project name and version or branch affected
- Type of vulnerability (e.g., deserialization, code execution, information disclosure)
- Step-by-step instructions to reproduce
- Proof-of-concept code (if available)
- Potential impact assessment

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge receipt,
validate the issue and assess severity, develop and test a fix, and publish a
security bulletin as appropriate. While NVIDIA does not currently operate a public
bug bounty program, externally reported issues are acknowledged under our
coordinated vulnerability disclosure policy.

## Security Architecture & Context

HealDA (Python package `healda`) is a training package for
observation-conditioned atmospheric data assimilation on the HEALPix grid. It
trains transformer-based networks on prepared ERA5/UFS state stores and
prepared observation archives, and writes checkpoints for downstream use.

This software operates at the **Library / CLI research-SDK** level. It is
executed directly by a researcher (via the `healda-train` entry point, typically
under `torchrun` or a SLURM launcher); it is **not** a long-running network
service and exposes no inbound API, listener, or authentication surface. Its
primary security responsibility is the safe handling of the artifacts it ingests
on a trusted host: model checkpoints, scientific datasets, and training
configuration.

**Repository Exposure Classification:** Public.
Basis: distributed as an open-source project on public GitHub; this document is
written to public-safe detail.

**Service Exposure Classification:** External / Regulated (medium confidence).
Basis: externally distributed open-source research release (Apache-2.0); it is
not a customer-facing or commercially-supported service and handles only
scientific data with operator-supplied credentials, but external distribution
places it above the internal tiers.

**Security boundaries.** Trusted inputs are the local execution environment and
the checkpoints, datasets, and CLI/preset configuration the operator chooses to
run. Untrusted inputs are any of those obtained from third parties. Data enters
through local files and S3-compatible object storage (zarr / HDF5 / Parquet /
`.checkpoint` zip archives), CLI arguments, `.env` variables, and an rclone
configuration file, and exits as written checkpoints, TensorBoard logs, and
optional Weights & Biases telemetry.

### Threat Model

The following scenarios represent the primary security concerns for this
project:

1. **Malicious model checkpoint deserialization:** checkpoints are zip archives
   containing `torch.load`-ed state (`healda/training/checkpoint.py`,
   `healda/training/loop.py`, `healda/training/distributed_checkpoint.py`).
   Current-format loads use `weights_only=True`, but the legacy-format fallback
   in `Checkpoint.read_model` uses `weights_only=False`, which deserializes
   arbitrary Python pickle data. Loading an untrusted legacy checkpoint can
   execute arbitrary code with the operator's privileges.
2. **Untrusted dataset parsing:** the data pipeline opens zarr, HDF5, Parquet,
   and DuckDB inputs through xarray, h5py, pyarrow, and duckdb, and parses
   checkpoint-embedded JSON (`model.json`, `metadata.json`). Crafted input files
   could trigger parser flaws or cause resource exhaustion.
3. **Supply-chain compromise of dependencies:** `earth2grid` is installed from a
   GitHub source archive of its default branch and `earth2studio` from a git
   source (see `[tool.uv.sources]` in `pyproject.toml`); `nvidia-physicsnemo` is
   pinned to a pre-release. A compromised or relocated upstream could inject code
   at install time.
4. **Credential exposure via configuration:** S3 credentials are read from the
   operator's rclone configuration (`healda/utils/storage.py`) and passed to
   s3fs / DuckDB; store paths and profile names come from `.env`. Committing a
   real `.env` or rclone config, logging the environment, or over-broad
   object-store credentials could leak credentials or data.
5. **Information disclosure via experiment telemetry:** Weights & Biases
   logging is enabled by default in the training loop (`wandb_enabled=True`)
   when the `wandb` package is importable and `WANDB_API_KEY` is set. Training
   on non-public data without disabling telemetry could disclose sensitive
   information to an external service.

### Critical Security Assumptions

- Assumes model checkpoints and datasets loaded by the operator are trusted; the
  project does not sandbox `torch.load` / pickle deserialization.
- Assumes the execution host (workstation or HPC node) is trusted and
  access-controlled; the software performs no authentication or authorization of
  its own.
- Assumes credentials are provided by a trusted operator via the environment and
  a local rclone config, and that the object store enforces its own access
  control; the project does not manage secret storage.
- Assumes dependencies are obtained from trusted sources; no integrity
  verification of packages is performed at install time.
- Assumes transport security (TLS) for object-store and telemetry traffic is
  handled by the underlying client libraries and infrastructure.
