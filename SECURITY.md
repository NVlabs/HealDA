# Security Policy: HealDA

## Security

NVIDIA is dedicated to the security and trust of our software products and
services, including all source code repositories managed through our organization.

If you need to report a security issue, use the contact points below. **Do not
report security vulnerabilities through public GitHub issues, discussions, or pull
requests.** If a potential security issue is disclosed publicly, NVIDIA maintainers
may limit public discussion and redirect the reporter to the private reporting
channels.

## Reporting a potential security vulnerability in an NVIDIA product

To report a potential security vulnerability in an NVIDIA product:

- Web: [Report a Security Vulnerability or NVIDIA AI Concern](https://www.nvidia.com/en-us/security/report-vulnerability/)
- Email: [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - NVIDIA encourages encryption with its [public PGP key](https://www.nvidia.com/en-us/security/pgp-key/).
  - Include the affected product or driver and its version or branch, the type of
    vulnerability, reproduction instructions, proof-of-concept or exploit code when
    available, and the potential impact.

NVIDIA follows a coordinated vulnerability disclosure process. See the
[NVIDIA PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/) for
details.

## NVIDIA Product Security

For security bulletins, policies, acknowledgements, and reporting resources, visit
the [NVIDIA Product Security portal](https://www.nvidia.com/en-us/product-security/).

## Security Architecture & Context

HealDA (Python package `healda`) is a training package for observation-conditioned
atmospheric data assimilation on the HEALPix grid. It trains transformer-based
networks on prepared ERA5/UFS state stores and prepared observation archives, and
writes checkpoints for downstream use.

This software operates at the **library / CLI research-SDK** level. It is executed
directly by a researcher through the `healda-train` entry point, typically under
`torchrun` or a SLURM launcher. It exposes no application-level API or
authentication surface; the only network listener it creates is PyTorch's
distributed rendezvous and collective ports during multi-process training, which are
unauthenticated and must be confined to a trusted cluster network. Its primary
security responsibility is the safe handling of the artifacts it ingests on a
trusted host: model checkpoints, scientific datasets, and training configuration.

**Repository Exposure Classification:** Public. Distributed as an open-source project
on public GitHub; this document is written to public-safe detail.

**Service Exposure Classification:** External. An externally distributed
open-source research release (Apache-2.0); not a customer-facing, commercial, or
regulated service.

**Security boundaries.** Trusted inputs are the local execution environment and the
checkpoints, datasets, and CLI/preset configuration the operator chooses to run.
Untrusted inputs are any of those obtained from third parties. Data enters through
local files, S3-compatible object storage, and plain HTTP downloads (zarr, HDF5,
Parquet, NetCDF, `.checkpoint` zip archives), through CLI arguments and `.env`
variables, and through an rclone configuration file. Data leaves as written
checkpoints, TensorBoard logs, and optional Weights & Biases telemetry.

### Threat Model

1. **Malicious checkpoint deserialization.** Checkpoints are zip archives whose
   tensors are read with `torch.load` (`healda/training/checkpoint.py`,
   `healda/training/loop.py`, `healda/training/distributed_checkpoint.py`).
   Current-format loads pass `weights_only=True` or rely on the `weights_only`
   default of the pinned PyTorch (>= 2.6), but the legacy-format fallback in
   `Checkpoint.read_model` runs for any archive without `metadata.json` and uses
   `weights_only=False`, which deserializes arbitrary pickle data. FSDP checkpoints
   are directories loaded with `torch.distributed.checkpoint.load`, whose metadata is
   pickle-serialized. Loading an untrusted checkpoint of either form can execute
   arbitrary code with the operator's privileges.
2. **Untrusted dataset parsing.** The data pipeline opens zarr, HDF5, Parquet,
   NetCDF, and DuckDB inputs through xarray, h5py, pyarrow, and duckdb, parses
   checkpoint-embedded JSON (`model.json`, `metadata.json`, `loop.json`), and
   `healda.utils.storage.ensure_downloaded` fetches arbitrary URLs over HTTP without
   integrity verification. Crafted inputs could trigger parser flaws or resource
   exhaustion.
3. **Supply-chain compromise of dependencies.** `earth2grid` is resolved from a
   GitHub source archive of its default branch (`[tool.uv.sources]`); `uv.lock`
   records a content hash for frozen installs, but a fresh resolution is unpinned.
   `nvidia-physicsnemo` uses a lower bound that admits pre-releases. A compromised
   or relocated upstream could inject code at install time.
4. **Credential exposure via configuration.** S3 credentials are read from the
   operator's rclone configuration (`healda/utils/storage.py`) and passed to s3fs and
   DuckDB, where they appear in the connection's SQL text; store paths and profile
   names come from `.env` and are printed at startup. Committing a real `.env` or
   rclone config, sharing job logs, or over-broad object-store credentials could
   leak credentials or data locations.
5. **Information disclosure via experiment telemetry.** Weights & Biases is a hard
   dependency; logging is enabled by default (`wandb_enabled=True`) and activates
   whenever `WANDB_API_KEY` is set, uploading metrics and the full training
   configuration to an external service. Training on non-public data without
   disabling it could disclose sensitive information.
6. **Unauthenticated distributed rendezvous.** `torch.distributed.init_process_group`
   with `init_method="env://"` binds a TCPStore on `MASTER_PORT` (default 29500) and
   NCCL opens further sockets. Any process that can reach these ports on the cluster
   network can join or disrupt a training job.

### Critical Security Assumptions

- Checkpoints and datasets loaded by the operator are trusted; the project does
  not sandbox `torch.load` or pickle deserialization.
- The execution host (workstation or HPC node) and its cluster network are trusted
  and access-controlled; the software performs no authentication or authorization of
  its own.
- Credentials are provided by a trusted operator via the environment and a local
  rclone config, and the object store enforces its own access control; the project
  does not manage secret storage.
- Dependencies are obtained from trusted sources; beyond the hashes in `uv.lock`, no
  integrity verification of packages or downloaded files is performed.
- Transport security (TLS) for object-store and telemetry traffic is handled by the
  underlying client libraries and infrastructure.
