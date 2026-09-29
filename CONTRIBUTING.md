# Contributing

This project is currently not accepting contributions. HealDA is licensed
under the [Apache License 2.0](LICENSE); this covers use, modification, and
redistribution of the code, but the project is not yet set up to accept
external contributions (e.g. no CLA/DCO process is in place).

For authorized development:

1. Create a focused branch and keep changes limited to one purpose.
2. Install development dependencies with `uv sync`.
3. Install checks with `uv run pre-commit install`.
4. Add simple tests for behavior changes.
5. Run `make lint`; if formatting fails, run `make format` and repeat lint.
6. Run the relevant unit tests. Distributed changes require
   `make test-distributed` in a supported CUDA environment.
7. Document user-visible changes under the Unreleased section of
   `CHANGELOG.md`.

Do not include credentials, private URLs, account names, machine-specific
paths, datasets, checkpoints, or other non-public artifacts. New training
recipes must state their data contract, compute assumptions, checkpoint
compatibility, and scientific validation status.

Bug reports should include a minimal reproduction, HealDA revision, Python and
dependency versions, GPU and driver information, launch command with secrets
removed, and the complete error message.
