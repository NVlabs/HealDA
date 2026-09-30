# Third-party license inventory

Run these commands from the repository root to regenerate
`third_party_licenses.csv`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-license uv sync --frozen --no-dev --all-extras
uv pip install --python .venv-license/bin/python pip-licenses

.venv-license/bin/pip-licenses \
  --format=csv \
  --with-urls \
  --ignore-packages healda pip-licenses prettytable wcwidth \
  > .license-info/third_party_licenses.csv
```

Review the generated file and replace any `UNKNOWN` license or URL fields with
information from the corresponding installed package metadata or official
project page.

HealDA does not vendor or redistribute any third-party source code; every
package listed is installed separately at install time under its own license
terms. The `nvidia-*` CUDA runtime wheels and the `cuda-toolkit` metapackage
are not open source and are subject to NVIDIA's own EULA when installed.
