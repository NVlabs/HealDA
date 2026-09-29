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

See [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md) for the full text of
each open-source license family, pooled once per family with a link to each
package's own license file, and
[`THIRD-PARTY-LICENSES-perpackage.md`](THIRD-PARTY-LICENSES-perpackage.md)
for the same information with each package's copyright notice and full
license text embedded inline, one package after another.
