# Virtualenv setup on ubuntu 24.04 desktop

Instructions for a system like this:

- ubuntu 24.04
- CUDA 13 (driver and runtime)


headal installation commands (last tested on 6/24/26)
```
uv venv --python 3.12 
source .venv/bin/activate

pip install torch hatchling numpy
pip install --no-build-isolation git+https://github.com/NVlabs/earth2grid.git
pip install --no-build-isolation transformer_engine[pytorch,core_cu13]

pip install -e .

# TE was built against a newer version of cublas than this package provides;
# uninstall it and use the system version instead
pip uninstall nvidia_cublas

pip install pre-commit
```
