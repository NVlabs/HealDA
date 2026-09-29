# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


RUFF = uvx ruff==0.6.9
lint:
	$(RUFF) format --check
	$(RUFF) check
	python3 ci/check_licenses.py

check-licenses:
	python3 ci/check_licenses.py

fix-licenses:
	python3 ci/check_licenses.py --fix

# uvx so this works without pre-commit installed. .pre-commit-config.yaml pins the
# hook versions, so adding a hook there applies here and to the git hook alike.
format:
	uvx pre-commit run -a

test:
	pytest tests/unit

# Reads the NNJA archive, so it skips where the archive is not mounted.
test-integration:
	pytest tests/integration

test-distributed:
	NCCL_DEBUG=warn torchrun --nproc_per_node 8 --local-ranks-filter 0  -m pytest tests/unit/test_distributed.py tests/unit/test_scatter_mean.py