# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-faithful NNJA satellite-wind (SATWND/AMV) archive: one row per AMV, each
carrying a complete wind vector with its own assigned pressure and quality metadata.

``archive_etl`` decodes and projects; ``amv_writer`` binds the AMV footer and contract
onto the shared ``archive_common`` day writer; ``run_production_etl`` is the driver.
"""
