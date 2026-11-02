# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flatten nested earth2bufr PrepBUFR batches, choosing which event to keep.

earth2bufr.flatten_batch keeps the FIRST element of each event stack, which by
NCEP convention is the NEWEST (most-processed) value. This module does the same
row expansion but lets the caller choose the element:

  take="first"  -> newest / most-processed value
  take="last"   -> oldest / bottom-of-stack value = the PREPRO (raw) value

Both use the same row expansion, so the two produce row-aligned tables.

The level list is expanded only in batches where some report has more than one
level, and a list that is empty in every row is dropped rather than promoted to
null columns.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def _unwrap_unit_list_columns(batch: pa.RecordBatch, take: str) -> pa.RecordBatch:
    if batch.num_rows == 0:
        return batch

    changed = True
    while changed:
        changed = False
        columns: dict[str, pa.Array] = {}
        existing_names: set[str] = set()

        for i in range(batch.num_columns):
            field = batch.schema.field(i)
            col = batch.column(i)

            if not pa.types.is_list(field.type):
                columns[field.name] = col
                existing_names.add(field.name)
                continue

            inner_type = field.type.value_type
            if not pa.types.is_struct(inner_type):
                columns[field.name] = col
                existing_names.add(field.name)
                continue

            lengths = pc.fill_null(pc.list_value_length(col), 0)
            max_len = pc.max(lengths).as_py()
            if max_len is None or max_len == 0:
                continue

            nonempty_mask = pc.greater(lengths, 0)
            existing_valid = pc.is_valid(col)
            combined_valid = pc.and_(existing_valid, nonempty_mask)

            offsets_np = col.offsets.to_numpy()
            valid_np = combined_valid.to_numpy(zero_copy_only=False)

            # first element -> start offset; last element -> end offset - 1
            idx_src = (offsets_np[1:] - 1) if take == "last" else offsets_np[:-1]
            indices_np = np.where(valid_np, idx_src, 0).astype(np.int64)

            idx_arr = pa.array(indices_np, type=pa.int64())
            unwrapped = pc.take(col.values, idx_arr)

            valid_buf = pa.array(valid_np).buffers()[1]
            unwrapped = pa.StructArray.from_buffers(
                inner_type,
                len(unwrapped),
                [valid_buf],
                children=[unwrapped.field(j) for j in range(inner_type.num_fields)],
            )

            parent_valid = combined_valid
            has_nulls = pc.any(pc.invert(parent_valid)).as_py()
            for j in range(inner_type.num_fields):
                sf = inner_type.field(j)
                fname = sf.name
                candidate = fname
                suffix = 2
                while candidate in existing_names:
                    candidate = f"{fname}_{suffix}"
                    suffix += 1
                child = unwrapped.field(j)
                if has_nulls:
                    if pa.types.is_list(sf.type):
                        child_valid_mask = pc.and_(pc.is_valid(child), parent_valid)
                        valid_buf = child_valid_mask.buffers()[1]
                        child = pa.ListArray.from_buffers(
                            sf.type,
                            len(child),
                            [valid_buf, child.buffers()[1]],
                            children=[child.values],
                        )
                    elif pa.types.is_struct(sf.type):
                        child_valid_mask = pc.and_(pc.is_valid(child), parent_valid)
                        valid_buf = child_valid_mask.buffers()[1]
                        child = pa.StructArray.from_buffers(
                            sf.type,
                            len(child),
                            [valid_buf],
                            children=[
                                child.field(k) for k in range(sf.type.num_fields)
                            ],
                        )
                    else:
                        null_scalar = pa.scalar(None, type=sf.type)
                        child = pc.if_else(parent_valid, child, null_scalar)
                columns[candidate] = child
                existing_names.add(candidate)
            changed = True

        if changed:
            batch = pa.RecordBatch.from_pydict(columns)

    return batch


def _flatten_keep_lists(batch: pa.RecordBatch, target: str) -> pa.RecordBatch:
    idx = batch.schema.get_field_index(target)
    if idx < 0:
        return batch
    list_col = batch.column(idx)
    lengths = pc.fill_null(pc.list_value_length(list_col), 0)
    mask = pc.greater(lengths, 0)
    filtered = batch.filter(mask)
    filtered_list = filtered.column(idx)
    if filtered.num_rows == 0:
        return pa.record_batch({}, schema=pa.schema([]))

    flat_values = pc.list_flatten(filtered_list)
    indices = pc.list_parent_indices(filtered_list)

    columns: dict[str, pa.Array] = {}
    for i in range(filtered.num_columns):
        f = filtered.schema.field(i)
        if f.name == target:
            continue
        columns[f.name] = pc.take(filtered.column(i), indices)

    if pa.types.is_struct(flat_values.type):
        for j in range(flat_values.type.num_fields):
            sf = flat_values.type.field(j)
            fname = sf.name
            candidate = fname
            suffix = 2
            while candidate in columns:
                candidate = f"{fname}_{suffix}"
                suffix += 1
            columns[candidate] = flat_values.field(j)
    else:
        columns[target] = flat_values
    return pa.RecordBatch.from_pydict(columns)


def recursive_flatten(batch: pa.RecordBatch, take: str = "last") -> pa.RecordBatch:
    """Row-expand multi-level obs then unwrap event stacks, keeping first/last.

    take="last" keeps the bottom-of-stack (PREPRO / raw) event per variable.
    take="first" keeps the newest.
    """
    # Phase 1: expand the multi-level list<struct> whose inner struct has "CAT".
    target = None
    for i in range(batch.num_columns):
        f = batch.schema.field(i)
        if pa.types.is_list(f.type):
            inner = f.type.value_type
            if pa.types.is_struct(inner):
                lengths = pc.fill_null(pc.list_value_length(batch.column(i)), 0)
                if (pc.max(lengths).as_py() or 0) > 1:
                    inner_names = {inner.field(j).name for j in range(inner.num_fields)}
                    if "CAT" in inner_names:
                        target = f.name
                        break
    if target is not None:
        batch = _flatten_keep_lists(batch, target)
    if batch.num_rows == 0:
        return batch

    # Phase 2: unwrap remaining unit/event list<struct> columns.
    return _unwrap_unit_list_columns(batch, take)
