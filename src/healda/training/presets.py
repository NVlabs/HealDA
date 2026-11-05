# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training presets as YAML files, one per preset, each a few changes over another.

A preset file has up to three top-level keys. ``extends`` names the preset this one starts
from; ``loop`` holds the fields that differ from it, nested the way the training loop
dataclass is (a mapping under ``obs_config`` changes only the keys it lists, a list or
scalar replaces the parent's value). Without ``extends``, ``loop`` is the whole loop and
anything missing takes the dataclass default. ``aliases`` lists other names that resolve
to the same preset. The file's stem is the preset name.

The shipped presets live in ``healda.configs.loops`` and may only extend each other. A file
passed to ``healda-train --config`` may extend any shipped preset, so a new experiment is a
file with ``extends`` and the handful of fields it changes. ``healda-train --verbose``
prints the resolved loop in this form, every field written out.

The loop type is passed in rather than imported, so the training CLI can import this module
to fill its registry without a cycle.
"""

import dataclasses
import importlib.resources
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Type, TypeVar

import yaml

from healda.utils.parsing import dataclass_field_type

T = TypeVar("T")

PRESET_KEYS = {"aliases", "extends", "loop"}
# Set per run, never part of a preset.
RUN_SPECIFIC_FIELDS = ("device", "run_dir", "wandb_id")

DEFAULT_DIR = Path(str(importlib.resources.files("healda.configs") / "loops"))


def _check_known_fields(dataclass_type, data: dict, path: str, where: str) -> None:
    known = {f.name: f for f in dataclasses.fields(dataclass_type)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"{where}: unknown field {path + key!r}")
        inner = dataclass_field_type(known[key].type)
        if inner is Any or not dataclasses.is_dataclass(inner):
            continue
        if isinstance(value, dict):
            _check_known_fields(inner, value, f"{path}{key}.", where)


def _as_fields(loop) -> dict:
    fields = dataclasses.asdict(loop)
    for name in RUN_SPECIFIC_FIELDS:
        fields.pop(name, None)
    # Tuples become lists so the file reads back the way it was written.
    return json.loads(json.dumps(fields))


def _merge(base: dict, override: dict) -> dict:
    # Mappings merge key by key; a list or scalar replaces the parent's value.
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: str | os.PathLike) -> dict:
    where = str(path)
    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{where}: expected a mapping at the top level")
    unknown = set(doc) - PRESET_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown top-level keys {sorted(unknown)}")
    loop = doc.get("loop")
    if loop is None:
        loop = {}
    if not isinstance(loop, dict):
        raise ValueError(f"{where}: 'loop' must be a mapping")
    if "loop" not in doc and "extends" not in doc:
        raise ValueError(f"{where}: needs a 'loop' mapping or 'extends'")
    aliases = doc.get("aliases") or []
    if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
        raise ValueError(f"{where}: 'aliases' must be a list of names")
    extends = doc.get("extends")
    if extends is not None and not isinstance(extends, str):
        raise ValueError(f"{where}: 'extends' must be a preset name")
    return {"aliases": aliases, "extends": extends, "loop": loop}


def _build(doc: dict, where: str, loop_type: Type[T], parent) -> T:
    _check_known_fields(loop_type, doc["loop"], "", where)
    fields = doc["loop"]
    if parent is not None:
        fields = _merge(_as_fields(parent), fields)
    # loads() carries the migrations and type checks a loop.json goes through.
    return loop_type.loads(json.dumps(fields))


def load_preset(
    path: str | os.PathLike,
    loop_type: Type[T],
    presets: Mapping[str, T] | None = None,
) -> tuple[T, list[str]]:
    """Return the loop in ``path`` and its alias names.

    ``presets`` is where ``extends`` is looked up, normally the training CLI's registry.
    """
    where = str(path)
    doc = _read(path)
    parent = None
    if doc["extends"] is not None:
        if presets is None or doc["extends"] not in presets:
            raise ValueError(
                f"{where}: extends {doc['extends']!r}, which is not a preset"
            )
        parent = presets[doc["extends"]]
    return _build(doc, where, loop_type, parent), doc["aliases"]


def load_preset_dir(directory: str | os.PathLike, loop_type: Type[T]) -> dict[str, T]:
    """Every ``*.yaml`` in ``directory`` under its stem and each of its aliases.

    ``extends`` resolves to another file in the directory, by stem or alias, in any order.
    """
    docs = {
        path.stem: (path, _read(path))
        for path in sorted(Path(directory).glob("*.yaml"))
    }
    stem_of: dict[str, str] = {}
    for stem, (path, doc) in docs.items():
        for name in (stem, *doc["aliases"]):
            if name in stem_of:
                raise ValueError(f"{path}: preset {name!r} is defined twice")
            stem_of[name] = stem

    loops: dict[str, T] = {}

    def resolve(stem: str, chain: tuple[str, ...]) -> T:
        if stem in loops:
            return loops[stem]
        path, doc = docs[stem]
        parent = None
        if doc["extends"] is not None:
            if doc["extends"] not in stem_of:
                raise ValueError(
                    f"{path}: extends {doc['extends']!r}, which is not a preset in {directory}"
                )
            parent_stem = stem_of[doc["extends"]]
            if parent_stem in chain:
                raise ValueError(
                    f"{path}: extends cycle {' -> '.join(chain + (stem, parent_stem))}"
                )
            parent = resolve(parent_stem, chain + (stem,))
        loops[stem] = _build(doc, str(path), loop_type, parent)
        return loops[stem]

    for stem in docs:
        resolve(stem, ())
    return {name: loops[stem] for name, stem in stem_of.items()}


def to_yaml(loop, aliases: Iterable[str] = (), header: str = "") -> str:
    """``loop`` as a preset file with every field written out."""
    doc = {"aliases": list(aliases), "loop": _as_fields(loop)}
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)
    if header:
        body = (
            "".join(f"# {line}".rstrip() + "\n" for line in header.splitlines()) + body
        )
    return body
