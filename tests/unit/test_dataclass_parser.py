# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from healda.utils.parsing import a, parse_args, parse_dict, Help
from typing import Optional, Any
import pytest
from enum import auto, Enum
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.01
    epochs: int = 10
    optional: Optional[bool] = False


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    model_name: str = "default_model"
    opt: a[str, Help("An example option.")] = "a"


@pytest.mark.parametrize("convert_underscore_to_hyphen", [True, False])
def test_parse_args(convert_underscore_to_hyphen):
    # Usage example
    sep = "-" if convert_underscore_to_hyphen else "_"
    args = [
        f"--model.learning{sep}rate",
        "0.1",
        "--model.epochs",
        "20",
        f"--model{sep}name",
        "my_model",
    ]

    with pytest.raises(SystemExit):
        parse_args(
            Config,
            [f"--model.learning{sep}rate", "not a num"],
            convert_underscore_to_hyphen=convert_underscore_to_hyphen,
        )

    expected = Config(
        model=ModelConfig(learning_rate=0.1, epochs=20), model_name="my_model"
    )
    assert (
        parse_args(
            Config, args, convert_underscore_to_hyphen=convert_underscore_to_hyphen
        )
        == expected
    )


def test_parse_dict():
    obj = {"model_name": "hello", "model": {"learning_rate": 0.1}}
    expected = Config(model=ModelConfig(learning_rate=0.1), model_name="hello")
    assert parse_dict(Config, obj) == expected

    with pytest.raises(ValueError):
        parse_dict(Config, {"model_name": 1})


def test_parse_args_optional():
    @dataclass
    class Config:
        a: Optional[int] = None

    c = parse_args(Config, ["--a", "1"])
    assert c == Config(1)


def test_parse_args_union():
    @dataclass
    class Config:
        a: int | None = None

    c = parse_args(Config, ["--a", "1"])
    assert c == Config(1)


def test_parse_args_any():
    @dataclass
    class Config:
        a: Any = None

    c = parse_args(Config, ["--a", "1"])
    assert c == Config("1")


def test_parse_args_bool_default_false():
    @dataclass
    class Config:
        a: bool = False

    c = parse_args(Config, ["--a"])
    assert c == Config(True)


def test_parse_args_bool_default_true():
    @dataclass
    class Config:
        a: bool = True

    c = parse_args(Config, ["--no-a"])
    assert c == Config(False)


def test_parse_args_bool_default_true_nested():
    @dataclass
    class Sub:
        a: bool = False

    @dataclass
    class Config:
        sub: Sub = field(default_factory=lambda: Sub(a=True))

    c = parse_args(Config, ["--sub.no_a"], convert_underscore_to_hyphen=False)
    assert c == Config(Sub(False))

    c = parse_args(Config, ["--sub.no-a"])
    assert c == Config(Sub(False))


def test_enum():
    class Options(Enum):
        a = auto()
        b = auto()

    @dataclass
    class CLI:
        opt: Options = Options.a

    c = parse_args(CLI, ["--opt", "b"])
    c.opt == Options.b

    c = parse_args(CLI, [])
    c.opt == Options.a


def test_parse_args_double_nested():
    @dataclass(eq=True)
    class SubSub:
        a: int = 1

    @dataclass(eq=True)
    class Sub:
        sub: SubSub = field(default_factory=SubSub)

    @dataclass(eq=True)
    class Config:
        sub: Sub = field(default_factory=Sub)

    c = parse_args(Config, ["--sub.sub.a", "1"], convert_underscore_to_hyphen=False)
    assert c == Config()


def test_parse_args_with_list_generic_type():
    """Ensure list[...] | None fields are compatible with parse_args strict checks."""

    @dataclass
    class Config:
        values: list[int] | None = None

    # This should not raise TypeError from isinstance() on a parameterized generic.
    cfg = parse_args(Config, args=[])
    assert cfg.values is None


def test_list_fields_collect_multiple_values():
    @dataclass
    class Config:
        report_types: list[int] = field(default_factory=list)
        sensors: list[str] = field(default_factory=list)

    assert parse_args(Config, args=[]) == Config([], [])
    cfg = parse_args(Config, ["--report-types", "240", "260", "--sensors", "iasi-pca"])
    assert cfg == Config([240, 260], ["iasi-pca"])
    # A bare flag means "empty", not "one value".
    assert parse_args(Config, ["--report-types"]).report_types == []
    # A single value is still a list, not a scalar.
    assert parse_args(Config, ["--sensors", "airs-pca"]).sensors == ["airs-pca"]


def test_list_field_element_type_is_enforced():
    @dataclass
    class Config:
        report_types: list[int] = field(default_factory=list)

    with pytest.raises(SystemExit):
        parse_args(Config, ["--report-types", "not-an-int"])


def test_list_field_default_is_not_shared_between_parses():
    @dataclass
    class Config:
        sensors: list[str] = field(default_factory=list)

    first = parse_args(Config, args=[])
    first.sensors.append("mutated")
    assert parse_args(Config, args=[]).sensors == []


def test_list_field_inside_nested_dataclass():
    @dataclass
    class Inner:
        sensors: list[str] = field(default_factory=list)

    @dataclass
    class Outer:
        inner: Inner = field(default_factory=Inner)

    cfg = parse_args(Outer, ["--inner.sensors", "a", "b"])
    assert cfg.inner.sensors == ["a", "b"]


def test_optional_bool_keeps_unset_distinct_from_false():
    """None means "leave the caller's own setting alone", so it must survive parsing."""

    @dataclass
    class Config:
        restore_fill: bool | None = None

    assert parse_args(Config, args=[]).restore_fill is None
    assert parse_args(Config, ["--restore-fill"]).restore_fill is True
    assert parse_args(Config, ["--no-restore-fill"]).restore_fill is False


def test_plain_bool_keeps_its_original_flag_spelling():
    """The two non-optional bool paths predate list/optional-bool support."""

    @dataclass
    class Config:
        off_by_default: bool = False
        on_by_default: bool = True

    assert parse_args(Config, args=[]) == Config(False, True)
    assert parse_args(Config, ["--off-by-default"]).off_by_default is True
    assert parse_args(Config, ["--no-on-by-default"]).on_by_default is False
    # A plain bool gains no opposite flag.
    with pytest.raises(SystemExit):
        parse_args(Config, ["--no-off-by-default"])


def test_optional_bool_defaulting_true_still_accepts_both_flags():
    @dataclass
    class Config:
        flag: bool | None = True

    assert parse_args(Config, args=[]).flag is True
    assert parse_args(Config, ["--no-flag"]).flag is False
    assert parse_args(Config, ["--flag"]).flag is True


def test_absent_optional_nested_dataclass_stays_none():
    """It used to materialise unconditionally, so `None` was unreachable from the CLI."""

    @dataclass
    class Inner:
        k: int = 4
        depth: int = 128

    @dataclass
    class Config:
        inner: Inner | None = None
        name: str = ""

    assert parse_args(Config, args=[]).inner is None
    assert parse_args(Config, ["--name", "x"]).inner is None
    # Naming any one field opts in; the rest keep Inner's own defaults.
    assert parse_args(Config, ["--inner.k", "7"]).inner == Inner(k=7, depth=128)
    assert parse_args(Config, ["--inner.depth", "64"]).inner == Inner(k=4, depth=64)


def test_present_optional_nested_dataclass_is_still_built():
    @dataclass
    class Inner:
        k: int = 4

    @dataclass
    class Config:
        inner: Inner | None = field(default_factory=lambda: Inner(k=9))

    assert parse_args(Config, args=[]).inner == Inner(k=9)
    assert parse_args(Config, ["--inner.k", "1"]).inner == Inner(k=1)


def test_sibling_prefix_does_not_materialise_an_absent_optional():
    """`inner2` starts with `inner`, so a bare startswith answered for the wrong field."""

    @dataclass
    class Inner:
        k: int = 4

    @dataclass
    class Config:
        inner: Inner | None = None
        inner2: Inner | None = None

    cfg = parse_args(Config, ["--inner2.k", "9"])
    assert cfg.inner is None
    assert cfg.inner2 == Inner(k=9)
