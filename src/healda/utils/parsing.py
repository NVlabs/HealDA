# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A small library for parsing cli directly to dataclasses

Example:

"""

from dataclasses import dataclass, fields, MISSING, is_dataclass
from types import UnionType
import enum
from typing import (
    Any,
    Type,
    TypeVar,
    Union,
    Annotated,
    get_origin,
)
import argparse


__all__ = ["Help", "parse_args", "parse_dict", "a"]

T = TypeVar("T")

a = Annotated


@dataclass(frozen=True)
class Help:
    """When used with annotated types this will add the help to argparse"""

    message: str


def _get_type_and_meta(t):
    """return the type and metadta of a potentially annotated type"""
    if get_origin(t) is Annotated:
        meta = t.__metadata__
        t = t.__origin__
    else:
        meta = []

    return _is_optional(t), _handle_optional(t), meta


def _is_optional(t):
    return get_origin(t) in [Union, UnionType]


def _handle_optional(t):
    """this returns the specified user type when wrapped in optional or a union type object

    Exmaples:

    _handle_optional(str | None) == str
    _handle_optional(Optional[str]) == str
    _handle_optional(Union[str, None]) == str

    """
    if get_origin(t) in [Union, UnionType]:
        types = [tt for tt in t.__args__ if tt is not type(None)]
        if len(types) > 1:
            raise ValueError(f"Union types not supported: {t}.")
        return types[0]
    else:
        return t


def is_enum(T):
    return isinstance(T, Type) and issubclass(T, enum.Enum)


def parse_args(
    opts: Type[T],
    args: list[str] | None = None,
    strict: bool = True,
    convert_underscore_to_hyphen: bool = True,
) -> T:
    """Parse a list of command line arguments into a dataclass

    Args:
        opts: the dataclass specification of the arguments
        args: the list of string command line arguments.
            If not provided, this is read from sys.argv.
        strict: if true, then check the types at runtime

    Returns:
        an instance of the `opts` dataclass

    """
    parser = argparse.ArgumentParser()

    def add_arguments(
        parser: argparse.ArgumentParser, dataclass_type, prefix="", absent=False
    ):
        for field in fields(dataclass_type):
            help_str = ""
            optional, T, meta = _get_type_and_meta(field.type)
            this_parser = parser
            for item in meta:
                if isinstance(item, Help):
                    help_str = item.message

            # Construct argument name with prefix if provided
            # Check for default values
            if isinstance(dataclass_type, type):
                default = (
                    field.default
                    if field.default is not MISSING
                    else (
                        field.default_factory()
                        if field.default_factory is not MISSING
                        else MISSING
                    )
                )
            else:
                default = getattr(dataclass_type, field.name)
            if absent:
                default = None

            def _get_arg_name(field_name: str, prefix: str, required):
                if convert_underscore_to_hyphen:
                    field_name = field_name.replace("_", "-")

                flag = "" if required and not prefix else "--"
                if prefix:
                    return f"{flag}{prefix}{field_name}"
                else:
                    return f"{flag}{field_name}"

            if is_dataclass(T):
                # An absent optional stays absent: default its fields to None so
                # construct_dataclass can tell "not asked for" from "asked for with the
                # field defaults", instead of always materialising the nested value.
                add_arguments(
                    parser,
                    default or T,
                    prefix=f"{prefix}{field.name}.",
                    absent=optional and default is None,
                )
            else:
                arg_name = _get_arg_name(
                    field.name, prefix, required=default is MISSING
                )
                if get_origin(T) is list:
                    (item_type,) = T.__args__
                    help_str += f" [{item_type.__name__} list, default: {default}]"
                    this_parser.add_argument(
                        arg_name,
                        nargs="*",
                        type=item_type,
                        default=default,
                        help=help_str,
                    )
                elif T is bool and optional:
                    # Tri-state: --flag, --no-flag, or unset to leave the default alone.
                    this_parser.add_argument(
                        arg_name,
                        action=argparse.BooleanOptionalAction,
                        default=default,
                        help=help_str,
                    )
                elif T is bool:
                    if default:
                        sep = "-" if convert_underscore_to_hyphen else "_"
                        arg_name = _get_arg_name(
                            field.name, prefix + f"no{sep}", required=default is MISSING
                        )
                        dest = f"{prefix}{field.name}" if prefix else f"{field.name}"
                        this_parser.add_argument(
                            arg_name,
                            action="store_false",
                            dest=dest,
                            help=help_str,
                        )
                    else:
                        this_parser.add_argument(
                            arg_name,
                            action="store_true",
                            default=None if absent else False,
                            help=help_str,
                        )
                elif is_enum(T):
                    this_parser.add_argument(
                        arg_name, default=default.name, choices=[x.name for x in T]
                    )
                elif T is Any:
                    this_parser.add_argument(
                        arg_name,
                        default=default,
                        help=help_str,
                    )
                else:
                    help_str += f" [{T.__name__}, default: {default}]"
                    this_parser.add_argument(
                        arg_name,
                        type=T,
                        default=default,
                        help=help_str,
                    )

    add_arguments(parser, opts)
    parsed_args = parser.parse_args(args)

    def construct_dataclass(dataclass_type, parsed_data, prefix="", partial=False):
        init_kwargs = {}
        for field in fields(dataclass_type):
            key = f"{prefix}{field.name}"
            optional, T, _ = _get_type_and_meta(field.type)
            if partial and getattr(parsed_data, key, None) is None:
                # Under an absent optional, only the fields actually passed were given a
                # value; the rest fall back to the nested dataclass's own defaults.
                continue
            if is_dataclass(T):
                if optional and not _contains_object(f"{key}.", parsed_data):
                    value = None
                else:
                    # Recursively build nested dataclass
                    value = construct_dataclass(
                        T,
                        parsed_data,
                        prefix=f"{prefix}{field.name}.",
                        partial=optional,
                    )
            else:
                # Use the argument value
                value = getattr(parsed_data, key)

            if is_enum(T):
                (value,) = [it for it in T if it.name == value]

            if strict and T is not Any:
                # Normalize parameterized generics (e.g. list[int]) to their origin
                # type for isinstance checks, since Python disallows using
                # parameterized generics directly in isinstance().
                origin = get_origin(T)
                check_type = origin or T
                allowed_types = (check_type, type(None)) if optional else (check_type,)
                if not isinstance(value, allowed_types):
                    raise ValueError(f"{value} is not a {T}")

            init_kwargs[field.name] = value
        return dataclass_type(**init_kwargs)

    return construct_dataclass(opts, parsed_args)


def _contains_object(prefix, data):
    """Was any field of this nested dataclass actually supplied?

    ``prefix`` ends in the separating dot, or a sibling named ``inner2`` would answer for
    ``inner``. An absent optional registers its fields with default None, so a key
    existing is not enough -- it has to hold a value.
    """
    for key, value in vars(data).items():
        if key.startswith(prefix) and value is not None:
            return True
    return False


def parse_dict(opts: Type[T], obj: dict, strict: bool = True) -> T:
    """Parse an untyped nested dictionary ``obj``` into a dataclass ``opts```

    If ``strict`` is true, then the types of the obj will be validated at runtime
    """

    def construct_dataclass(dataclass_type, data):
        init_kwargs = {}
        for field in fields(dataclass_type):
            field_name = field.name
            if field_name in data:
                value = data[field_name]
                is_optional, T, _ = _get_type_and_meta(field.type)
                if is_dataclass(T) and isinstance(value, dict):
                    # Recursively construct nested dataclass
                    value = construct_dataclass(T, value)
                if strict and T is not Any:
                    # Allow None for optional types
                    if not (is_optional and value is None):
                        origin = get_origin(T)
                        check_type = origin or T
                        # JSON has no tuple type, so a tuple-typed field
                        # round-trips through dumps/loads as a list. Coerce it
                        # back so both the check and the dataclass get a tuple.
                        if check_type is tuple and isinstance(value, list):
                            value = tuple(value)
                        if not isinstance(value, check_type):
                            raise ValueError(field, "is not a", T)
                init_kwargs[field_name] = value
            elif field.default is not MISSING:
                init_kwargs[field_name] = field.default
            elif field.default_factory is not MISSING:
                init_kwargs[field_name] = field.default_factory()
            else:
                raise ValueError(
                    f"Field '{field_name}' is required but not provided in data."
                )

        return dataclass_type(**init_kwargs)

    return construct_dataclass(opts, obj)
