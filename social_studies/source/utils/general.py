import importlib
import pkgutil
import string
from enum import Enum
from types import ModuleType

from pydantic import BaseModel, ConfigDict


def contains_whitespace(s):
    return any((whitespace_char in s) for whitespace_char in string.whitespace)


def import_all_submodules(package) -> list[ModuleType]:
    mods = []
    prefix = package.__name__ + "."
    for m in pkgutil.iter_modules(package.__path__, prefix):
        mods.append(importlib.import_module(m.name))
    return mods


def make_enum(name: str, values: set[str]) -> type[Enum]:
    members = {v: v for v in sorted(values)}
    return Enum(name, members)


class BaseModelWithExtraFields(BaseModel):
    model_config = ConfigDict(extra="allow")


def flatten_dict(
    d,
    parent_key="",
    sep=".",
    map_to_basic_types=False,
    basic_types=(bool, str, bytes, int, float),
):
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(
                flatten_dict(
                    value,
                    new_key,
                    sep,
                    map_to_basic_types,
                    basic_types,
                )
            )
        else:
            if map_to_basic_types and not isinstance(value, basic_types):
                value = str(value)
            items[new_key] = value

    return items
