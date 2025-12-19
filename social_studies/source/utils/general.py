import string
import importlib
import pkgutil
from enum import Enum
from types import ModuleType


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
