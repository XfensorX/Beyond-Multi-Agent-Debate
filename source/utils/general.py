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


def unflatten_dict(d, sep="."):
    """
    Recursively unflattens a dict that may contain a mix of:
      - dotted keys (flattened)
      - nested dicts (already unflattened)
      - nested dicts that themselves contain dotted keys

    Example:
      {
        "a.b": 1,
        "a": {"c.d": 2},
        "x": {"y": {"z.w": 3}}
      }
    becomes:
      {
        "a": {"b": 1, "c": {"d": 2}},
        "x": {"y": {"z": {"w": 3}}}
      }
    """
    if not isinstance(d, dict):
        return d

    result = {}

    def merge_dicts(dst, src):
        """Deep-merge src into dst (dicts only)."""
        for k, v in src.items():
            if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                merge_dicts(dst[k], v)
            else:
                dst[k] = v

    def set_path(root, parts, value):
        cur = root
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]

        last = parts[-1]
        if isinstance(cur.get(last), dict) and isinstance(value, dict):
            merge_dicts(cur[last], value)
        else:
            cur[last] = value

    for key, value in d.items():
        # Recurse first so mixed dicts inside values get unflattened too
        if isinstance(value, dict):
            value = unflatten_dict(value, sep=sep)

        parts = key.split(sep) if isinstance(key, str) and sep in key else [key]
        set_path(result, parts, value)

    return result
