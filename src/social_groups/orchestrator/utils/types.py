from enum import Enum
from pathlib import Path
from typing import Any, Union

JobId = int
PathLike = Union[str, Path]
ModelId = str


def to_hydra_value(x: Any) -> str:
    """
    Generaterd by ChatGPT.
    Transforms a value to a valid hydra command line format
    (e.g. removes " from the keys of dictionary).
    """
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, str):
        # Always quote strings; escape backslashes and quotes
        s = x.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    if isinstance(x, dict):
        items = ",".join(f"{k}:{to_hydra_value(v)}" for k, v in x.items())
        return "{" + items + "}"
    if isinstance(x, (list, tuple)):
        items = ",".join(to_hydra_value(v) for v in x)
        return "[" + items + "]"
    if isinstance(x, Enum):
        return str(x.value)

    # custom types
    return to_hydra_value(str(x))
