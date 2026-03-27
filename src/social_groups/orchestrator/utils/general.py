import asyncio
import shlex
from functools import wraps
from typing import Any


def run_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def make_exported_variables_block(env: dict[str, Any]) -> str:
    """
    Produces bash export lines for a dictionary.

    - Normal values:   export KEY='value'
    - List values:     KEY=( 'item1' 'item2' 'item3' )

    Everything is passed through str() first.
    """
    lines = []

    for k, v in env.items():
        if isinstance(v, (tuple, set, list)):
            items = [shlex.quote(str(item)) for item in v]
            lines.append(f"export {k}=({' '.join(items)})")
        elif isinstance(v, bool):
            lines.append(f"export {k}={'true' if v else 'false'}")
        else:
            value_str = str(v)
            lines.append(f"export {k}={shlex.quote(value_str)}")

    return "\n".join(lines) + "\n"
