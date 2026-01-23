import asyncio
import shlex
from functools import wraps


def run_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def make_exported_variables_block(env: dict[str, str]) -> str:
    """
    Produces lines like:
      export KEY='VALUE'
    safely quoted for bash.
    """
    lines = []
    for k, v in env.items():
        # Quote the value safely for bash
        lines.append(f"export {k}={shlex.quote(str(v))}")
    return "\n".join(lines) + "\n"
