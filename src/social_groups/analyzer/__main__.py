import functools
import os
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

import typer
from rich import print
from typer import Typer

from social_groups.analyzer.algorithm.experiment_reader import build_parquet_files
from social_groups.analyzer.sync_tex import (
    git_commit_and_push,
    sync_exact_with_confirmation,
)
from social_groups.directories import (
    DAGSTER_REPORT_DIR,
    PARQUET_ANALYSIS_DIR,
    TEX_PROJECT_REPORT_DIR,
)
from social_groups.orchestrator.utils.general import run_async

R = TypeVar("R")


# TODO: refactor this
def timer(name: str | None = None):
    """Decorator that prints execution time for sync or async functions."""

    def decorator(func: Callable[..., Coroutine[Any, Any, R] | R]):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            start = time.perf_counter()
            display_name = name or func.__name__

            try:
                print(f"  Executing {display_name}")
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                print(f"✅ {display_name} completed in {elapsed:.2f} seconds")

        return async_wrapper

    return decorator


app = Typer(no_args_is_help=True)


@app.command("parse", help="Parse results to produce parquet files.")
@run_async
@timer("Producing Parquet files ...")
async def produce_parquet():
    os.makedirs(PARQUET_ANALYSIS_DIR, exist_ok=True)
    await build_parquet_files(PARQUET_ANALYSIS_DIR)


@app.command(name="sync-tex", help="Sync DAGster reports → university LaTeX project")
def sync_tex(
    from_dir: Path = typer.Argument(
        DAGSTER_REPORT_DIR, help="Source directory (dir of dagster reports)"
    ),
    to_dir: Path = typer.Argument(
        TEX_PROJECT_REPORT_DIR, help="Target directory (tex.cloud project folder)"
    ),
):
    typer.secho(f"Syncing {from_dir} → {to_dir}", bold=True)
    sync_exact_with_confirmation(from_dir, to_dir)

    typer.secho("\nGit operations in target directory:", bold=True)
    git_commit_and_push(to_dir)


def main():
    app()


if __name__ == "__main__":
    main()
