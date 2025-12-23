from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    TimeRemainingColumn,
)

import sys
from typing import Iterable, Iterator, TypeVar


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_tracebacks: bool = True,
) -> None:
    """
    Configure global logging:
    - Pretty console logs via RichHandler
    - Optional rotating log file (machine-friendly debugging)
    """
    if rich_tracebacks:
        install_rich_traceback(show_locals=False)

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Clear any existing handlers (important with Hydra / re-entry)
    root.handlers.clear()

    console_handler = RichHandler(
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )
    console_handler.setLevel(level.upper())
    root.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(level.upper())
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str = "Working",
    logger=None,
    log_interval_s: int = 60,
    progress: Optional[Progress] = None,
) -> Iterator[T]:
    """
    Iterate with:
      - Rich progress when stdout is a TTY
      - logger.info every `log_every_pct` percent (always, even non-TTY)

    Args:
      iterable: any iterable
      total: total items (optional if iterable has __len__)
      desc: label for progress/logging
      logger: logging.Logger (optional)
      log_interval_s: second step for logging (default 60)
      progress: optional Rich Progress instance to reuse/customize
    """
    is_tty = sys.stdout.isatty()

    if total is None and hasattr(iterable, "__len__"):
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    last_log_ts = 0.0  # monotonic seconds

    def maybe_log(done: int, *, force: bool = False):
        nonlocal last_log_ts
        if not logger or not total:
            return

        now = time.monotonic()
        if not force and (now - last_log_ts) < log_interval_s:
            return

        pct = int((done / total) * 100)
        logger.info("%s: %d%% (%d/%d)", desc, pct, min(done, total), total)
        last_log_ts = now

    if logger and total:
        logger.info("%s: 0%% (0/%d)", desc, total)

    if is_tty:
        prog = progress or Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with prog:
            task_id = prog.add_task(desc, total=total or 0)

            done = 0
            for item in iterable:
                yield item
                done += 1

                if total is not None:
                    prog.update(task_id, completed=done)
                else:
                    prog.advance(task_id, 1)

                maybe_log(done)

            # Finalize
            if total is not None:
                prog.update(task_id, completed=total)
            if logger and total:
                maybe_log(total)
                logger.info("%s: done", desc)
    else:
        done = 0
        for item in iterable:
            yield item
            done += 1
            maybe_log(done)

        if logger and total:
            maybe_log(total, force=True)
            logger.info("%s: done", desc)
