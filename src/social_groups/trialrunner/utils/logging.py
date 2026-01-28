from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar

from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.traceback import install as install_rich_traceback

from social_groups.directories import LOG_FILE_NAME
from social_groups.trialrunner.config import LOG_LEVEL

DEFAULT_CONSOLE_WIDTH = 240
log = logging.getLogger("setup-logging")


def is_rich_renderable(x) -> bool:
    return hasattr(x, "__rich_console__") or hasattr(x, "__rich__")


class RichRenderableHandler(RichHandler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = record.msg
        if (
            isinstance(msg, ConsoleRenderable)
            or hasattr(msg, "__rich_console__")
            or hasattr(msg, "__rich__")
        ):
            try:
                traceback = None
                log_renderable = self.render(
                    record=record, traceback=traceback, message_renderable=msg
                )
                self.console.print(log_renderable)
            except Exception:
                self.handleError(record)
            return
        # Fallback: normal RichHandler behavior for strings, numbers, etc.
        super().emit(record)


class RichToTextFormatter(logging.Formatter):
    """
    Turns Rich renderables into plain text for file logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        if is_rich_renderable(record.msg):
            console = Console(record=True, width=DEFAULT_CONSOLE_WIDTH, quiet=True)
            console.print(record.msg)
            rendered = console.export_text(clear=True)
            original_msg = record.msg
            try:
                record.msg = rendered.rstrip("\n")
                record.args = ()
                return super().format(record)
            finally:
                record.msg = original_msg
        return super().format(record)


def setup_logging(
    output_directory: Path,
    level: str = LOG_LEVEL,
    rich_tracebacks: bool = True,
) -> None:
    """
    Configure global logging:
    - Pretty console logs via RichHandler
    - Optional rotating log file (machine-friendly debugging)
    """

    log_file_path = output_directory / LOG_FILE_NAME
    if rich_tracebacks:
        install_rich_traceback(show_locals=False)

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Clear any existing handlers (important with Hydra / re-entry)
    root.handlers.clear()

    console_handler = RichRenderableHandler(
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )
    console_handler.setLevel(level.upper())
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console_handler)

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_console = Console(file=open(log_file_path, "wt"), width=DEFAULT_CONSOLE_WIDTH)

    # Configure the RichHandler to use this file-based console
    file_handler = RichRenderableHandler(
        console=file_console,
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )
    file_handler.setLevel(level.upper())
    # file_handler.setFormatter(
    #     RichToTextFormatter(
    #         fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #     )
    # )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    log.info("Logging file: %s", log_file_path)
    log.info("Log Level: %s", level)


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
