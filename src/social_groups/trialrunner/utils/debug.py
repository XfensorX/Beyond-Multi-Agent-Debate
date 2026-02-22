import datetime
import functools
import os
from typing import Any, Callable

from rich import print as rprint
from rich.console import Console
from rich.traceback import Traceback


def with_retry_on_fail(
    max_retries: int = 4,
    log_dir: str = "debug_failures",
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator: retries the function on exception, logs rich HTML tracebacks
    with all local variables to disk on every failure (except possibly the last).

    Usage (temporary debugging):

        @with_retry_on_fail(max_retries=3)
        def call_payment_gateway(...):
            ...

    Files go into log_dir/ with names like:
        call_payment_gateway_ZeroDivisionError_2025-04-12_14-30-22_attempt2.html
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return result

                except retry_on as exc:
                    if attempt == max_retries:
                        # last try → let it bubble up normally
                        raise

                    # ── Save rich traceback + locals + call info ──
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    exc_name = type(exc).__name__

                    fname = f"{func.__name__}_{exc_name}_{timestamp}_attempt{attempt + 1}.html"
                    os.makedirs(log_dir, exist_ok=True)
                    path = os.path.join(log_dir, fname)

                    # Rich traceback magic (locals in every frame!)
                    tb = Traceback.from_exception(
                        type(exc),
                        exc,
                        exc.__traceback__,
                        show_locals=True,
                        max_frames=120,
                        word_wrap=True,
                    )

                    console = Console(record=True, width=160)
                    console.rule(
                        f"[bold red]FAILURE — {func.__name__}  "
                        f"(attempt {attempt + 1}/{max_retries + 1})[/bold red]"
                    )
                    console.print(tb)

                    console.rule("[bold cyan]Call information[/bold cyan]")
                    console.print(
                        f"[bold]Function:[/] {func.__qualname__}  [dim]({func.__module__})[/dim]"
                    )
                    console.print(f"[bold]Args:[/] {args!r}")
                    console.print(f"[bold]Kwargs:[/] {kwargs!r}")
                    console.print(f"[bold]Exception:[/] {exc}")
                    console.print(f"[bold]Time:[/] {timestamp}")

                    html_content = console.export_html(inline_styles=True)

                    with open(path, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    rprint(
                        f"[bold red]Logged failure →[/bold red] "
                        f"[link=file://{os.path.abspath(path)}]{path}[/link]"
                    )
                    rprint("[yellow]Retrying...[/yellow]")

            # unreachable (either returned or raised above)
            raise RuntimeError("Unreachable — bug in retry decorator")

        return wrapper

    return decorator
