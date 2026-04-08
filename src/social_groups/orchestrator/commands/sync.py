import shlex
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from social_groups.general.run_commands import run_ssh


def get_remote_run_dirs(ssh_login: str, host_base_dir: str) -> list[Path]:
    """
    List all run directories in format: node/experiment/run
    Returns relative paths as list of Path objects
    """
    cmd = f'find "{shlex.quote(host_base_dir)}" -mindepth 3 -maxdepth 3 -type d'
    result = run_ssh(ssh_login, remote_cmd=cmd, capture=True, mkdir=False)

    lines = result.stdout.strip().splitlines()
    remote_base = Path(host_base_dir)

    relative_paths = []
    for line in lines:
        full = Path(line.strip())
        if not full.is_relative_to(remote_base):
            continue
        rel = full.relative_to(remote_base)
        # We expect: node/experiment/yyyy-mm-dd_...
        if len(rel.parts) == 3:
            relative_paths.append(rel)
        else:
            raise NotImplementedError()

    return sorted(relative_paths)


def warn_about_existing_conflict(local_target: Path, console: Console):
    """Big visible warning when target already exists"""
    msg = Text.assemble(
        ("⚠️  ", "bold red"),
        ("TARGET DIRECTORY ALREADY EXISTS!\n", "bold red"),
        f"\n{local_target}\n\n",
        "Please check manually before proceeding.\n",
        style="white on red",
    )

    console.print(
        Panel(
            msg,
            title="CONFLICT DETECTED",
            border_style="bold red",
            padding=(1, 2),
            expand=False,
        )
    )
