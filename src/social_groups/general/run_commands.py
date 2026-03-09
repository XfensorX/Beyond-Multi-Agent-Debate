import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from social_groups.orchestrator.utils.types import PathLike

logger = logging.getLogger(__name__)


def run_local(
    args: list[str], check: bool = False, cwd: PathLike | None = None
) -> subprocess.CompletedProcess:
    try:
        logger.debug(f"Running {' '.join(args)}")
        return subprocess.run(args, cwd=cwd, check=check)
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr or "")
        raise RuntimeError(f"Command failed: {' '.join(args)}") from e


def run_ssh(
    login: str,
    remote_cmd: str,
    *,
    capture: bool = True,
    input_text: Optional[str] = None,
    workdir: Optional[PathLike] = None,
    mkdir: bool = True,
) -> subprocess.CompletedProcess:
    """
    Runs: ssh <login> bash -lc "<remote_cmd>"
    Optionally:
      - cd into `workdir`
      - mkdir -p workdir first
      - send `input_text` to stdin
    """
    if workdir is not None:
        wd = shlex.quote(str(Path(workdir)))

        prefix = []
        if mkdir:
            prefix.append(f"mkdir -p {wd}")
        prefix.append(f"cd {wd}")

        remote_cmd = " && ".join(prefix + [remote_cmd])

    argv = ["ssh", login, "bash", "-lc", f'"{remote_cmd}"']

    logger.debug(
        f"Running: {argv} {f'with extra stdin-input: {input_text[:20]}...' if input_text else ''}"
    )

    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=capture,
        check=True,
    )
