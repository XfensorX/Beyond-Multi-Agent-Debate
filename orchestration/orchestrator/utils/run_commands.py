import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from orchestration.orchestrator.utils.types import PathLike

logger = logging.getLogger(__name__)


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

    logger.debug(f"Running: {argv}")

    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=capture,
        check=True,
    )
