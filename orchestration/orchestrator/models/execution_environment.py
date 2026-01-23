from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from orchestration.orchestrator.utils.run_commands import run_ssh
from orchestration.orchestrator.utils.types import JobId

logger = logging.getLogger(__name__)


class ExecutionEnvironment(Enum):
    PASCAL = "pascal"
    NEUMANN = "neumann"
    LOCAL = "local"


class ExecutionConfig(BaseModel):
    where: ExecutionEnvironment
    ssh_login: str | None
    project_dir: Path
    slurm_log_dir_inside_project: Path

    def submit_sbatch(self, sbatch_file: str) -> JobId:
        # This is important, as the uv run - command should be executed in the project_environment
        sbatch_submit_dir = self.project_dir / self.slurm_log_dir_inside_project
        logger.info(
            f"Submit sbatch file of {self.__class__.__name__} to {self.where.name}"
        )
        logger.info(f"Output File in: {sbatch_submit_dir}")
        cp = run_ssh(
            self.ssh_login,
            "sbatch --parsable --export=NONE",
            input_text=sbatch_file,
            workdir=sbatch_submit_dir,
        )

        return int(cp.stdout.strip().split()[-1])  # ~~ 'Submitted batch job XXXX'
