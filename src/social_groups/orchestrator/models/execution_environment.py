import logging
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel

from social_groups.directories import ORCHESTRATION_CONFIGS_DIR
from social_groups.general.run_commands import run_ssh
from social_groups.orchestrator.config import (
    EXECUTION_ENVIRONMENT_CONFIG_NAME,
)
from social_groups.orchestrator.utils.types import JobId

logger = logging.getLogger(__name__)


class ExecutionLocation(Enum):
    PASCAL = "pascal"
    NEUMANN = "neumann"
    LOCAL = "local"


class ExecutionLocationConfig(BaseModel):
    where: ExecutionLocation
    ssh_login: str
    project_dir: Path
    slurm_log_dir_inside_project: Path

    def get_logdir(self) -> Path:
        return self.project_dir / self.slurm_log_dir_inside_project

    def submit_sbatch(self, sbatch_file: str) -> JobId:
        # This is important, as the uv run - command should be executed in the project_environment
        sbatch_submit_dir = self.get_logdir()
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


def load_execution_config(where: ExecutionLocation) -> ExecutionLocationConfig:
    file_path = (
        ORCHESTRATION_CONFIGS_DIR / where.value / EXECUTION_ENVIRONMENT_CONFIG_NAME
    )
    with open(file_path) as f:
        return ExecutionLocationConfig.model_validate(yaml.safe_load(f))
