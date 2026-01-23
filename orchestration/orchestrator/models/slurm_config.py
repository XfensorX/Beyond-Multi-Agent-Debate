from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel

from orchestration.orchestrator.config import (
    get_slurm_log_filename,
)


class SlurmConfiguration(BaseModel):
    job_name: str
    time: timedelta
    cpus_per_task: int
    memory_GB: int

    @property
    def formatted_time(self) -> str:
        sec = self.time.seconds
        days = self.time.days
        return f"{sec // 3600 + days * 24:02}:{(sec // 60) % 60:02}:{sec % 60:02}"

    def create_batch_file_header(self) -> str:
        header = "\n".join(
            [
                "#!/bin/bash",
                f"#SBATCH --job-name={self.job_name}",
                f"#SBATCH --time={self.formatted_time}",
                f"#SBATCH --cpus-per-task={self.cpus_per_task}",
                f"#SBATCH --mem={self.memory_GB}G",
                f"#SBATCH --output={get_slurm_log_filename(self.job_name)}",  # TODO: Change output file
                "",
                "set -euo pipefail",
            ]
        )

        return header
