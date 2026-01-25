from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel

from social_groups.orchestrator.config import get_slurm_log_filename


class SlurmConfiguration(BaseModel):
    time: timedelta
    cpus_per_task: int
    memory_GB: int
    gres: str | None = None
    partition: str | None = None

    @property
    def formatted_time(self) -> str:
        sec = self.time.seconds
        days = self.time.days
        return f"{sec // 3600 + days * 24:02}:{(sec // 60) % 60:02}:{sec % 60:02}"

    def create_batch_file_header(self, job_name: str) -> str:
        parts = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --time={self.formatted_time}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={self.memory_GB}G",
            f"#SBATCH --output={get_slurm_log_filename(job_name)}",  # TODO: Change output file
        ]

        if self.partition:
            parts.append(f"#SBATCH --partition={self.partition}")

        if self.gres:
            parts.append(f"#SBATCH --gres={self.gres}")

        parts.append("")
        parts.append("set -euo pipefail")

        header = "\n".join(parts)

        return header
