from datetime import timedelta

from pydantic import BaseModel


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
                "#SBATCH --output=%x-%j.out",  # TODO: Change output file
                "",
                "set -euo pipefail",
            ]
        )

        return header
