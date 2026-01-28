from __future__ import annotations

from social_groups.orchestrator.utils.types import JobId

YAML_ENDING = ".yaml"
EXECUTION_ENVIRONMENT_CONFIG_NAME = "general" + YAML_ENDING


def get_slurm_log_filename(job_name: str, jobId: JobId | None = None) -> str:
    if jobId is None:
        return f"{job_name}-%j.out"
    else:
        return f"{job_name}-{jobId}.out"


def random_string_to_job_name_appendix(model_id: str) -> str:
    return model_id.replace("/", "__")
