from __future__ import annotations

from social_groups.orchestrator.config import random_string_to_job_name_appendix
from social_groups.orchestrator.models.execution_environment import ExecutionLocation
from social_groups.orchestrator.services import (
    SlurmServiceName,
    TgiConfiguration,
    VLLMConfiguration,
    load_config,
)
from social_groups.orchestrator.utils.slurm import SlurmJobInfo
from social_groups.trialrunner.config import Backend, BackendInfoWithEndpoint


def parse_backend_endpoint_from_slurm_job(
    job: SlurmJobInfo, where: ExecutionLocation
) -> BackendInfoWithEndpoint:
    if TgiConfiguration.job_name_is_matching_this_service(job.job_name):
        config: TgiConfiguration = load_config(SlurmServiceName["tgi"], where)
        backend = Backend.L3S_TGI
    elif VLLMConfiguration.job_name_is_matching_this_service(job.job_name):
        config: VLLMConfiguration = load_config(SlurmServiceName["vllm"], where)
        backend = Backend.vLLMExternal
    else:
        raise ValueError(
            "The given jobname does not match any registered InferenceBackend Configurations."
        )

    found = None
    for model_id, info in config.llm_models.items():
        if random_string_to_job_name_appendix(model_id) in job.job_name:
            if found is not None:
                raise ValueError(
                    "Found several fitting models for this job. The configs are not consistent."
                )

            found = BackendInfoWithEndpoint(
                backend=backend,
                model_name=model_id,
                endpoint=f"http://{job.node}:{info.port}",
            )

    if found is not None:
        return found
    else:
        raise ValueError(
            "The Running job could not be mapped to a backend in the config. "
            "The config must have changed since this job is running."
        )
