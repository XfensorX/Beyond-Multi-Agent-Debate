from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

from orchestration.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from orchestration.orchestrator.models.slurm_config import SlurmConfiguration
from orchestration.orchestrator.services.base import (
    SlurmService,
    register_slurm_service,
)


class ModelConfiguration(BaseModel):
    model_id: str
    max_batch_prefill_tokens: int
    max_total_tokens: int
    max_input_tokens: int


@register_slurm_service("tgi")
class TgiConfiguration(SlurmService):
    port: int
    sif_path_in_project: Path
    data_bind_directory: Path
    huggingface_cache_directory: Path

    llm_model: ModelConfiguration

    @field_validator("slurm_config")
    @classmethod
    def require_partition_specification(
        cls, v: SlurmConfiguration
    ) -> SlurmConfiguration:
        if v.partition is None:
            raise ValueError("Please specify partition parameter for TGI.")
        return v

    @model_validator(mode="after")
    def prepend_model_name_to_job_name(self):
        self.slurm_config.job_name += f"___{self.llm_model.model_id.replace('/', '__')}"
        return self

    def create_env_dict(self, exec_config: ExecutionLocationConfig) -> dict[str, str]:
        return {}

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        use_flash_attention = False

        match self.slurm_config.partition:
            case "ampere":
                use_flash_attention = True

        parts = [
            "exec apptainer run",
            "--nv",
            f'-B "{self.data_bind_directory}:/data"',
            f'-B "{self.huggingface_cache_directory}:/root/.cache/huggingface"',
            f"--env USE_FLASH_ATTENTION={'TRUE' if use_flash_attention else 'FALSE'}",
            "--env RUST_LOG_STYLE=never",
            f"{exec_config.project_dir / self.sif_path_in_project}",
            f'--model-id "{self.llm_model.model_id}"',
            '--hostname "0.0.0.0"',
            f'--port "{self.port}"',
            f'--max-batch-prefill-tokens "{self.llm_model.max_batch_prefill_tokens}"',
            f'--max-total-tokens "{self.llm_model.max_total_tokens}"',
            f'--max-input-tokens "{self.llm_model.max_input_tokens}"',
        ]

        return " ".join(parts)
