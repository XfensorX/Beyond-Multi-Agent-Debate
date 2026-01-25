from pathlib import Path

from pydantic import BaseModel, computed_field, field_validator

from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from social_groups.orchestrator.models.slurm_config import SlurmConfiguration
from social_groups.orchestrator.services.base import (
    SlurmService,
    register_slurm_service,
)
from social_groups.orchestrator.utils.types import ModelId


class ModelConfiguration(BaseModel):
    port: int

    max_batch_prefill_tokens: int
    max_total_tokens: int
    max_input_tokens: int


def model_id_to_job_name_appendix(model_id: str) -> str:
    return model_id.replace("/", "__")


@register_slurm_service("tgi")
class TgiConfiguration(SlurmService):
    sif_path_in_project: Path
    data_bind_directory: Path
    huggingface_cache_directory: Path
    llm_models: dict[ModelId, ModelConfiguration]
    _chosen_model_id: ModelId | None = None  # The model to actually run

    @computed_field
    @property
    def used_job_name(self) -> str:
        if self._chosen_model_id is None:
            return "tgi___{model_id_placeholder}"

        return "tgi" + f"___{model_id_to_job_name_appendix(self._chosen_model_id)}"

    @staticmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool:
        return given_job_name.startswith("tgi___")

    @field_validator("slurm_config")
    @classmethod
    def require_partition_specification(
        cls, v: SlurmConfiguration
    ) -> SlurmConfiguration:
        if v.partition is None:
            raise ValueError("Please specify partition parameter for TGI.")
        return v

    def check_model_config_exists(self, to_test: list[ModelId]):
        all_model_ids = set(self.llm_models.keys())
        not_available = set(to_test) - all_model_ids
        if not_available:
            raise ValueError(
                "\nInvalid Model Ids: \n - "
                + "\n - ".join(sorted(not_available))
                + "\n\n>> Please add them to the yaml-config first."
                "\n\nAvailable models: \n - "
                + "\n - ".join(sorted(self.llm_models.keys()))
            )

    def set_used_model(self, modelid: ModelId):
        self.check_model_config_exists([modelid])
        self._chosen_model_id = modelid

    def create_env_dict(self, exec_config: ExecutionLocationConfig) -> dict[str, str]:
        return {}

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        use_flash_attention = False

        match self.slurm_config.partition:
            case "ampere":
                use_flash_attention = True

        if self._chosen_model_id is None:
            raise ValueError(
                "Have to set the used model before generation of slurm job file."
            )

        used_model = self.llm_models[self._chosen_model_id]
        parts = [
            "exec apptainer run",
            "--nv",
            f'-B "{self.data_bind_directory}:/data"',
            f'-B "{self.huggingface_cache_directory}:/root/.cache/huggingface"',
            f"--env USE_FLASH_ATTENTION={'TRUE' if use_flash_attention else 'FALSE'}",
            "--env RUST_LOG_STYLE=never",
            f"{exec_config.project_dir / self.sif_path_in_project}",
            f'--model-id "{self._chosen_model_id}"',
            '--hostname "0.0.0.0"',
            f'--port "{used_model.port}"',
            f'--max-batch-prefill-tokens "{used_model.max_batch_prefill_tokens}"',
            f'--max-total-tokens "{used_model.max_total_tokens}"',
            f'--max-input-tokens "{used_model.max_input_tokens}"',
        ]

        return " ".join(parts)
