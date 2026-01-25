from pathlib import Path

from pydantic import computed_field

from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from social_groups.orchestrator.services.base import register_slurm_service
from social_groups.orchestrator.services.base_inference import (
    BaseInferenceService,
    model_id_to_job_name_appendix,
)


@register_slurm_service("tgi")
class TgiConfiguration(BaseInferenceService):
    sif_path_in_project: Path
    data_bind_directory: Path
    huggingface_cache_directory: Path

    @computed_field
    @property
    def used_job_name(self) -> str:
        if self._chosen_model_id is None:
            return "tgi___{model_id_placeholder}"

        return "tgi" + f"___{model_id_to_job_name_appendix(self._chosen_model_id)}"

    @staticmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool:
        return given_job_name.startswith("tgi___")

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
        ]
        if used_model.max_batch_prefill_tokens is not None:
            parts.append(
                f'--max-batch-prefill-tokens "{used_model.max_batch_prefill_tokens}"'
            )
        if used_model.max_total_tokens is not None:
            parts.append(f'--max-total-tokens "{used_model.max_total_tokens}"')
        if used_model.max_input_tokens is not None:
            parts.append(f'--max-input-tokens "{used_model.max_input_tokens}"')

        return " ".join(parts)
