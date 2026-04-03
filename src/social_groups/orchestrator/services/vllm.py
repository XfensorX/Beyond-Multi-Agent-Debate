import re

from pydantic import computed_field

from social_groups.orchestrator.config import random_string_to_job_name_appendix
from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from social_groups.orchestrator.services.base import register_slurm_service
from social_groups.orchestrator.services.base_inference import (
    BaseInferenceService,
)


@register_slurm_service("vllm")
class VLLMConfiguration(BaseInferenceService):
    @computed_field
    @property
    def used_job_name(self) -> str:
        if self._chosen_model_id is None:
            return "vllm___{model_id_placeholder}"

        return (
            "vllm" + f"___{random_string_to_job_name_appendix(self._chosen_model_id)}"
        )

    @staticmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool:
        return given_job_name.startswith("vllm___")

    def create_env_dict(self, exec_config: ExecutionLocationConfig) -> dict[str, str]:
        return {}

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        if self._chosen_model_id is None:
            raise ValueError(
                "Have to set the used model before generation of slurm job file."
            )
        used_model = self.llm_models[self._chosen_model_id]

        number_gpus = int(
            re.fullmatch(r"gpu:(\d+)", self.slurm_config.gres).groups()[0]
        )

        if used_model.max_batch_prefill_tokens or used_model.max_input_tokens:
            raise NotImplementedError()

        return (
            f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'} && "
            f"uv run vllm serve {self._chosen_model_id} "
            f"--host=0.0.0.0 "
            f"--port={used_model.port} "
            + (
                f"--enable-auto-tool-choice --tool-call-parser {used_model.tool_call_parser} "
                if used_model.tool_call_parser
                else ""
            )
            + (
                f"--max-model-len={used_model.max_total_tokens} "
                if used_model.max_total_tokens
                else ""
            )
            + (f"--data-parallel-size={number_gpus} " if number_gpus != 1 else "")
            + (f"--api-server-count={number_gpus} " if number_gpus != 1 else "")
            + (
                f"--tokenizer_mode={used_model.tokenizer_mode} "
                if used_model.tokenizer_mode
                else ""
            )
            + (
                f"--config_format={used_model.config_format} "
                if used_model.config_format
                else ""
            )
            + (
                f"--load_format={used_model.load_format} "
                if used_model.load_format
                else ""
            )
            # f"--quantization=" # For the future
        )
