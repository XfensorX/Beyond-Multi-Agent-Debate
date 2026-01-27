from pathlib import Path

from pydantic import BaseModel, computed_field

from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from social_groups.orchestrator.services.base import (
    SlurmService,
    register_slurm_service,
)
from social_groups.orchestrator.utils.types import to_hydra_value
from social_groups.trialrunner.config import BackendInfoWithEndpoint


class ExperimentStartingInformation(BaseModel):
    model_backends: list[BackendInfoWithEndpoint]
    experiment_name: str
    phoenix_server_endpoint: str
    phoenix_graphql_endpoint: str
    use_hydra_multirun: bool


@register_slurm_service("experiment")
class ExperimentConfiguration(SlurmService):
    experiment_configuration_dir_inside_project: Path
    _starting_info: ExperimentStartingInformation | None = None

    @computed_field
    @property
    def used_job_name(self) -> str:
        if self._starting_info is None:
            return "experiment___{experiment_name_placeholder}"

        return "experiment" + f"___{self._starting_info.experiment_name}"

    @staticmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool:
        return given_job_name.startswith("experiment___")

    def add_starting_info(self, starting_info: ExperimentStartingInformation):
        self._starting_info = starting_info

    def create_env_dict(self, exec_config: ExecutionLocationConfig) -> dict[str, str]:
        return {}

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        if self._starting_info is None:
            raise ValueError("starting_info has to be given first.")

        model_backends_formatted = to_hydra_value(
            [b.model_dump() for b in self._starting_info.model_backends]
        )

        parts = [
            f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'} &&",
            f"cd {exec_config.project_dir} &&",
            "uv run trial",
        ]

        if self._starting_info.use_hydra_multirun:
            parts.append("-m")

        parts.extend(
            [
                f'--config-dir="{exec_config.project_dir / self.experiment_configuration_dir_inside_project}"',
                f'+experiment="{self._starting_info.experiment_name}"',
                f"'execution.model_backends={model_backends_formatted}'",
                f'execution.phoenix_server_url="{self._starting_info.phoenix_server_endpoint}"',
                f'execution.phoenix_graphql_url="{self._starting_info.phoenix_graphql_endpoint}"',
            ]
        )
        return " ".join(parts)
