from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from social_groups.orchestrator.services.base import (
    SlurmService,
    register_slurm_service,
)


@register_slurm_service("phoenix")
class PhoenixConfiguration(SlurmService):
    port: int

    @property
    def used_job_name(self) -> str:
        return "phoenix"

    @staticmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool:
        return given_job_name == "phoenix"

    def create_env_dict(self, exec_config: ExecutionLocationConfig) -> dict[str, str]:
        working_dir = (  # TODO: make "results" a global variable
            exec_config.project_dir / "results" / exec_config.where.value / "sqlite"
        )
        return {
            "PHOENIX_PORT": str(self.port),
            "PHOENIX_ALLOW_EXTERNAL_RESOURCES": "false",
            "PHOENIX_WORKING_DIR": str(working_dir),
            "PHOENIX_SQL_DATABASE_URL": f"sqlite:///{str(working_dir)}/phoenix.db",
        }

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        return f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'} && uv run phoenix serve"
