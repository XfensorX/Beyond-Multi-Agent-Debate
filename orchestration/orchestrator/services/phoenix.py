from orchestration.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from orchestration.orchestrator.services.base import (
    SlurmService,
    register_slurm_service,
)


@register_slurm_service("phoenix")
class PhoenixConfiguration(SlurmService):
    port: int

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
        return f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'} && uv sync && uv run phoenix serve"
