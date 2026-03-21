from pathlib import Path

from pydantic import BaseModel

from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocationConfig,
)
from social_groups.orchestrator.services.base import (
    SlurmService,
    register_slurm_service,
)


class PostgresConfig(BaseModel):
    shared_buffers: str  # e.g. 4GB
    work_mem: str  # e.g. 128MB
    port: int

    sif_location_inside_project: Path  # e.g. assets/orchestration/postgres.sif


@register_slurm_service("phoenix-pg")
class PhoenixWithPostgresConfiguration(SlurmService):
    port: int
    postgres: PostgresConfig

    @property
    def used_job_name(self) -> str:
        return "phoenix-pg"

    @staticmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool:
        return given_job_name == "phoenix-pg"

    def create_env_dict(self, exec_config: ExecutionLocationConfig) -> dict[str, str]:
        working_dir = (  # This cannot use the global variable, because it is used in orchestrator
            exec_config.project_dir / "results" / exec_config.where.value / "postgres"
        )

        postgres_dir = working_dir / "pgdata"
        postgres_run_dir = working_dir / "pgrun"

        return {
            "PGDATA": str(postgres_dir),
            "PGRUN": str(postgres_run_dir),
            "PGPORT": self.postgres.port,
            "POSTGRES_PASSWORD": "this-must-not-be-secure",
            ##
            "PHOENIX_PORT": str(self.port),
            "PHOENIX_ALLOW_EXTERNAL_RESOURCES": "false",
            "PHOENIX_WORKING_DIR": str(working_dir),
            "PHOENIX_TELEMETRY_ENABLED": "false",
            "PHOENIX_SQL_DATABASE_URL": "postgresql://postgres:${POSTGRES_PASSWORD}@localhost:${PGPORT}/postgres?sslmode=disable",
        }

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        pg_sif = exec_config.project_dir / self.postgres.sif_location_inside_project

        prepare_pg_data = (
            'mkdir -p "$PGDATA" && mkdir -p "PGRUN" && chmod 700 "$PGDATA"'
        )
        start_postgres = f"""
        apptainer instance start \
        --bind "$PGDATA:/var/lib/postgresql/data" \
        --bind "$PGRUN:/var/run/postgresql" \
        {pg_sif} pg-server \
        -c port="$PGPORT" \
        -c listen_addresses='*' \
        -c shared_buffers={self.postgres.shared_buffers} \
        -c work_mem={self.postgres.work_mem}
        """

        wait_for_postgres = "sleep 10"

        start_phoenix = f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'} && uv run phoenix serve"

        return "\n".join(
            [prepare_pg_data, start_postgres, wait_for_postgres, start_phoenix]
        )
