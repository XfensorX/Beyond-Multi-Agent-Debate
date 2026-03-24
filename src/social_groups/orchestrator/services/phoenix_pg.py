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


DEFAULT_DATABASE_NAME = "postgres"


@register_slurm_service("phoenix-pg")
class PhoenixWithPostgresConfiguration(SlurmService):
    port: int
    postgres: PostgresConfig
    database_user: str

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
        phoenix_dir = working_dir / "phoenix"
        postgres_run_dir = working_dir / "pgrun"

        wal_dir = f"/dev/shm/pgwal_{self.database_user.replace('.', '_')}"

        pw = "thismustnotbesecure"

        return {
            "PGDATA": str(postgres_dir),
            "PGWAL": str(wal_dir),
            "PGRUN": str(postgres_run_dir),
            "PGPORT": str(self.postgres.port),
            "POSTGRES_PORT": str(self.postgres.port),
            # "POSTGRES_PASSWORD": pw,
            ##
            "PHOENIX_PORT": str(self.port),
            "PHOENIX_ALLOW_EXTERNAL_RESOURCES": "false",
            "PHOENIX_WORKING_DIR": str(phoenix_dir),
            "PHOENIX_TELEMETRY_ENABLED": "false",
            "PHOENIX_SQL_DATABASE_URL": f"postgresql://{self.database_user}@/{DEFAULT_DATABASE_NAME}?host={str(postgres_run_dir)}&port={self.postgres.port}",
            # "PHOENIX_SQL_DATABASE_URL": f"postgresql://{self.database_user}@localhost:{self.postgres.port}/postgres?sslmode=disable",
        }

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        pg_sif = exec_config.project_dir / self.postgres.sif_location_inside_project

        prepare_wal_dir = "mkdir -p $PGWAL && chmod 700 $PGWAL"
        prepare_wal_archive = (
            "mkdir -p $PGWAL/archive_status && chmod 700 $PGWAL/archive_status"
        )
        prepare_pg_data = "mkdir -p $PGDATA && chmod 700 $PGDATA"
        prepare_pg_run = "mkdir -p $PGRUN && chmod 700 $PGRUN"

        clean_old_pid = "rm -f $PGDATA/postmaster.pid"

        apptainer_options = "--no-mount bind-paths --bind $PGDATA:/var/lib/postgresql/data --bind $PGRUN:/var/run/postgresql --writable-tmpfs"

        init = (
            f"apptainer exec {apptainer_options} {str(pg_sif)} "
            # "env LANG=C LC_ALL=C "
            "bash -c "
            f'"{prepare_wal_dir} && initdb -D /var/lib/postgresql/data --waldir=$PGWAL"'
        )

        init_postgres_db = f"""
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    {init}
fi
        """

        symlink_pg_wal = "ln -s $PGWAL $PGDATA/pg_wal"

        reset_wal_dir = "(pg_resetwal -f -D $PGDATA || true)"

        capture_pg_id = "PG_PID=$!"
        register_cleanup_for_postgres = f"""
cleanup() {{
    echo "Cleanup running at $(date)"
    apptainer exec {apptainer_options} {str(pg_sif)} psql -d {DEFAULT_DATABASE_NAME} -U {self.database_user} -c 'CHECKPOINT;'
    kill --signal SIGTERM PG_PID
    echo "Cleanup finished at $(date)"
}}
trap cleanup EXIT SIGTERM SIGINT
        """

        start_postgres = (
            # f"apptainer instance start {apptainer_options} {str(pg_sif)} phoenix-postgres "
            # "nohup "
            f"apptainer exec {apptainer_options} {str(pg_sif)} "
            "bash -c "
            '"'
            f"{clean_old_pid} && {prepare_wal_dir} && {prepare_wal_archive} && {symlink_pg_wal} && {reset_wal_dir} && "
            "postgres "
            "-D /var/lib/postgresql/data "
            "-p $PGPORT "
            # "-h /var/run/postgresql "
            "-k /var/run/postgresql "
            f"--shared_buffers={self.postgres.shared_buffers} "
            f"--work_mem={self.postgres.work_mem} "
            "--huge_pages=off "
            "--checkpoint_timeout=1min "
            '" '
            "&"
        )

        wait_for_postgres = f"until apptainer exec {apptainer_options} {pg_sif} pg_isready -h /var/run/postgresql -p $PGPORT -U {self.database_user} -d postgres; do \n sleep 1 \ndone"

        start_phoenix = f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'} && uv run phoenix serve"

        return "\n".join(
            [
                prepare_wal_dir,
                prepare_pg_run,
                prepare_pg_data,
                init_postgres_db,
                start_postgres,
                capture_pg_id,
                register_cleanup_for_postgres,
                wait_for_postgres,
                start_phoenix,
            ]
        )
