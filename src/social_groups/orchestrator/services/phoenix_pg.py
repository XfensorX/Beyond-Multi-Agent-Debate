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
        backup_wal_directory = working_dir / "wal_backup"
        postgres_run_dir = working_dir / "pgrun"

        wal_dir = f"/dev/shm/pgwal_{self.database_user.replace('.', '_')}"

        pw = "thismustnotbesecure"

        return {
            "PGDATA": str(postgres_dir),
            "PGWAL": str(wal_dir),
            "PG_WAL_BACKUP": str(backup_wal_directory),
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
        apptainer_options = "--no-mount bind-paths --bind $PGDATA:/var/lib/postgresql/data --bind $PGRUN:/var/run/postgresql --writable-tmpfs"

        prepare_pg_data = "mkdir -p $PGDATA && chmod 700 $PGDATA"
        prepare_pg_backup_wal = "mkdir -p $PG_WAL_BACKUP && chmod 700 $PG_WAL_BACKUP"
        prepare_pg_run = "mkdir -p $PGRUN && chmod 700 $PGRUN"
        prepare_wal_dir = "mkdir -p $PGWAL && chmod 700 $PGWAL"
        symlink_pg_wal = "ln -s $PGWAL $PGDATA/pg_wal"
        wal_backup = (
            "rm -rf $PG_WAL_BACKUP && cp -R $PGWAL $PG_WAL_BACKUP && echo WAL-BACKUP"
        )
        signal_pg_recovery_needed = 'touch "$PGDATA/recovery.signal"'
        wal_load_backup = "rm -rf $PGWAL && cp -R $PG_WAL_BACKUP $PGWAL"

        init = (
            f"apptainer exec {apptainer_options} {str(pg_sif)} "
            "bash -c "
            f'"{prepare_wal_dir} && '
            f"initdb -D /var/lib/postgresql/data --waldir=$PGWAL --auth-local=trust --auth-host=trust && "
            f'{wal_backup} "'
        )

        init_postgres_db = f"""
if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
    {init}
fi
        """

        capture_pg_pid = "PG_PID=$!"

        register_cleanup_for_postgres = """
cleanup() {
    # prevent running twice
    [[ -v CLEANUP_DONE ]] && return
    CLEANUP_DONE=1
    
    echo "Cleanup running at $(date)"
    kill -TERM $PHOENIX_PID || true
    kill -TERM $PG_PID || true
    wait "$PHOENIX_PID" || true
    wait "$PG_PID" || true
    echo "        finished at $(date)"
}
trap 'cleanup' EXIT SIGTERM SIGINT
        """

        start_postgres = (
            # f"apptainer instance start {apptainer_options} {str(pg_sif)} phoenix-postgres "
            # "nohup "
            f"apptainer exec {apptainer_options} {str(pg_sif)} "
            "bash -c "
            "'"
            f"""
            set -euo pipefail
        
            {prepare_wal_dir}
            {wal_load_backup}
            {symlink_pg_wal}
        
            cleanup() {{
                # prevent running twice
                [[ -v APPTAINER_CLEANUP_DONE ]] && return
                APPTAINER_CLEANUP_DONE=1
                
                echo "TRAP - Stopping Postgres from inside apptainer..."
                kill -TERM "$APPTAINER_PG_PID" || true
                wait "$APPTAINER_PG_PID" || true
                {wal_backup}
            }}
            trap cleanup EXIT TERM INT
            
        
            # Start Postgres in foreground (best) or background + wait
            postgres -D /var/lib/postgresql/data \\
            -p "$PGPORT" \\
            -k /var/run/postgresql \\
            --shared_buffers=4GB \\
            --work_mem=128MB \\
            --huge_pages=off \\
            --checkpoint_timeout=1min \\
            -c "restore_command=cp $PG_WAL_BACKUP/%f %p" \\
            &
        
            APPTAINER_PG_PID=$!
            wait $APPTAINER_PG_PID
            """
            "' "
            "& "
        )

        wait_for_postgres = f"until apptainer exec {apptainer_options} {pg_sif} pg_isready -h /var/run/postgresql -p $PGPORT -U {self.database_user} -d postgres; do \n sleep 1 \ndone"

        activate_env = (
            f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'}"
        )
        start_phoenix = "uv run phoenix serve &"
        capture_phoenix_pid = "PHOENIX_PID=$!"

        wait_until_everything_stopped = "wait $PHOENIX_PID && wait $PG_PID"

        return "\n".join(
            [
                prepare_wal_dir,
                prepare_pg_backup_wal,
                prepare_pg_run,
                prepare_pg_data,
                init_postgres_db,
                signal_pg_recovery_needed,
                start_postgres,
                capture_pg_pid,
                register_cleanup_for_postgres,
                wait_for_postgres,
                activate_env,
                start_phoenix,
                capture_phoenix_pid,
                wait_until_everything_stopped,
            ]
        )
