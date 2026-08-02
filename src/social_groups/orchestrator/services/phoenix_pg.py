from pathlib import Path
from typing import Any

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
    graphql_port: int
    postgres: PostgresConfig
    database_user: str

    phoenix_sql_alchemy_pool_size: int
    phoenix_sql_alchemy_max_overflow: int

    phoenix_internal_ports: list[int]
    phoenix_internal_graphql_ports: list[int]

    ngingx_conf_location_inside_project: Path
    ngingx_sif_location_inside_project: Path

    @property
    def used_job_name(self) -> str:
        return "phoenix-pg"

    @staticmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool:
        return given_job_name == "phoenix-pg"

    def create_env_dict(self, exec_config: ExecutionLocationConfig) -> dict[str, Any]:
        working_dir = (  # This cannot use the global variable, because it is used in orchestrator
            exec_config.project_dir / "results" / exec_config.where.value / "postgres"
        )

        postgres_dir = working_dir / "pgdata"
        phoenix_dir = working_dir / "phoenix"
        postgres_run_dir = working_dir / "pgrun"
        recovery_dir = working_dir / "full_backup"

        wal_dir = f"/dev/shm/pgwal_{self.database_user.replace('.', '_')}"

        return {
            "PHOENIX_PORTS": self.phoenix_internal_ports,
            "PHOENIX_GRPC_PORTS": self.phoenix_internal_graphql_ports,
            # Postgres
            "PGDATA": postgres_dir,
            "PGWAL": wal_dir,
            "PG_WAL_BACKUP": (working_dir / "wal_backup"),
            "PG_FULL_BACKUP_DIR": recovery_dir,
            "PG_WAL_COMBINED": (working_dir / "wal_combined"),
            "PGRUN": postgres_run_dir,
            "PGPORT": self.postgres.port,
            "POSTGRES_PORT": self.postgres.port,
            #
            # Phoenix
            "PHOENIX_ALLOW_EXTERNAL_RESOURCES": "false",
            "PHOENIX_WORKING_DIR": phoenix_dir,
            "PHOENIX_TELEMETRY_ENABLED": "false",
            "PHOENIX_SQL_DATABASE_URL": f"postgresql://{self.database_user}@/{DEFAULT_DATABASE_NAME}?host={str(postgres_run_dir)}&port={self.postgres.port}",
            #
            # Phoenix Performance
            "PHOENIX_SQLALCHEMY_POOL_SIZE": self.phoenix_sql_alchemy_pool_size,
            "PHOENIX_SQLALCHEMY_MAX_OVERFLOW": self.phoenix_sql_alchemy_max_overflow,
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
        }

    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str:
        pg_sif = exec_config.project_dir / self.postgres.sif_location_inside_project
        nginx_sif = exec_config.project_dir / self.ngingx_sif_location_inside_project
        nginx_conf = exec_config.project_dir / self.ngingx_conf_location_inside_project

        apptainer_options = (
            "--no-mount bind-paths "
            "--bind $PGDATA:/var/lib/postgresql/data "
            "--bind $PGRUN:/var/run/postgresql  "
            "--bind $PG_WAL_BACKUP:$PG_WAL_BACKUP "
            "--bind $PG_WAL_COMBINED:$PG_WAL_COMBINED "
            "--bind $PG_FULL_BACKUP_DIR:$PG_FULL_BACKUP_DIR "
            "--writable-tmpfs"
        )

        prepare_wal_dir = "mkdir -p $PGWAL && chmod 700 $PGWAL"
        wal_backup = 'cp -a "$PGWAL"/. "$PG_WAL_BACKUP"/ && echo WAL-BACKUP'
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
    INITIALIZED_DB_THIS_RUN=1
fi
        """

        capture_pg_pid = "PG_PID=$!"

        register_cleanup_for_postgres = """
cleanup() {
    # prevent running twice
    [[ -v CLEANUP_DONE ]] && return
    CLEANUP_DONE=1
    
    echo "Cleanup running at $(date)"
    
    # first stop all phoenix related stuff to flush the buffers
    for PID in "${PHOENIX_PIDS[@]}"; do
        kill -TERM "$PID" || true
    done
    kill -TERM "$NGINX_PID" || true
    for PID in "${PHOENIX_PIDS[@]}"; do
        wait "$PID" || true
    done
    wait "$NGINX_PID" || true
    
    kill -TERM $PG_PID || true
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
            rm -rf $PGDATA/pg_wal && ln -s $PGWAL $PGDATA/pg_wal
            echo \"$(readlink -f \"$PGWAL\")\" && find \"$PGWAL\" -mindepth 1 -delete && cp -a \"$PG_WAL_BACKUP\"/. \"$PGWAL\"/
        
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
            -c "restore_command=cp $PG_WAL_COMBINED/%f %p" \\
            &
            
            APPTAINER_PG_PID=$!
            
            until pg_isready -h /var/run/postgresql -p $PGPORT -U {self.database_user} -d postgres; do \n sleep 1 \ndone
            
            pg_basebackup -D $PG_FULL_BACKUP_DIR/$(date +%Y%m%d_%H%M%S) \\
            -F t \\
            -z \\
            -P \\
            -X stream \\
            -c fast \\
            &
            
            APPTAINER_PG_BACKUP_PID=$!
                    
            wait $APPTAINER_PG_BACKUP_PID
            wait $APPTAINER_PG_PID
            """
            "' "
            "& "
        )

        wait_for_postgres = f"until apptainer exec {apptainer_options} {pg_sif} pg_isready -h /var/run/postgresql -p $PGPORT -U {self.database_user} -d postgres; do \n sleep 1 \ndone"

        activate_env = (
            f"source {exec_config.project_dir / '.venv' / 'bin' / 'activate'}"
        )
        start_phoenix = f"""
PHOENIX_PIDS=()
NGINX_PID=""
for i in "${{!PHOENIX_PORTS[@]}}"; do
    PHOENIX_GRPC_PORT=${{PHOENIX_GRPC_PORTS[$i]}} \\
    PHOENIX_PORT=${{PHOENIX_PORTS[$i]}} \\
    uv run phoenix serve &
    PHOENIX_PIDS+=($!)
    
    if [[ -v INITIALIZED_DB_THIS_RUN ]]; then
        if [[ $i -eq 0 ]]; then
            sleep 20
        fi
    fi
done

echo "Starting NGINX (Apptainer)..."

apptainer exec \\
  --bind "{str(nginx_conf)}:{str(nginx_conf)}" \\
  --no-mount bind-paths \\
  --writable-tmpfs \\
  {nginx_sif} \\
  nginx -g "daemon off;" -c '{str(nginx_conf)}' &
  
NGINX_PID=$!


# Wait for everything
wait "${{PHOENIX_PIDS[@]}}"
wait "$NGINX_PID"
        """

        wait_postgres_stopped = "wait $PG_PID"

        return "\n".join(
            [
                prepare_wal_dir,
                "mkdir -p $PG_WAL_BACKUP && chmod 700 $PG_WAL_BACKUP",
                "mkdir -p $PGRUN && chmod 700 $PGRUN",
                "mkdir -p $PGDATA && chmod 700 $PGDATA",
                init_postgres_db,
                'touch "$PGDATA/recovery.signal"',
                start_postgres,
                capture_pg_pid,
                register_cleanup_for_postgres,
                wait_for_postgres,
                activate_env,
                start_phoenix,
                wait_postgres_stopped,
            ]
        )
