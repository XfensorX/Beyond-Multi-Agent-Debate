from __future__ import annotations

import logging

import typer

from orchestration.orchestrator.config import (
    load_config,
    load_execution_config,
)
from orchestration.orchestrator.models.execution_environment import (
    ExecutionEnvironment,
)
from orchestration.orchestrator.services.base import (
    SlurmServiceName,
)
from orchestration.orchestrator.utils.general import run_async
from orchestration.orchestrator.utils.slurm import wait_for_job_to_start

app = typer.Typer()
logging.basicConfig(level=logging.DEBUG)


@app.command("start")
@run_async
async def run_service_on_slurm(service: SlurmServiceName, where: ExecutionEnvironment):
    typer.echo(f"Starting {service} on {where}")

    service_config = load_config(service, where)
    exec_config = load_execution_config(where)

    typer.echo(exec_config.model_dump_json(indent=4))
    typer.echo(service_config.model_dump_json(indent=4))

    job_id = exec_config.submit_sbatch(
        service_config.create_job_file_content(exec_config=exec_config)
    )

    print(f"JOB ID: {job_id}")

    job_info = await wait_for_job_to_start(exec_config.ssh_login, job_id)

    print(job_info)


@app.command("stop")
def stop_service_on_slurm(service: SlurmServiceName, where: ExecutionEnvironment):
    typer.echo(f"Starting {service} on {where}")
    raise NotImplementedError()


if __name__ == "__main__":
    app()
