from __future__ import annotations

import logging
from typing import Annotated

import typer

from orchestration.orchestrator.config import (
    get_slurm_log_filename,
)
from orchestration.orchestrator.models.execution_environment import (
    ExecutionLocation,
    load_execution_config,
)
from orchestration.orchestrator.services.base import (
    SlurmServiceName,
    load_config,
)
from orchestration.orchestrator.services.phoenix import PhoenixConfiguration
from orchestration.orchestrator.utils.general import run_async
from orchestration.orchestrator.utils.run_commands import run_local, run_ssh
from orchestration.orchestrator.utils.slurm import (
    get_job_info_by_name,
    wait_for_job_to_start,
)

app = typer.Typer()
logging.basicConfig(level=logging.DEBUG)


@app.command("start", help="[blue]{location} {service}[/blue] Start a Slurm job.")
@run_async
async def run_service_on_slurm(
    where: Annotated[ExecutionLocation, typer.Argument(help="Where to run.")],
    service: Annotated[
        SlurmServiceName, typer.Argument(help="Which service to start.")
    ],
):
    typer.echo(f"Starting {service} on {where}")

    service_config = load_config(service, where)
    exec_config = load_execution_config(where)

    typer.echo(exec_config.model_dump_json(indent=4))
    typer.echo(service_config.model_dump_json(indent=4))

    job_id = exec_config.submit_sbatch(
        service_config.create_job_file_content(exec_config=exec_config)
    )

    job_info = await wait_for_job_to_start(exec_config.ssh_login, job_id)

    print(job_info.model_dump_json(indent=4))


@app.command("show", help="[blue]{location}          [/blue] Show all slurm jobs.")
def show_services(where: ExecutionLocation):
    exec_config = load_execution_config(where)
    cmd = "squeue -O jobid:8,name:32,state:16,timeused:10,reasonlist:32,tres-per-job:30,tres-alloc:0 -u $(whoami)"
    out = run_ssh(exec_config.ssh_login, cmd, capture=True)
    typer.echo(out.stdout)


@app.command("stop", help="[blue]{location} {service}[/blue] Stop a service. ")
def stop_service(where: ExecutionLocation, service: SlurmServiceName):
    service_config = load_config(service, where)
    exec_config = load_execution_config(where)
    cmd = f"scancel -n {service_config.slurm_config.job_name}"

    out = run_ssh(exec_config.ssh_login, cmd, capture=True)
    typer.echo(out.stdout)


@app.command(
    "log", help="[blue]{location} {service}[/blue] Show the logs of a service."
)
def show_logs(where: ExecutionLocation, service: SlurmServiceName):
    exec_config = load_execution_config(where)
    service_config = load_config(service, where)

    infos = get_job_info_by_name(
        exec_config.ssh_login, service_name=service_config.slurm_config.job_name
    )

    if not infos:
        typer.echo("No jobs found running")
    # TODO: lets user select the jobId
    if len(infos) > 1:
        typer.echo("Multiple jobs found running, selecting first.")
    job_id = infos[0].job_id

    # TODO: maybe make interactive with "less" or so
    cmd = f"cat {exec_config.get_logdir() / get_slurm_log_filename(service_config.slurm_config.job_name, job_id)}"

    out = run_ssh(exec_config.ssh_login, cmd, capture=True)
    typer.echo(out.stdout)


@app.command(
    "pipe",
    help="[blue]{location} {service}[/blue] Pipe a services output with ssh to local port.",
)
def pipe_ssh(where: ExecutionLocation, service: SlurmServiceName):
    exec_config = load_execution_config(where)
    service_config = load_config(service, where)
    if not isinstance(service_config, PhoenixConfiguration):
        raise NotImplementedError()

    infos = get_job_info_by_name(
        exec_config.ssh_login, service_name=service_config.slurm_config.job_name
    )
    # TODO: lets user select the jobId
    if len(infos) > 1:
        typer.echo("Multiple jobs found running, selecting first.")
    node = infos[0].node
    port = service_config.port

    run_local(  # Blocking
        [
            "ssh",
            "-N",
            "-L",
            f"localhost:{port}:localhost:{port}",
            node,
        ]
    )


if __name__ == "__main__":
    app()
