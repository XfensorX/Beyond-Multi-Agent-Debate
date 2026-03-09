from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
from asyncio import as_completed
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Optional

import questionary
import rich
import typer
from questionary import Choice
from rich import print
from rich.text import Text

from social_groups.directories import RESULTS_DIR
from social_groups.general.run_commands import run_local, run_ssh
from social_groups.orchestrator.commands.sync import (
    get_remote_run_dirs,
    warn_about_existing_conflict,
)
from social_groups.orchestrator.config import (
    get_slurm_log_filename,
    random_string_to_job_name_appendix,
)
from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocation,
    load_execution_config,
)
from social_groups.orchestrator.services import (
    BaseInferenceService,
    ExperimentConfiguration,
    ExperimentStartingInformation,
    PhoenixConfiguration,
    SlurmServiceName,
    load_config,
)
from social_groups.orchestrator.utils.general import run_async
from social_groups.orchestrator.utils.running_backends import (
    parse_backend_endpoint_from_slurm_job,
)
from social_groups.orchestrator.utils.slurm import (
    SlurmJobInfo,
    get_slurm_job_info,
    wait_for_job_to_start,
)

app = typer.Typer(no_args_is_help=True)
console = rich.console.Console()
logging.basicConfig(level=logging.DEBUG)


@app.command(
    "start",
    help="[blue]{location} {service}[/blue] Start a Slurm job. (append --llm or --exp if necessary)",
)
@run_async
async def run_service_on_slurm(
    where: Annotated[ExecutionLocation, typer.Argument(help="Where to run.")],
    service: Annotated[
        SlurmServiceName, typer.Argument(help="Which service to start.")
    ],
    llm: Optional[list[str]] = typer.Option(
        None,
        "--llm",
        "-l",
        help="If starting an inference service, declaring which LLMs to start. Can be used several times to start several services.",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "--exp",
        "-e",
        help="If starting an experiment, declaring which experiment to start.",
    ),
    multirun: Optional[bool] = typer.Option(
        False,
        "--multirun",
        "-m",
        help="Activates the hydra multirun when starting an experiment.",
    ),
    gpus: Optional[int] = typer.Option(
        None,
        "--gpus",
        "-g",
        help="How many GPUs to use if is a LLM job.",
    ),
):
    # TODO: refactor this method
    if llm:
        if len(llm) != len(set(llm)):
            raise typer.BadParameter(
                "Cannot start multiple services with the same LLM."
            )

    exec_config = load_execution_config(where)
    print(exec_config.model_dump_json(indent=4))

    service_config = load_config(service, where)
    print(service_config.model_dump_json(indent=4))

    if isinstance(service_config, BaseInferenceService):
        if not llm:
            raise typer.BadParameter(
                "Must specify llms when starting TGI. (with --llm or -l)"
            )
        started_jobs = []
        try:
            service_config.check_model_config_exists(llm)
        except ValueError as e:
            raise typer.BadParameter(str(e))

        for llm_name in llm:
            print(f"Starting {service} on {where} for model {llm_name}")
            service_config.set_used_model(llm_name, used_gpus=gpus)
            job_id = exec_config.submit_sbatch(
                service_config.create_job_file_content(exec_config=exec_config)
            )
            started_jobs.append(
                asyncio.create_task(
                    wait_for_job_to_start(exec_config.ssh_login, job_id)
                )
            )

        for job in as_completed(started_jobs):
            job_info = await job
            print(job_info.model_dump_json(indent=4))

    elif isinstance(service_config, ExperimentConfiguration):
        if experiment is None:
            raise typer.BadParameter("Must specify experiment name. (with -e)")

        print("Fetching for job infos.")

        current_jobs = get_slurm_job_info(exec_config.ssh_login)

        phoenix_jobs: list[SlurmJobInfo] = list(
            filter(
                lambda j: PhoenixConfiguration.job_name_is_matching_this_service(
                    j.job_name
                ),
                current_jobs,
            )
        )

        if len(phoenix_jobs) != 1:
            raise typer.BadParameter(
                f"Needs 1 phoenix job. Detected {len(phoenix_jobs)}"
            )

        phoenix_job = phoenix_jobs[0]

        # TODO: make typesafe
        # noinspection PyTypeChecker
        phoenix_config: PhoenixConfiguration = load_config(
            SlurmServiceName["phoenix"], where
        )

        service_config.add_starting_info(
            ExperimentStartingInformation(
                experiment_name=experiment,
                phoenix_server_endpoint=f"http://{phoenix_job.node}:{phoenix_config.port}",
                phoenix_graphql_endpoint=f"http://{phoenix_job.node}:4317",  # TODO: make this configurable
                model_backends=[
                    endpoint
                    for info in current_jobs
                    if (endpoint := parse_backend_endpoint_from_slurm_job(info, where))
                ],
                use_hydra_multirun=multirun if multirun else False,
            )
        )
        print(service_config._starting_info)

        job_file = service_config.create_job_file_content(exec_config=exec_config)

        print("---Submitted Job File: ---------------------")
        print(job_file)
        print("--------------------------------------------")

        print(f"Starting {service} on {where} for experiment {experiment}")
        job_id = exec_config.submit_sbatch(job_file)

        job_info = await wait_for_job_to_start(exec_config.ssh_login, job_id)

        print(job_info.model_dump_json(indent=4))

    else:
        print(f"Starting {service} on {where}")

        job_id = exec_config.submit_sbatch(
            service_config.create_job_file_content(exec_config=exec_config)
        )

        job_info = await wait_for_job_to_start(exec_config.ssh_login, job_id)

        print(job_info.model_dump_json(indent=4))


@app.command("show", help="[blue]{location}          [/blue] Show all slurm jobs.")
def show_services(where: ExecutionLocation):
    exec_config = load_execution_config(where)
    cmd = "squeue -O jobid:8,name:50,state:16,timeused:10,reasonlist:32,tres-per-job:30,tres-alloc:0 -u $(whoami)"
    out = run_ssh(exec_config.ssh_login, cmd, capture=True)
    print(out.stdout)


@app.command("stop", help="[blue]{location}        [/blue] Stop selected services. ")
def stop_service(where: ExecutionLocation):
    exec_config = load_execution_config(where)

    infos = get_slurm_job_info(exec_config.ssh_login)
    to_cancel: list[SlurmJobInfo] = questionary.checkbox(
        "Select Jobs to stop immediately:",
        choices=[Choice(title=str(it), value=it) for it in infos],
    ).ask()

    cmd = f"scancel {' '.join([str(j.job_id) for j in to_cancel])}"
    out = run_ssh(exec_config.ssh_login, cmd, capture=True)
    print(out)


# TODO: add command to schedule a job that kills service on finish of another service


@app.command(
    "log",
    help="[blue]{location}[/blue] [yellow]{service}[/yellow] Show the logs of a service. (Omit {service} to search through all.)",
)
def show_logs(
    where: ExecutionLocation, service: Optional[SlurmServiceName] = typer.Argument(None)
):
    exec_config = load_execution_config(where)
    infos = get_slurm_job_info(exec_config.ssh_login, query_history=True)

    if service is not None:
        service_config = load_config(service, where)
        infos = [
            j
            for j in infos
            if service_config.job_name_is_matching_this_service(j.job_name)
        ]

    if not infos:
        print("No jobs found running")

    if len(infos) > 1:
        used_info: SlurmJobInfo = questionary.select(
            "Multiple jobs found running, please select the log to use: ",
            choices=[Choice(title=str(it), value=it) for it in infos],
            qmark=">",
            pointer="➤",
        ).ask()
    else:
        used_info = infos[0]

    # TODO: maybe make interactive with "less" or so
    file_path = exec_config.get_logdir() / get_slurm_log_filename(
        used_info.job_name, used_info.job_id
    )
    cmd = f"cat {file_path}"

    out = run_ssh(exec_config.ssh_login, cmd, capture=True)
    print(f"\nShowing Log for: [blue]{used_info}[/blue]")
    print(f"Located in {file_path} \n")
    print(Text.from_ansi(out.stdout))


@app.command(
    "pipe",
    help="[blue]{location} {service}[/blue] Pipe a services output with ssh to local port.",
)
def pipe_ssh(where: ExecutionLocation, service: SlurmServiceName):
    exec_config = load_execution_config(where)
    service_config = load_config(service, where)

    if not (
        isinstance(service_config, PhoenixConfiguration)
        or isinstance(service_config, BaseInferenceService)
    ):
        raise NotImplementedError()

    infos = [
        info
        for info in get_slurm_job_info(exec_config.ssh_login)
        if service_config.job_name_is_matching_this_service(info.job_name)
    ]

    if len(infos) > 1:
        print("Multiple jobs found running, selecting first.")
        selected: SlurmJobInfo = questionary.select(
            "Multiple jobs found running, please select the one to pipe to: ",
            choices=[Choice(title=str(it), value=it) for it in infos if it.job_name],
            qmark=">",
            pointer="➤",
        ).ask()
        used_info: SlurmJobInfo = selected
    else:
        used_info = infos[0]

    node = used_info.node

    if isinstance(service_config, PhoenixConfiguration):
        port = service_config.port
    elif isinstance(service_config, BaseInferenceService):
        port = None
        for backend in set(
            endpoint
            for info in infos
            if (endpoint := parse_backend_endpoint_from_slurm_job(info, where))
        ):
            if (
                random_string_to_job_name_appendix(backend.model_name)
                in used_info.job_name
            ):
                port = backend.endpoint.split(":")[-1]
                break

        if port is None:
            raise RuntimeError(
                "Selected Job Not Found in configs. You must have deleted the config after start."
            )

    else:
        raise NotImplementedError()

    run_local(  # Blocking
        [
            "ssh",
            "-N",
            "-L",
            f"localhost:{port}:localhost:{port}",
            node,
        ]
    )


@app.command(
    "sync",
    help="[blue]{location}          [/blue] Sync selected experiment runs from host",
)
def sync_results(
    where: ExecutionLocation,
    experiment_filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="A filter used on the experiment name."
    ),
):
    exec_config = load_execution_config(where)

    HOST_LOGIN = exec_config.ssh_login
    HOST_BASE = str(Path(exec_config.project_dir) / "results" / "multirun")
    LOCAL_BASE = RESULTS_DIR / "multirun" / "final"

    console.rule("Remote experiment sync")

    with console.status("[cyan]Listing remote runs…"):
        remote_run_relpaths = get_remote_run_dirs(HOST_LOGIN, HOST_BASE)

    if not remote_run_relpaths:
        console.print("[yellow]No run directories found in[/yellow]", HOST_BASE)
        return

    choices = []
    for rel_path in remote_run_relpaths:
        node, exp, run = rel_path.parts
        if experiment_filter and experiment_filter not in exp:
            continue
        choices.append({"name": f"{exp:50}  •  {run:24}  ({node})", "value": rel_path})
    choices.sort(key=itemgetter("name"))

    selected: list[Path] = questionary.checkbox(
        "Select run directories to sync:",
        choices=choices,
        qmark="↳",
        pointer="→",
        validate=lambda x: len(x) > 0 or "Select at least one run",
    ).ask()

    if not selected:
        console.print("[grey]Nothing selected → exiting.[/]")
        return

    # ── Conflict check ──────────────────────────────────────────────
    should_not_use = set()
    for rel_path in selected:
        if (LOCAL_BASE / Path(*rel_path.parts[1:-1])).exists():
            warn_about_existing_conflict(rel_path.parent, console)
            if not questionary.confirm(
                f"Continue syncing anyway to [yellow]{rel_path}[/] ?",
                default=False,
            ).ask():
                console.print(f"→ Skipping {rel_path}", style="yellow")
                should_not_use.add(rel_path)

    to_sync = [
        (Path(HOST_BASE) / rel_path, LOCAL_BASE / Path(*rel_path.parts[1:]))
        for rel_path in selected
        if rel_path not in should_not_use
    ]

    console.print(f"\n[bold cyan]Will sync {len(selected)} run(s):[/]")
    for from_path, to_path in to_sync:
        console.print(
            f"  • {str(Path(*from_path.parts[-4:])):75} → {str(Path(*to_path.parts[-4:]))}",
            style="dim",
        )

    if not questionary.confirm("Proceed with rsync?", default=True).ask():
        return

    for from_path, to_path in to_sync:
        rsync_cmd = [
            "rsync",
            "-ah",
            "--info=progress2",
            "--info=name0",
            "--no-inc-recursive",
            "--human-readable",
            f"{HOST_LOGIN}:{from_path}/",
            f"{to_path}/",
        ]

        console.rule(f"Syncing  {from_path}")
        console.print(" ".join(shlex.quote(x) for x in rsync_cmd[:5]) + " ...")

        try:
            to_path.mkdir(parents=True, exist_ok=True)
            run_local(rsync_cmd)
            console.print(f"[green]✓ Done[/] {rel_path}", style="green")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]rsync failed[/] for {rel_path}", style="red")
            console.print(e)
            if not questionary.confirm("Continue with next runs?", default=True).ask():
                break

    console.rule("Finished")
    console.print(f"[green]Sync completed.[/] Target folder:\n{LOCAL_BASE}")


def main():
    app()


if __name__ == "__main__":
    main()
