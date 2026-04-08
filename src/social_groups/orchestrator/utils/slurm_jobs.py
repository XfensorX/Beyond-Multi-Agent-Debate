from __future__ import annotations

import typer

from social_groups.orchestrator.models.execution_environment import ExecutionLocation
from social_groups.orchestrator.services import (
    PhoenixConfiguration,
    SlurmServiceName,
    load_config,
)
from social_groups.orchestrator.services.phoenix_pg import (
    PhoenixWithPostgresConfiguration,
)
from social_groups.orchestrator.utils.slurm import SlurmJobInfo

PhoenixConfigType = PhoenixConfiguration | PhoenixWithPostgresConfiguration


def with_phoenix_config(
    job: SlurmJobInfo, where: ExecutionLocation
) -> tuple[PhoenixConfigType, SlurmJobInfo] | None:
    if PhoenixConfiguration.job_name_is_matching_this_service(job.job_name):
        config = load_config(SlurmServiceName["phoenix"], where)
    elif PhoenixWithPostgresConfiguration.job_name_is_matching_this_service(
        job.job_name
    ):
        config = load_config(SlurmServiceName["phoenix-pg"], where)
    else:
        config = None
    if config is None:
        return None

    if not (
        isinstance(config, PhoenixConfiguration)
        or isinstance(config, PhoenixWithPostgresConfiguration)
    ):
        raise RuntimeError(f"Expected PhoenixConfigType but got {type(config)}")

    return config, job


async def extract_phoenix(
    current_jobs: list[SlurmJobInfo], where: ExecutionLocation
) -> tuple[PhoenixConfigType, SlurmJobInfo]:
    phoenix_jobs = [
        cj
        for cj in map(lambda j: with_phoenix_config(j, where), current_jobs)
        if cj is not None
    ]

    if len(phoenix_jobs) != 1:
        typer.echo(
            f"Detected {len(phoenix_jobs)} phoenix instances. Single phoenix-pg instance preferred."
        )
        phoenix_job = [
            (c, j)
            for (c, j) in phoenix_jobs
            if isinstance(c, PhoenixWithPostgresConfiguration)
        ]
        if len(phoenix_job) != 1:
            raise typer.BadParameter(
                f"Detected {len(phoenix_jobs)} phoenix-pg instances. Single phoenix-pg instance needed!"
            )

    return phoenix_jobs[0]
