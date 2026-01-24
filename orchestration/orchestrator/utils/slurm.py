# Generated with ChatGPT

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from orchestration.orchestrator.utils.run_commands import run_ssh
from orchestration.orchestrator.utils.types import JobId

logger = logging.getLogger(__name__)


class JobNotStartedError(Exception):
    pass


async def wait_for_job_to_start(
    login: str,
    job_id: JobId,
    no_retries: int = 5,
    wait_seconds_after_check: int = 5,
) -> SlurmJobInfo:
    for _ in range(no_retries):
        logger.info(f"Trying to fetch job info {job_id}")
        out = (
            await asyncio.to_thread(
                run_ssh, login, f"squeue -j {job_id} -h -o %N | head -n1"
            )
        ).stdout

        if out.strip() != "(null)" and out.strip():
            return await asyncio.to_thread(get_slurm_job_info, login, job_id)

        await asyncio.sleep(wait_seconds_after_check)

    raise JobNotStartedError()


class SlurmJobInfo(BaseModel):
    job_id: int = Field(..., description="Slurm job id")
    state: str = Field(..., description="Job state, e.g. PENDING/RUNNING/COMPLETED")
    node: Optional[str] = Field(
        None, description="Allocated node (None if not allocated yet)"
    )
    cpus: Optional[int] = Field(None, ge=1, description="Allocated CPUs")
    gpus: int = Field(0, ge=0, description="Allocated GPUs (derived from GRES)")
    gres_raw: Optional[str] = Field(
        None, description="Raw GRES field from Slurm, e.g. gpu:2"
    )
    mem_raw: Optional[str] = Field(
        None, description="Raw memory field from Slurm, e.g. 64G"
    )

    @field_validator("node", mode="before")
    @classmethod
    def normalize_node(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        if not v or v == "(null)" or v == "None":
            return None
        return v

    @staticmethod
    def parse_gpu_count_from_gres(gres: str | None) -> int:
        """
        Examples seen in the wild:
          gpu:2
          gpu:a100:1
          gpu:rtx8000:4
          gpu:2(IDX:0-1)
          (null)
        """
        if not gres:
            return 0
        gres = gres.strip()
        if not gres or gres == "(null)":
            return 0

        # Find first occurrence like gpu:<anything>:<N> or gpu:<N>
        # We'll extract the last ":<int>" if present, else "gpu:<int>"
        m = re.search(r"\bgpu(?::[A-Za-z0-9_-]+)*:(\d+)\b", gres)
        if m:
            return int(m.group(1))

        # Sometimes it might just be "gpu" without count (rare)
        if "gpu" in gres:
            return 1

        return 0

    @classmethod
    def from_squeue_line(cls, line: str) -> "SlurmJobInfo":
        """
        Expects: "%i|%T|%N|%C|%b|%m"
        """
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) != 6:
            raise ValueError(
                f"Unexpected squeue format. Got {len(parts)} fields: {parts}"
            )

        job_id_s, state, node, cpus_s, gres_raw, mem_raw = parts

        cpus = None
        if cpus_s and cpus_s != "(null)":
            try:
                cpus = int(cpus_s)
            except ValueError:
                cpus = None

        gpus = cls.parse_gpu_count_from_gres(gres_raw)

        return cls(
            job_id=int(job_id_s),
            state=state,
            node=node,
            cpus=cpus,
            gpus=gpus,
            gres_raw=gres_raw if gres_raw and gres_raw != "(null)" else None,
            mem_raw=mem_raw if mem_raw and mem_raw != "(null)" else None,
        )

    @classmethod
    def from_sacct_line(cls, line: str) -> "SlurmJobInfo":
        """
        Expects sacct output from:

            sacct -n name -X --parsable2 -o JobIDRaw,State,NodeList,AllocCPUS,ReqGRES,ReqMem

        Which yields:

            "%i|%T|%N|%C|%b|%m"
        """
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) != 6:
            raise ValueError(
                f"Unexpected sacct format. Got {len(parts)} fields: {parts}"
            )

        job_id_s, state, node, cpus_s, gres_raw, mem_raw = parts

        # Normalize null-ish values
        def norm(s: str) -> Optional[str]:
            if not s:
                return None
            if s in {"(null)", "Unknown", "None"}:
                return None
            return s

        # Job id (should be numeric if JobIDRaw was used)
        try:
            job_id = int(job_id_s)
        except ValueError:
            # Fallback: if someone used JobID instead of JobIDRaw, we might see "123.batch"
            # Take the base numeric part.
            base = job_id_s.split(".", 1)[0]
            job_id = int(base)

        # CPUs
        cpus: Optional[int] = None
        cpus_s = norm(cpus_s) or ""
        if cpus_s:
            try:
                cpus = int(cpus_s)
            except ValueError:
                cpus = None

        gres_raw_n = norm(gres_raw)
        mem_raw_n = norm(mem_raw)

        # GPU count derived from GRES string
        gpus = cls.parse_gpu_count_from_gres(gres_raw_n)

        # Node sometimes comes as "node[01-02]" or empty
        node_n = norm(node)

        return cls(
            job_id=job_id,
            state=state,
            node=node_n,
            cpus=cpus,
            gpus=gpus,
            gres_raw=gres_raw_n,
            mem_raw=mem_raw_n,
        )


def get_slurm_job_info(login: str, job_id: int) -> SlurmJobInfo | None:
    """
    Returns:
      - SlurmJobInfo if job is still visible in squeue (PENDING/RUNNING/etc)
      - None if it is not in squeue anymore (finished/removed from queue)
    """
    cmd = f"squeue -j {job_id} -h -o '%i|%T|%N|%C|%b|%m'"
    out = run_ssh(login, cmd, capture=True)

    line = out.stdout.strip()
    if not line:
        return None

    # squeue can output multiple lines (job steps etc) on some setups; take first
    first_line = line.splitlines()[0]
    return SlurmJobInfo.from_squeue_line(first_line)


def get_job_info_by_name(
    login: str, service_name: str, query_history: bool = False
) -> list[SlurmJobInfo]:
    """
    Returns a list of SlurmJobInfo objects for jobs matching `service_name`
    that are still visible in squeue (PENDING/RUNNING/etc).
    """
    if query_history:
        cmd = (
            f"sacct -n --name {service_name} -X --parsable2 "
            f"-o JobIDRaw,State,NodeList,AllocCPUS,ReqTRES,ReqMem"
        )
    else:
        cmd = f"squeue -n {service_name} -h -o '%i|%T|%N|%C|%b|%m'"

    out = run_ssh(login, cmd, capture=True)

    lines = [l.strip() for l in out.stdout.strip().splitlines()]

    if query_history:
        infos = [SlurmJobInfo.from_squeue_line(line) for line in lines if line]
    else:
        infos = [SlurmJobInfo.from_sacct_line(line) for line in lines if line]

    return infos
