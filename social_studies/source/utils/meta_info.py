from __future__ import annotations

from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel


import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple


class GitInfo(BaseModel):
    repo_root: Optional[str] = None
    remote_url: Optional[str] = None
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    commit_short_sha: Optional[str] = None
    is_dirty: Optional[bool] = None
    diff_sha256: Optional[str] = None  # hash of `git diff` output (not the diff itself)
    describe: Optional[str] = None  # e.g. tag-ish description


class ExecutionInfo(BaseModel):
    started_at_utc: str
    argv: List[str]
    command: str


class ExperimentMetaInfo(BaseModel):
    execution: ExecutionInfo
    git: Optional[GitInfo] = None
    output_directory: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(
    cmd: List[str], cwd: Optional[str] = None, timeout_s: float = 2.0
) -> Tuple[int, str, str]:
    """Run command, return (returncode, stdout, stderr) as strings. Best-effort safe."""
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return 999, "", f"{type(e).__name__}: {e}"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _git_info_best_effort() -> Optional[GitInfo]:
    # If git isn't installed or we're not in a repo, return None.
    rc, root, _ = _run(["git", "rev-parse", "--show-toplevel"], timeout_s=1.5)
    if rc != 0 or not root:
        return None

    rc, sha, _ = _run(["git", "rev-parse", "HEAD"], cwd=root, timeout_s=1.5)
    rc2, short_sha, _ = _run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, timeout_s=1.5
    )
    rc3, branch, _ = _run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, timeout_s=1.5
    )

    rc4, status, _ = _run(["git", "status", "--porcelain"], cwd=root, timeout_s=2.0)
    is_dirty = (status.strip() != "") if rc4 == 0 else None

    rc5, remote_url, _ = _run(
        ["git", "remote", "get-url", "origin"], cwd=root, timeout_s=1.5
    )
    if rc5 != 0:
        remote_url = ""

    rc6, desc, _ = _run(
        ["git", "describe", "--tags", "--always", "--dirty"], cwd=root, timeout_s=2.0
    )
    if rc6 != 0:
        desc = ""

    diff_hash: Optional[str] = None
    if is_dirty:
        rc7, diff, _ = _run(["git", "diff"], cwd=root, timeout_s=3.0)
        if rc7 == 0 and diff:
            diff_hash = _sha256_text(diff)

    return GitInfo(
        repo_root=root,
        remote_url=remote_url or None,
        branch=branch or None,
        commit_sha=sha or None,
        commit_short_sha=short_sha or None,
        is_dirty=is_dirty,
        diff_sha256=diff_hash,
        describe=desc or None,
    )


def generate_meta_information() -> ExperimentMetaInfo:
    """
    Generate best-effort experiment meta information without requiring additional input.

    Captures:
    - execution: timestamps + argv/command
    - git: commit/branch/dirty + diff hash (if in a git repo)
    """
    started_at = _utc_now_iso()

    argv = list(sys.argv)
    command = " ".join(argv) if argv else Path(sys.executable).name

    git_info = _git_info_best_effort()

    return ExperimentMetaInfo(
        execution=ExecutionInfo(started_at_utc=started_at, argv=argv, command=command),
        git=git_info,
        output_directory=Path(HydraConfig.get().runtime.output_dir),
    )
