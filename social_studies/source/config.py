from __future__ import annotations

from pathlib import Path


def project_dir():
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    return project_dir() / "configs"


def results_dir() -> Path:
    return project_dir() / "results"
