from __future__ import annotations

import os
from enum import Enum
from functools import cache
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def project_dir():
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    return project_dir() / "configs"


def results_dir() -> Path:
    return project_dir() / "results"


class BackendInfo(BaseModel):
    base_url: str
    api_key: str | None = ""


class Backend(Enum):
    INTERWEB = "interweb"
    LMSTUDIO = "LMStudio"
    L3S_TGI = "L3S-TGI"


BACKENDS: dict[Backend, BackendInfo] = {
    Backend.INTERWEB: BackendInfo(
        base_url="https://interweb.l3s.uni-hannover.de/v1",
        api_key=os.getenv("INTERWEB_API_KEY"),
    ),
    Backend.LMSTUDIO: BackendInfo(base_url="http://127.0.0.1:1234/v1", api_key=""),
    Backend.L3S_TGI: BackendInfo(base_url="http://localhost:8000/v1", api_key=""),
}


@cache
def get_llm(backend: Backend, model_name: str):
    return ChatOpenAI(
        model=model_name,
        base_url=BACKENDS[backend].base_url,
        api_key=BACKENDS[backend].api_key,
    )
