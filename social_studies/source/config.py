import os
from enum import Enum
from functools import cache

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

LOG_FILE_NAME = "stdout.log"
TRACK_FILE_NAME = "experiment_result.jsonl"

CLI_TITLE = "Social Studies"
CLI_SUBTITLE = "Agent Swarm Experiments"

LOG_LEVEL = "INFO"


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
