import os
from enum import Enum
from functools import cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, ConfigDict

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
    # FIXME: should this be defined somewhere else?
    Backend.INTERWEB: BackendInfo(
        base_url="https://interweb.l3s.uni-hannover.de/v1",
        api_key=os.getenv("INTERWEB_API_KEY"),
    ),
    Backend.LMSTUDIO: BackendInfo(base_url="http://127.0.0.1:1234/v1", api_key=""),
    Backend.L3S_TGI: BackendInfo(base_url="http://localhost:8000/v1", api_key=""),
}


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Backend
    model_name: str
    max_new_tokens: int
    top_k: int
    top_p: float
    typical_p: float
    temperature: float
    repetition_penalty: float


@cache
def get_llm(config: LLMConfig) -> ChatHuggingFace:
    model = HuggingFaceEndpoint(
        task="text-generation",
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        typical_p=config.typical_p,
        # do_sample=False,
        # model=config.model_name, TODO: how to assure model is correct
        endpoint_url=BACKENDS[config.backend].base_url,
        huggingfacehub_api_token=BACKENDS[config.backend].api_key,
    )

    return ChatHuggingFace(llm=model)
