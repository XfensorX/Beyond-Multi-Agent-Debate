from __future__ import annotations

import os
from enum import Enum
from functools import cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, ConfigDict

from social_groups.trialrunner.utils import global_config_holder

LOG_FILE_NAME = "stdout.log"
TRACK_FILE_NAME = "experiment_result.jsonl"

CLI_TITLE = "Social Studies"
CLI_SUBTITLE = "Agent Swarm Experiments"

LOG_LEVEL = "INFO"


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_new_tokens: int
    top_k: int
    top_p: float
    typical_p: float
    temperature: float
    repetition_penalty: float


class BackendInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Backend
    model_name: str


class BackendInfoWithEndpoint(BackendInfo):
    endpoint: str


@cache
def get_llm(config: LLMConfig, backend: BackendInfo) -> ChatHuggingFace:
    if backend.backend != Backend.L3S_TGI:
        raise NotImplementedError()
        # FIXME: if additional backends need to be added, this method has to change

    if backend.model_name != "Qwen/Qwen2.5-0.5B-Instruct":
        raise NotImplementedError()  # see below

    api_key = os.getenv(
        global_config_holder.global_hydra_config.execution.backend_api_key_env_vars.get(
            backend.backend, ""
        ),
        "",
    )

    base_url = global_config_holder.global_hydra_config.execution.get_endpoint(backend)

    model = HuggingFaceEndpoint(
        model="ignored",  # this is ignored by TGI
        task="text-generation",
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        typical_p=config.typical_p,
        endpoint_url=base_url,
        huggingfacehub_api_token=api_key,
    )

    return ChatHuggingFace(llm=model)


class Backend(Enum):
    INTERWEB = "interweb"
    LMSTUDIO = "LMStudio"
    L3S_TGI = "L3S-TGI"
