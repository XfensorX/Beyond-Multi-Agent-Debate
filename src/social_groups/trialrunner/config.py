from __future__ import annotations

import os
from enum import Enum
from functools import cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict

from social_groups.trialrunner.utils import global_config_holder

LOG_FILE_NAME = "stdout.log"
META_FILE_NAME = "meta.yaml"
TRACK_FILE_NAME = "experiment_result.jsonl"

CLI_TITLE = "Social Studies"
CLI_SUBTITLE = "Agent Swarm Experiments"

LOG_LEVEL = "INFO"


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_new_tokens: int
    top_k: int | None
    top_p: float | None
    typical_p: float | None
    temperature: float | None
    repetition_penalty: float | None


class BackendInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Backend
    model_name: str


class BackendInfoWithEndpoint(BackendInfo):
    endpoint: str


@cache
def get_llm(config: LLMConfig, backend: BackendInfo) -> ChatHuggingFace | ChatOpenAI:
    api_key = os.getenv(
        global_config_holder.global_hydra_config.execution.backend_api_key_env_vars.get(
            backend.backend, ""
        ),
        "",
    )
    base_url = global_config_holder.global_hydra_config.execution.get_endpoint(backend)

    if backend.backend == Backend.L3S_TGI:
        model = HuggingFaceEndpoint(
            # model="ignored",  # this is ignored by TGI
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

    elif backend.backend == Backend.vLLMExternal:
        if (
            config.top_k is not None
            or config.typical_p is not None
            or config.repetition_penalty is not None
        ):
            raise NotImplementedError(
                "ChatOpenAi does not support these variables: top_k, typical_p, repetition_penalty"
            )
        return ChatOpenAI(
            model=backend.model_name,
            base_url=base_url + "/v1",
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
        )
    else:
        raise NotImplementedError()


class Backend(Enum):
    INTERWEB = "interweb"
    LMSTUDIO = "LMStudio"
    L3S_TGI = "L3S-TGI"
    vLLMInternal = "vLLMInternal"
    vLLMExternal = "vLLMExternal"
