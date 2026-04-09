from __future__ import annotations

import logging
import os
from enum import Enum
from functools import cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from pydantic import BaseModel, ConfigDict, SecretStr

from social_groups.trialrunner.utils import global_config_holder

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
    seed: int | None = None


class BackendInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Backend
    model_name: str


class BackendInfoWithEndpoint(BackendInfo):
    endpoint: str


class DebateAgent(BaseModel):
    params: LLMConfig
    backend: BackendInfo
    number_of_agents: int = 1


logger = logging.getLogger(__name__)


def get_structured_output_method(model_name: str):
    return "json_schema" if "mistral" in model_name else "function_calling"


@cache
def get_llm(
    config: LLMConfig,
    backend: BackendInfo,
    with_thinking: bool | None = None,
):
    api_key = os.getenv(
        global_config_holder.global_hydra_config.execution.backend_api_key_env_vars.get(
            backend.backend, ""
        ),
        "",
    )

    base_url = global_config_holder.global_hydra_config.execution.get_endpoint(backend)

    if config.seed is None:
        raise NotImplementedError(
            "There should always be a seed set on the LLM Config."
        )
    if backend.backend == Backend.L3S_TGI:
        if with_thinking is not None:
            raise NotImplementedError()

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
            seed=config.seed,
        )

        return ChatHuggingFace(llm=model)

    elif backend.backend.value in {
        Backend.vLLMExternal.value,
        Backend.LMSTUDIO.value,
    }:
        if config.top_k is not None or config.typical_p is not None:
            raise NotImplementedError(
                "Does not support these variables: top_k, typical_p"
            )

        if "mistralai" in backend.model_name:
            model = backend.model_name

            if (with_thinking is False) and "Reasoning" in backend.model_name:
                old_model = model
                model = old_model.replace("Reasoning", "Instruct")

                logger.warning(
                    f"Called ChatMistralAI with Reasoning but want no thinking mode,"
                    f" switch from {old_model} to {model} for these kind of requests."
                )

            if (with_thinking is True) and "Instruct" in backend.model_name:
                old_model = model
                model = old_model.replace("Instruct", "Reasoning")

                logger.warning(
                    f"Called ChatMistralAI with Instruct but want thinking mode,"
                    f" switch from {old_model} to {model} for these kind of requests."
                )
            llm = ChatMistralAI(
                model=model,
                base_url=global_config_holder.global_hydra_config.execution.get_endpoint(
                    BackendInfo(backend=backend.backend, model_name=model)
                )
                + "/v1",
                api_key=SecretStr(api_key or "-"),
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
                random_seed=config.seed,
                timeout=None,
            )

            if config.repetition_penalty is not None:
                llm = llm.bind(
                    extra_body={"repetition_penalty": config.repetition_penalty}
                )

            return llm

        elif "Qwen" in backend.model_name:
            llm = ChatQwen(
                model=backend.model_name,
                base_url=base_url + "/v1",
                api_key=SecretStr(api_key or "-"),
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
                seed=config.seed,
            )

            if with_thinking is not None:
                llm = llm.bind(
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": with_thinking}
                    }
                )

            if config.repetition_penalty is not None:
                llm = llm.bind(
                    extra_body={"repetition_penalty": config.repetition_penalty}
                )

            return llm

        else:
            if with_thinking is not None:
                raise NotImplementedError()
            if config.repetition_penalty is not None:
                raise NotImplementedError(
                    "ChatOpenAi does not support these variables: repetition_penalty"
                )
            return ChatOpenAI(
                model=backend.model_name,
                base_url=base_url + "/v1",
                api_key=SecretStr(api_key or "-"),
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
                seed=config.seed,
                # timeout=global_config_holder.global_hydra_config.execution.llm_request_timeout,
                # max_retries=global_config_holder.global_hydra_config.execution.llm_request_max_retries,
            )
    else:
        raise NotImplementedError()


class Backend(Enum):
    INTERWEB = "interweb"
    LMSTUDIO = "LMStudio"
    L3S_TGI = "L3S-TGI"
    vLLMInternal = "vLLMInternal"
    vLLMExternal = "vLLMExternal"
