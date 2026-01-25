from __future__ import annotations

from typing import Generic

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from social_groups.trialrunner.config import (
    Backend,
    BackendInfo,
    BackendInfoWithEndpoint,
)
from social_groups.trialrunner.decision_schemes.base import ConfigurationOptions
from social_groups.trialrunner.experiment.main_registry import (
    DECISION_SCHEMES,
    DataConnectorName,
    DecisionSchemeName,
)
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo


class MainConfig(BaseModel):
    experiment: ExperimentConfig
    execution: ExecutionConfig
    meta_info: ExperimentMetaInfo


class ExperimentInformation(BaseModel):
    description: str


class ExperimentConfig(BaseModel):
    name: str
    data: DataConnectorName
    strategy: DecisionSchemeStrategyConfig
    additional_info: ExperimentInformation


class ExecutionConfig(BaseModel):
    num_workers: int = Field(ge=1)
    phoenix_graphql_url: str
    phoenix_server_url: str
    model_backends: list[BackendInfoWithEndpoint]
    backend_api_key_env_vars: dict[Backend, str]

    @field_validator("model_backends")
    def assert_model_backends_unique(
        cls, backends: list[BackendInfoWithEndpoint]
    ) -> list[BackendInfoWithEndpoint]:
        keys = [(m.model_name, m.backend) for m in backends]
        if len(keys) != len(set(keys)):
            from collections import Counter

            dup = [k for k, c in Counter(keys).items() if c > 1]
            raise ValueError(f"Duplicate (model_name, backend) combinations: {dup}")
        return backends

    def get_endpoint(self, info: BackendInfo) -> str:
        for item in self.model_backends:
            if item.backend == info.backend and item.model_name == info.model_name:
                return item.endpoint

        raise KeyError(
            f"No Endpoint found for (model_name={info.model_name!r}, backend={info.backend!r})"
        )


class DecisionSchemeStrategyConfig(BaseModel, Generic[ConfigurationOptions]):
    name: DecisionSchemeName
    configuration: ConfigurationOptions

    @field_validator("configuration", mode="after")
    @classmethod
    def parse_correct_configuration(
        cls, value: ConfigurationOptions, info: ValidationInfo
    ) -> str:
        model_cls = DECISION_SCHEMES.get(info.data["name"])
        config_values = value.model_dump()

        # model_dump such that extra fields are used
        return model_cls(config_values).validate_config(config_values)
