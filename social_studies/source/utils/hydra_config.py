from __future__ import annotations

from typing import Generic

from decision_schemes.base import ConfigurationOptions
from experiment.main_registry import (
    DECISION_SCHEMES,
    DataConnectorName,
    DecisionSchemeName,
)
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from utils.meta_info import ExperimentMetaInfo


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
