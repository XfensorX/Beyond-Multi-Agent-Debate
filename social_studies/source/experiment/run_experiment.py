from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from config import results_dir
from data_connectors.mmlu_pro import run_test_set
from decision_schemes.base import ConfigurationOptions
from main_registry import DECISION_SCHEMES, DecisionSchemeName


class DataSource(Enum):
    MMLUPro = "MMLUPro"


class MetaInformation(BaseModel):
    description: str
    number_of_agents: int
    used_rounds: int


class DecisionSchemeStrategyConfig(BaseModel):
    name: DecisionSchemeName
    configuration: ConfigurationOptions

    @field_validator("configuration", mode="after")
    @classmethod
    def parse_correct_configuration(
        cls, value: ConfigurationOptions, info: ValidationInfo
    ) -> str:
        model_cls = DECISION_SCHEMES.get(info.data["name"])
        return model_cls.configuration_parameters.model_validate(
            value.model_dump()  # model_dump such that extra fields are used
        )


class ExperimentConfig(BaseModel):
    name: str
    # TODO: results directory structuring
    results_subdir: Path
    data: DataSource
    strategy: DecisionSchemeStrategyConfig
    meta_information: MetaInformation


class ExperimentInfo(BaseModel):
    results_dir: Path


def run_experiment(experiment_config: ExperimentConfig) -> ExperimentInfo:
    try:
        decision_scheme = DECISION_SCHEMES.get(experiment_config.strategy.name)
    except KeyError:
        raise NotImplementedError(
            f"Strategy '{experiment_config.strategy}' not implemented or registered."
        )

    try:
        match experiment_config.data:
            case DataSource.MMLUPro:
                run_test_set(
                    decision_scheme(experiment_config.strategy.configuration),
                    experiment_config.name,
                )
            case _:
                raise NotImplementedError(
                    f"{experiment_config.data} is not implemented."
                )
    except KeyboardInterrupt:
        # TODO: better logging
        print("Keyboard Interrupt detected. Ending early, but gracefully")
    return ExperimentInfo(
        results_dir=results_dir() / Path(experiment_config.results_subdir)
    )
