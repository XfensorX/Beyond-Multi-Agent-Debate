from __future__ import annotations

from pathlib import Path
from typing import Generic

from pydantic import BaseModel, field_validator
from concurrent.futures import ThreadPoolExecutor

from pydantic import Field
from pydantic_core.core_schema import ValidationInfo

from data_connectors.base import DataConnector
from decision_schemes.base import DecisionScheme, ConfigurationOptions
from config import results_dir
from experiment.main_registry import (
    DECISION_SCHEMES,
    DATA_CONNECTORS,
    DecisionSchemeName,
    DataConnectorName,
)


class MetaInformation(BaseModel):
    description: str
    number_of_agents: int
    used_rounds: int


class ExperimentConfig(BaseModel):
    name: str
    # TODO: results directory structuring
    results_subdir: Path
    data: DataConnectorName
    strategy: DecisionSchemeStrategyConfig
    meta_information: MetaInformation


class ExperimentInfo(BaseModel):
    results_dir: Path
    # TODO: add stuff


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


class RunConfig(BaseModel):
    num_workers: int = 1
    seed: int = 1
    subset: float = Field(1.0, gt=0.0, le=1.0)


def run_experiment(experiment_config: ExperimentConfig) -> ExperimentInfo:
    try:
        decision_scheme = DECISION_SCHEMES.get(experiment_config.strategy.name)
    except KeyError:
        raise NotImplementedError(
            f"Strategy '{experiment_config.strategy}' not implemented or registered."
        )

    try:
        data = DATA_CONNECTORS.get(experiment_config.data)
    except KeyError:
        raise NotImplementedError(
            f"Data Connector '{experiment_config.data}' not implemented or registered."
        )

    try:  # TODO: put run config into some hydra yaml
        execute_experiment(
            data(),
            decision_scheme(experiment_config.strategy.configuration),
            RunConfig(),
        )

    except KeyboardInterrupt:
        # TODO: better logging
        print("Keyboard Interrupt detected. Ending early, but gracefully")
    return ExperimentInfo(
        results_dir=results_dir() / Path(experiment_config.results_subdir)
    )


def execute_experiment(
    data_connector: DataConnector,
    decision_scheme: DecisionScheme,
    run_config: RunConfig,
):
    # TODO: implement tracker

    # TODO: implement seed and subset
    data_iterator = data_connector.iterate_data()

    def run_single_example(example):
        example_in = data_connector.prepare_example(example)
        return example_in, decision_scheme.run_example(example_in)

    with ThreadPoolExecutor(max_workers=run_config.num_workers) as executor:
        # TODO: move experiment tracking to different class
        with open("results/temporary.json", "w") as output_file:
            for example_input, example_output in executor.map(
                run_single_example, data_iterator
            ):
                # TODO: handle tqdm with logging for clarity
                output_file.write(
                    f"{{in: {example_input.model_dump_json(indent=None)}, out: {example_output.model_dump_json(indent=None)}}} \n"
                )
