from __future__ import annotations

import logging
from pathlib import Path
from typing import Generic

from pydantic import BaseModel, field_validator
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from pydantic import Field
from pydantic_core.core_schema import ValidationInfo

from data_connectors.base import DataConnector
from decision_schemes.base import DecisionScheme, ConfigurationOptions
from experiment.main_registry import (
    DECISION_SCHEMES,
    DATA_CONNECTORS,
    DecisionSchemeName,
    DataConnectorName,
)
from utils.logging import progress_iter
from utils.tracking import ExperimentTracker, TrackEntry

logger = logging.getLogger(__name__)


class MetaInformation(BaseModel):
    description: str
    number_of_agents: int
    used_rounds: int


class ExperimentConfig(BaseModel):
    name: str
    data: DataConnectorName
    strategy: DecisionSchemeStrategyConfig
    meta_information: MetaInformation


class ExperimentInfo(BaseModel):
    output_directory: Path


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


class ExecutionConfig(BaseModel):
    num_workers: int = Field(ge=1)
    phoenix_graphql_url: str
    phoenix_server_url: str


def run_experiment(
    experiment_config: ExperimentConfig,
    run_config: ExecutionConfig,
    info: ExperimentInfo,
):
    try:
        logger.info(f"Building Strategy {experiment_config.strategy.name}")
        decision_scheme = DECISION_SCHEMES.get(experiment_config.strategy.name)(
            experiment_config.strategy.configuration
        )
    except KeyError:
        raise NotImplementedError(
            f"Strategy '{experiment_config.strategy}' not implemented or registered."
        )

    try:
        logger.info(f"Building Data {experiment_config.data}")
        data = DATA_CONNECTORS.get(experiment_config.data)()
    except KeyError:
        raise NotImplementedError(
            f"Data Connector '{experiment_config.data}' not implemented or registered."
        )

    try:
        with ExperimentTracker(info.output_directory) as tracker:
            execute_experiment(
                data,
                decision_scheme,
                run_config,
                tracker,
            )

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected. Ending early, but gracefully")


def execute_experiment(
    data_connector: DataConnector,
    decision_scheme: DecisionScheme,
    execution_config: ExecutionConfig,
    tracker: ExperimentTracker,
):
    # TODO: implement seed and subset

    def run_single_example(example):
        example_in = data_connector.prepare_example(example)
        return example_in, decision_scheme.run_example(example_in)

    in_flight_cap = 2 * execution_config.num_workers
    executor = ThreadPoolExecutor(max_workers=execution_config.num_workers)
    futures = set()

    try:
        data_it = iter(  # TODO: this tracks started tasks, not finished ones, should be changed
            progress_iter(
                data_connector.iterate_data(),
                total=data_connector.data_length(),
                desc="Running Examples ...",
                logger=logger,
            )
        )

        for _ in range(in_flight_cap):
            futures.add(executor.submit(run_single_example, next(data_it)))

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                example_input, example_output = fut.result()
                tracker.write_line(
                    TrackEntry(input=example_input, output=example_output)
                )

                try:
                    futures.add(executor.submit(run_single_example, next(data_it)))
                except StopIteration:
                    pass

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected. Collecting Workers...")
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    finally:
        executor.shutdown(wait=True, cancel_futures=True)
