from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from data_connectors.base import DataConnector
from decision_schemes.base import DecisionScheme
from experiment.main_registry import (
    DECISION_SCHEMES,
    DATA_CONNECTORS,
)
from utils.hydra_config import (
    ExecutionConfig,
    MainConfig,
)
from utils.logging import progress_iter
from utils.tracking import ExperimentTracker, TrackEntry

logger = logging.getLogger(__name__)


def run_experiment(config: MainConfig):
    try:
        logger.info(f"Building Strategy {config.experiment.strategy.name}")
        decision_scheme = DECISION_SCHEMES.get(config.experiment.strategy.name)(
            config.experiment.strategy.configuration
        )
    except KeyError:
        raise NotImplementedError(
            f"Strategy '{config.experiment.strategy}' not implemented or registered."
        )

    try:
        logger.info(f"Building Data {config.experiment.data}")
        data = DATA_CONNECTORS.get(config.experiment.data)()
    except KeyError:
        raise NotImplementedError(
            f"Data Connector '{config.experiment.data}' not implemented or registered."
        )

    try:
        with ExperimentTracker(config.meta_info.output_directory) as tracker:
            execute_experiment(data, decision_scheme, tracker, config.execution)

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected. Ending early, but gracefully")


def execute_experiment(
    data_connector: DataConnector,
    decision_scheme: DecisionScheme,
    tracker: ExperimentTracker,
    execution_config: ExecutionConfig,
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
