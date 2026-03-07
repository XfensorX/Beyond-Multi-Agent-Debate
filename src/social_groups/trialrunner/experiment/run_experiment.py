from __future__ import annotations

import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from social_groups.general.tracking import ExperimentTracker, TrackEntry
from social_groups.general.utils.standard_library import flatten_dict
from social_groups.trialrunner.data_connectors.base import DataConnector, Example
from social_groups.trialrunner.decision_schemes.base import DecisionScheme
from social_groups.trialrunner.experiment.main_registry import (
    DATA_CONNECTORS,
    DECISION_SCHEMES,
)
from social_groups.trialrunner.utils.exception_handling import ExceptionEncoder
from social_groups.trialrunner.utils.hydra_config import (
    ExecutionConfig,
    MainConfig,
)
from social_groups.trialrunner.utils.logging import progress_iter
from social_groups.trialrunner.utils.phoenix import phoenix_example_span

logger = logging.getLogger(__name__)


def run_experiment(config: MainConfig, output_directory: Path):
    try:
        logger.info(f"Building Strategy {config.experiment.strategy.name}")
        decision_scheme = DECISION_SCHEMES.get(config.experiment.strategy.name)(
            config.experiment.strategy.configuration
        )
        logger.info("... Done Building Strategy")
    except KeyError:
        raise NotImplementedError(
            f"Strategy '{config.experiment.strategy}' not implemented or registered."
        )

    try:
        logger.info(f"Building Data {config.experiment.data}")
        data = DATA_CONNECTORS.get(config.experiment.data)()
        logger.info("... Done Building Data")
    except KeyError:
        raise NotImplementedError(
            f"Data Connector '{config.experiment.data}' not implemented or registered."
        )

    try:
        with ExperimentTracker(output_directory) as tracker:
            execute_experiment(data, decision_scheme, tracker, config.execution)

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected. Ending early, but gracefully")


def execute_experiment(
    data_connector: DataConnector,
    decision_scheme: DecisionScheme,
    tracker: ExperimentTracker,
    execution_config: ExecutionConfig,
):
    def run_single_example(example: Example):
        with phoenix_example_span(
            example_id=example.question_id,
            attributes={"question": example.model_dump_json()},
        ) as (
            span,
            span_info,
        ):
            example_in = data_connector.prepare_example(example)
            try:
                example_out = decision_scheme.run_example(example_in)
                span.set_attributes(
                    flatten_dict(
                        {"output": example_out.model_dump_json(exclude={"history"})},
                        map_to_basic_types=True,
                    )
                )
            except Exception as e:
                logger.error(
                    f"Received '{str(e)}', skipping example with ID {example.question_id}."
                )
                span.set_attributes(
                    flatten_dict(
                        {"exception": json.dumps(e, cls=ExceptionEncoder)},
                        map_to_basic_types=True,
                    )
                )
                example_out = None

            return example_in, example_out, span_info

    in_flight_cap = 2 * execution_config.num_workers
    executor = ThreadPoolExecutor(max_workers=execution_config.num_workers)
    futures = set()

    try:
        data_it = iter(data_connector.iterate_data())
        total = data_connector.data_length()

        progress_it = progress_iter(
            range(total), total=total, desc="Running Examples ...", logger=logger
        )

        for _ in range(in_flight_cap):
            try:
                futures.add(executor.submit(run_single_example, next(data_it)))
            except StopIteration:
                break
            time.sleep(
                execution_config.seconds_between_start_of_new_experiment
            )  # Prevent Bursting the LLM Server with a lot of workers.

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                example_input, example_output, phoenix_span_info = fut.result()

                tracker.write_line(
                    TrackEntry(
                        input=example_input,
                        output=example_output,
                        phoenix_span_info=phoenix_span_info,
                    )
                )

                try:
                    next(progress_it)
                except StopIteration:
                    pass

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

    except Exception as e:
        logger.error("Received Unhandled exception: %s", e)
        raise

    finally:
        logger.info("Shutting Down the experiment executor. Collecting Workers...")
        executor.shutdown(wait=True, cancel_futures=True)
