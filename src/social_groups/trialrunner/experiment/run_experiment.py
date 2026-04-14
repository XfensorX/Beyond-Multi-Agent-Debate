from __future__ import annotations

import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpcore
import httpx
import huggingface_hub.errors
import langgraph_sdk.errors
import openai

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
from social_groups.trialrunner.utils.phoenix import (
    phoenix_example_span,
    phoenix_log_span,
)

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


MAXIMUM_RETRIES_PER_EXPERIMENT = 3
EXCEPTIONS_TO_RETRY = (
    openai.BadRequestError,
    langgraph_sdk.errors.BadRequestError,
    huggingface_hub.errors.BadRequestError,
    httpx.TimeoutException,
    httpcore.TimeoutException,
    huggingface_hub.errors.InferenceEndpointTimeoutError,
    huggingface_hub.errors.InferenceTimeoutError,
    langgraph_sdk.errors.APITimeoutError,
    openai.APITimeoutError,
)


def execute_experiment(
    data_connector: DataConnector,
    decision_scheme: DecisionScheme,
    tracker: ExperimentTracker,
    execution_config: ExecutionConfig,
):
    def run_single_example(example: Example):
        with phoenix_example_span(
            span_name=str(example.question_id),
            example_id=str(example.question_id),
            attributes={"question": example.model_dump_json()},
        ) as (span, span_info):
            error = None
            example_in = data_connector.prepare_example(example)
            for attempt in range(1, MAXIMUM_RETRIES_PER_EXPERIMENT + 1):
                try:
                    example_out = decision_scheme.run_example(example_in)
                    span.set_attributes(
                        flatten_dict(
                            {
                                "output": example_out.model_dump_json(
                                    exclude={"history"}
                                )
                            },
                            map_to_basic_types=True,
                        )
                    )
                    return example_in, example_out, span_info

                except EXCEPTIONS_TO_RETRY as e:
                    logger.info(f"Received exception {e}. Retrying.")

                    with phoenix_log_span(f"Exception {e} caught. Retrying ..."):
                        time.sleep(min(2**attempt, 10))  # simple exponential backoff

                    if attempt == MAXIMUM_RETRIES_PER_EXPERIMENT:
                        error = e

                except Exception as e:
                    logger.error(
                        f"Received uncaught exception {e}. Cancelling Example immediately."
                    )
                    error = e
                    break

            logger.error(
                f"Received '{str(error)}' after trying {attempt} times, skipping example with ID {example.question_id} ({tracker.out_path.parent})."
            )
            if error:
                span.set_attributes(
                    flatten_dict(
                        {"exception": json.dumps(error, cls=ExceptionEncoder)},
                        map_to_basic_types=True,
                    )
                )
            return example_in, None, span_info

    in_flight_cap = 2 * execution_config.num_workers
    executor = ThreadPoolExecutor(max_workers=execution_config.num_workers)
    futures = set()

    try:
        data_it = iter(data_connector.iterate_data())
        total = data_connector.data_length()

        progress_it = progress_iter(
            range(total),
            total=total,
            desc=f"(#{tracker.out_path.parent.name.split('_')[-1]}) Running Examples ...",
            logger=logger,
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
