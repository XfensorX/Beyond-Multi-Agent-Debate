import logging
import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, List

import httpx
import polars as pl
from rich import print
from typer import Typer

from social_groups.analyzer.models import Answer, Experiment, Question, Run
from social_groups.analyzer.models.base import append_parquet_row
from social_groups.analyzer.utils import (
    create_parquet_writer,
    get_span_attributes,
    iter_jsonl_zst,
    read_hydra_config,
    read_meta_config,
)
from social_groups.directories import (
    MULTIRUN_FINAL_RESULTS_DIR,
    PARQUET_ANALYSIS_DIR,
    RUNS_FINAL_RESULTS_DIR,
    TRACK_FILE_NAME_COMPRESSED,
)
from social_groups.orchestrator.utils.general import run_async
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo
from social_groups.trialrunner.utils.tracking import TrackEntry

app = Typer(no_args_is_help=True)


MAX_PARALLEL_REQUESTS = 5
MAX_IDS_PER_REQUEST = 2
IN_QUEUE_MAXSIZE = 10000
OUT_QUEUE_MAXSIZE = 10000
CHUNK_SIZE = 100
MAX_RETRIES = 1000
PHOENIX_GRAPHQL_ENDPOINT = "http://localhost:6006/graphql"  # TODO: make configurable

QUEUE_TIMEOUT = 10  # seconds

SENTINEL = "___SENTINEL___"
EXCEPTION_SENTINEL = "___EXCEPTION_SENTINEL___"

logger = logging.getLogger("PARQUET BUILDER")


@dataclass(slots=True)
class Package:
    span_id: str
    entry: TrackEntry
    run_id: int

    span_info: dict[str, Any] | None = None


SpanAttributesFuture = Future[dict[str, dict[str, Any]]]


def main_process_loop(
    in_q: queue.Queue,
    out_q: queue.Queue,
):
    retries = 0
    pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS)

    submitted_requests: set[SpanAttributesFuture] = set()
    submitted_batches: dict[SpanAttributesFuture, list[Package]] = {}

    pending: list[Package] = []

    finishing = threading.Event()
    immediate_shutdown = threading.Event()

    def retrieve_from_queue():
        try:
            item = in_q.get(timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            return None
        except TimeoutError:
            return None

        if item == SENTINEL:
            finishing.set()
            return None
        elif item == EXCEPTION_SENTINEL:
            immediate_shutdown.set()
            return None
        else:
            return item

    try:
        while True:
            job = retrieve_from_queue()
            if job is not None:
                pending.append(job)

            if immediate_shutdown.is_set():
                raise RuntimeError(
                    "Retrieved Immediate Shutdown notice. Cancelling all requests."
                )

            if finishing.is_set() and not pending and not submitted_requests:
                out_q.put(SENTINEL)
                break

            if (len(pending) >= MAX_IDS_PER_REQUEST) or (
                finishing.is_set() and pending
            ):
                batch = pending[:MAX_IDS_PER_REQUEST]
                pending = pending[MAX_IDS_PER_REQUEST:]
                future = pool.submit(
                    get_span_attributes,
                    span_ids=[b.span_id for b in batch],
                    phoenix_graphql_endpoint=PHOENIX_GRAPHQL_ENDPOINT,
                )
                submitted_requests.add(future)
                submitted_batches[future] = batch

            if any(task.done() for task in submitted_requests):
                done_tasks, submitted_requests = wait(
                    submitted_requests, timeout=0.1, return_when=FIRST_COMPLETED
                )
                for task in done_tasks:
                    try:
                        span_infos = task.result()
                        for b in submitted_batches.pop(task):
                            b.span_info = span_infos[b.span_id]
                            try:
                                out_q.put(b, timeout=QUEUE_TIMEOUT)
                            except queue.Full:
                                raise RuntimeError(
                                    "The Main Process does not empty the Queue fast enough."
                                )

                    except (httpx.ConnectTimeout, httpx.ReadTimeout):
                        if retries < MAX_RETRIES:
                            logger.error(
                                "Connection error, trying to resubmit. (Do you have connection to phoenix graphql endpoint?)"
                            )
                            batch = submitted_batches.pop(task)
                            new_future = pool.submit(
                                get_span_attributes,
                                span_ids=[b.span_id for b in batch],
                                phoenix_graphql_endpoint=PHOENIX_GRAPHQL_ENDPOINT,
                            )
                            submitted_requests.add(new_future)
                            submitted_batches[new_future] = batch

                        else:
                            logger.error(
                                f"Batch failed after {MAX_RETRIES} retries; giving up.",
                            )
                            raise

                    except Exception as e:
                        logger.error("Unknown error, stopping procedure.")
                        raise RuntimeError(
                            f"Did not correctly handle {e} in main process loop"
                        ) from e

    except Exception as e:
        out_q.put(e)
        raise
    finally:
        logger.info("Trying shutting down Main Process Loop..")
        pool.shutdown(cancel_futures=True, wait=True)
        logger.info("Shut down Main Process Loop..")


def drain_results_nonblocking(out_q: queue.Queue, buffer: List[Package]) -> None:
    for _ in range(MAX_IDS_PER_REQUEST):
        try:
            msg = out_q.get_nowait()
        except queue.Empty:
            return

        if isinstance(msg, Exception):
            raise msg

        if msg == SENTINEL:
            out_q.put(SENTINEL)
            return

        buffer.append(msg)


ExperimentName = str


def read_experiment_paths() -> dict[ExperimentName, list[Path]]:
    project_paths_per_experiment: dict[ExperimentName, list[Path]] = defaultdict(list)

    for experiment_dir in RUNS_FINAL_RESULTS_DIR.iterdir():
        for run_dir in experiment_dir.iterdir():
            project_paths_per_experiment[experiment_dir.name].append(run_dir)

    for experiment_dir in MULTIRUN_FINAL_RESULTS_DIR.iterdir():
        for run_dir in experiment_dir.iterdir():
            for sub_run_dir in run_dir.iterdir():
                if not sub_run_dir.is_dir():
                    continue  # hydra multirun config files

                project_paths_per_experiment[experiment_dir.name].append(sub_run_dir)

    return project_paths_per_experiment


async def build_parquet_files(output_directory: Path):
    models = [Question, Answer, Run, Experiment]

    writers = {
        model: create_parquet_writer(
            output_directory / f"{model.__name__}.parquet", model.get_polars_schema()
        )
        for model in models
    }
    id_generators = {model: count() for model in models}

    # maps from run_id to configs
    run_configs: dict[int, MainConfig] = {}
    run_meta_infos: dict[int, ExperimentMetaInfo] = {}

    seen_questions: dict[
        bytes, int
    ] = {}  # tracks question_hashes and respective question_ids to track duplicate questions

    buffer: list[Package] = []

    def try_flush(buf: list[Package], force: bool = False) -> None:
        if not buf or (len(buffer) < CHUNK_SIZE and not force):
            return

        logger.warning(f"flush buffer with {len(buf)} items")

        question_items = [
            Question.from_raw_data(
                assigned_id=next(id_generators[Question]),
                entry=b.entry,
                hydra_config=run_configs[b.run_id],
                span_attributes=b.span_info,
                meta_info=run_meta_infos[b.run_id],
            )
            for b in buf
        ]

        question_ids = []
        new_questions = []

        for q in question_items:
            q_hash = q.get_id_independent_hash()

            if q_hash in seen_questions:
                _id = seen_questions[q_hash]
            else:
                _id = q.question_id
                seen_questions[q_hash] = _id
                new_questions.append(q)

            question_ids.append(_id)

        writers[Question].write_table(Question.create_parquet_table(new_questions))

        answer_items = [
            Answer.from_raw_data(
                assigned_id=next(id_generators[Answer]),
                run_id=b.run_id,
                entry=b.entry,
                hydra_config=run_configs[b.run_id],
                question_id=q_id,
                meta_info=run_meta_infos[b.run_id],
            )
            for b, q_id in zip(buf, question_ids)
        ]

        writers[Answer].write_table(Answer.create_parquet_table(answer_items))

        buf.clear()

    in_q: queue.Queue = queue.Queue(maxsize=IN_QUEUE_MAXSIZE)
    out_q: queue.Queue = queue.Queue(maxsize=OUT_QUEUE_MAXSIZE)

    project_paths_per_experiment = read_experiment_paths()

    thread = threading.Thread(target=main_process_loop, args=(in_q, out_q))
    thread.start()

    try:
        for experiment_name, project_paths in project_paths_per_experiment.items():
            experiment_id = next(id_generators[Experiment])

            append_parquet_row(
                writers[Experiment],
                Experiment.from_raw_data(
                    assigned_id=experiment_id, experiment_name=experiment_name
                ),
            )

            for project_path in project_paths:
                print(f"Reading: {project_path}")

                run_id = next(id_generators[Run])

                run_configs[run_id] = read_hydra_config(project_path)
                run_meta_infos[run_id] = read_meta_config(project_path)

                append_parquet_row(
                    writers[Run],
                    Run.from_raw_data(
                        assigned_id=run_id,
                        hydra_config=run_configs[run_id],
                        experiment_id=experiment_id,
                        meta_info=run_meta_infos[run_id],
                    ),
                )

                for item in iter_jsonl_zst(project_path / TRACK_FILE_NAME_COMPRESSED):
                    entry = TrackEntry.model_validate(item)
                    in_q.put(
                        Package(
                            span_id=entry.phoenix_span_info.span_id_hex,
                            run_id=run_id,
                            entry=entry,
                            span_info=None,
                        )
                    )

                drain_results_nonblocking(out_q, buffer)
                try_flush(buffer)

        in_q.put(SENTINEL)

        while (msg := out_q.get()) != SENTINEL:
            if isinstance(msg, Exception):
                raise msg

            buffer.append(msg)
            try_flush(buffer)

        try_flush(buffer, force=True)

    except Exception:
        in_q.put(EXCEPTION_SENTINEL)
        raise

    finally:
        thread.join()

        for writer in writers.values():
            writer.close()

    if (
        pl.read_parquet(writers[Question].where)
        .drop("question_id")
        .is_duplicated()
        .any()
    ):
        logger.error(
            "There was a hash collision in the Questions. You have to save the individual ones and handle collisions properly."
        )
        raise NotImplementedError()


@app.command("parse", help="Parse results to produce parquet files.")
@run_async
async def produce_parquet():
    print("Producing Parquet files ...")

    os.makedirs(PARQUET_ANALYSIS_DIR, exist_ok=True)
    await build_parquet_files(PARQUET_ANALYSIS_DIR)


@app.command("show", help="Show nothing.")
def show_results():
    print("There is nothing to show.")


def main():
    app()


if __name__ == "__main__":
    main()
