import logging
import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import count
from pathlib import Path
from typing import Any, List

import httpx
import polars as pl
import typer
from rich import print
from rich.live import Live
from rich.table import Table
from typer import Typer

from social_groups.analyzer.answer_info_cache import AnswerInfoCache
from social_groups.analyzer.models import Answer, Experiment, Question, Run
from social_groups.analyzer.package_handling import (
    PendingCollection,
    ReceivePackage,
    SendPackage,
)
from social_groups.analyzer.sync_tex import (
    git_commit_and_push,
    sync_exact_with_confirmation,
)
from social_groups.analyzer.utils import (
    create_parquet_writer,
    get_span_attributes,
    read_hydra_config,
    read_meta_config,
    worker_initializer,
)
from social_groups.directories import (
    DAGSTER_REPORT_DIR,
    MULTIRUN_FINAL_RESULTS_DIR,
    PARQUET_ANALYSIS_DIR,
    RUNS_FINAL_RESULTS_DIR,
    TEX_PROJECT_REPORT_DIR,
    TRACK_FILE_NAME_COMPRESSED,
)
from social_groups.general.tracking import TrackEntry, iter_jsonl_zst
from social_groups.general.utils.urls import replace_host_with_localhost
from social_groups.orchestrator.utils.general import run_async
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo

app = Typer(no_args_is_help=True)


MAX_PARALLEL_REQUESTS = 10
MAX_IDS_PER_REQUEST = 20
IN_QUEUE_MAXSIZE = 10000
OUT_QUEUE_MAXSIZE = 10000
CHUNK_SIZE = 1000
MAX_RETRIES = 2**30

QUEUE_TIMEOUT = 10  # seconds

SENTINEL = "___SENTINEL___"
EXCEPTION_SENTINEL = "___EXCEPTION_SENTINEL___"

logger = logging.getLogger("PARQUET BUILDER")
logger.setLevel(logging.DEBUG)

TRANSIENT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)

SpanAttributesFuture = Future[dict[str, dict[str, Any]]]


def main_process_loop(in_q: queue.Queue, out_q: queue.Queue):
    retries = 0
    pool = ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_REQUESTS, initializer=worker_initializer
    )

    currently_pooled_requests = 0

    submitted_requests: set[SpanAttributesFuture] = set()
    submitted_batches: dict[SpanAttributesFuture, list[SendPackage]] = {}

    pending = PendingCollection()

    finishing = threading.Event()
    immediate_shutdown = threading.Event()

    def retrieve_from_queue() -> None | SendPackage:
        try:
            item = in_q.get_nowait()
        except queue.Empty:
            return None
        # except TimeoutError:
        #     return None

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
            while pending.total_pending < MAX_PARALLEL_REQUESTS * MAX_IDS_PER_REQUEST:
                if (job := retrieve_from_queue()) is None:
                    break
                pending.add(job)

            if immediate_shutdown.is_set():
                raise RuntimeError(
                    "Retrieved Immediate Shutdown notice. Cancelling all requests."
                )

            if finishing.is_set() and pending.is_empty() and not submitted_requests:
                out_q.put(SENTINEL)
                break

            done_tasks, submitted_requests = wait(
                submitted_requests,
                timeout=None if (finishing.is_set() and pending.is_empty()) else 0.0,
                return_when=FIRST_COMPLETED,
            )

            for task in done_tasks:
                currently_pooled_requests -= 1
                try:
                    span_infos = task.result()
                    for b in submitted_batches.pop(task):
                        new_b = ReceivePackage(id=b.id, span_info=span_infos[b.span_id])
                        try:
                            out_q.put(new_b, timeout=QUEUE_TIMEOUT)
                        except queue.Full:
                            raise RuntimeError(
                                "The Main Process does not empty the Queue fast enough."
                            )

                except TRANSIENT_ERRORS as e:
                    if retries < MAX_RETRIES:
                        logger.error(
                            f"{e}, trying to resubmit. (Do you have connection to phoenix graphql endpoint?)"
                        )
                        batch = submitted_batches.pop(task)
                        new_future = pool.submit(
                            get_span_attributes,
                            span_ids=[b.span_id for b in batch],
                            phoenix_graphql_endpoint=(
                                batch[0].phoenix_graphql_endpoint
                            ),
                        )
                        currently_pooled_requests += 1

                        submitted_requests.add(new_future)
                        submitted_batches[new_future] = batch
                        retries += 1
                        if retries % 100 == 0 and retries > 0:
                            logger.error(
                                f"Total of {retries} retries reached. (Will cancel at {MAX_RETRIES})"
                            )

                    else:
                        logger.error(
                            f"Batch failed after {MAX_RETRIES} retries; giving up.",
                        )
                        raise e

                except Exception as e:
                    logger.error(f"Unknown error, stopping procedure. ({e})")
                    raise RuntimeError(
                        f"Did not correctly handle {e} in main process loop"
                    ) from e

            if (pending.current_largest_batch_size() >= MAX_IDS_PER_REQUEST) or (
                finishing.is_set() and not pending.is_empty()
            ):
                while currently_pooled_requests < 3 * MAX_PARALLEL_REQUESTS:
                    batch = pending.get_batch(MAX_IDS_PER_REQUEST)
                    if not batch:
                        break

                    future = pool.submit(
                        get_span_attributes,
                        span_ids=[b.span_id for b in batch],
                        phoenix_graphql_endpoint=batch[0].phoenix_graphql_endpoint,
                    )
                    currently_pooled_requests += 1

                    submitted_requests.add(future)
                    submitted_batches[future] = batch

    except Exception as e:
        out_q.put(e)
        raise
    finally:
        logger.info("Trying shutting down Main Process Loop..")
        pool.shutdown(cancel_futures=True, wait=True)
        logger.info("Shut down Main Process Loop..")


def drain_results_nonblocking(out_q: queue.Queue, buffer: List[ReceivePackage]) -> None:
    while True:
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
        if experiment_dir == ".DS_Store":
            continue
        for run_dir in experiment_dir.iterdir():
            project_paths_per_experiment[experiment_dir.name].append(run_dir)

    for experiment_dir in MULTIRUN_FINAL_RESULTS_DIR.iterdir():
        if experiment_dir == ".DS_Store":
            continue
        for run_dir in experiment_dir.iterdir():
            if run_dir == ".DS_Store":
                continue
            for sub_run_dir in run_dir.iterdir():
                if not sub_run_dir.is_dir():
                    continue  # hydra multirun config files

                project_paths_per_experiment[experiment_dir.name].append(sub_run_dir)

    return project_paths_per_experiment


async def build_parquet_files(output_directory: Path):
    total_put_in_queue = 0
    total_flushed = 0

    models = [Question, Answer, Run, Experiment]

    id_generators = {model: count() for model in models}

    # maps from run_id to configs
    run_configs: dict[int, MainConfig] = {}
    run_meta_infos: dict[int, ExperimentMetaInfo] = {}

    answer_infos_cache = AnswerInfoCache()

    seen_questions: dict[
        bytes, int
    ] = {}  # tracks question_hashes and respective question_ids to track duplicate questions

    buffer: list[ReceivePackage] = []

    writers = {
        model: create_parquet_writer(
            output_directory / f"{model.__name__}.parquet", model.get_polars_schema()
        )
        for model in models
    }

    def try_flush(
        buf: list[ReceivePackage], force: bool = False
    ) -> list[ReceivePackage]:
        nonlocal total_flushed

        while len(buf) >= CHUNK_SIZE or (len(buf) > 0 and force):
            size = min(len(buf), CHUNK_SIZE)
            chunk = buf[:size]
            buf = buf[size:]

            new_questions = []

            question_ids = []
            entries = []
            run_ids = []

            for b in chunk:
                b_entry, b_run_id = answer_infos_cache.retrieve_answer_info(b.id)

                q = Question.from_raw_data(
                    assigned_id=next(id_generators[Question]),
                    entry=b_entry,
                    hydra_config=run_configs[b_run_id],
                    span_attributes=b.span_info,
                    meta_info=run_meta_infos[b_run_id],
                )

                q_hash = q.get_id_independent_hash()

                if q_hash in seen_questions:
                    q_id = seen_questions[q_hash]
                else:
                    q_id = q.question_id
                    seen_questions[q_hash] = q_id
                    new_questions.append(q)

                entries.append(b_entry)
                run_ids.append(b_run_id)
                question_ids.append(q_id)

            if new_questions:
                writers[Question].write_table(
                    Question.create_parquet_table(new_questions)
                )

            answer_items = [
                Answer.from_raw_data(
                    assigned_id=next(id_generators[Answer]),
                    run_id=r,
                    entry=e,
                    hydra_config=run_configs[r],
                    question_id=q_id,
                    meta_info=run_meta_infos[r],
                    span_attributes=b.span_info,
                )
                for b, e, r, q_id in zip(chunk, entries, run_ids, question_ids)
            ]

            writers[Answer].write_table(Answer.create_parquet_table(answer_items))

            total_flushed += len(chunk)

        return buf

    in_q: queue.Queue = queue.Queue(maxsize=IN_QUEUE_MAXSIZE)
    out_q: queue.Queue = queue.Queue(maxsize=OUT_QUEUE_MAXSIZE)

    project_paths_per_experiment = read_experiment_paths()

    thread = threading.Thread(target=main_process_loop, args=(in_q, out_q))
    thread.start()

    def make_process_status_table() -> Table:
        table = Table(show_header=False, border_style="dim")
        table.add_row("[b]in_q[/b] size", f"[cyan]{in_q.qsize():>4}[/cyan]")
        table.add_row("[b]out_q[/b] size", f"[cyan]{out_q.qsize():>4}[/cyan]")
        table.add_row(
            "[b]Total Submitted[/b]", f"[green]{total_put_in_queue:>6}[/green]"
        )
        table.add_row("[b]Total Written[/b]", f"[green]{total_flushed:>6}[/green]")
        return table

    runs = []
    experiments = []

    with Live(get_renderable=make_process_status_table) as live:
        try:
            for experiment_name, project_paths in project_paths_per_experiment.items():
                logger.info(f"Reading: {experiment_name}")

                experiment_id = next(id_generators[Experiment])

                experiments.append(
                    Experiment.from_raw_data(
                        assigned_id=experiment_id, experiment_name=experiment_name
                    )
                )

                for project_path in project_paths:
                    run_id = next(id_generators[Run])

                    run_configs[run_id] = read_hydra_config(project_path)
                    run_meta_infos[run_id] = read_meta_config(project_path)

                    runs.append(
                        Run.from_raw_data(
                            assigned_id=run_id,
                            hydra_config=run_configs[run_id],
                            experiment_id=experiment_id,
                            meta_info=run_meta_infos[run_id],
                        )
                    )

                    for item in iter_jsonl_zst(
                        project_path / TRACK_FILE_NAME_COMPRESSED
                    ):
                        entry = TrackEntry.model_validate(item)

                        if entry.phoenix_span_info.span_id_hex is None:
                            raise NotImplementedError(
                                "The SpanID should be always set."
                            )

                        in_q.put(
                            SendPackage(
                                id=answer_infos_cache.register_answer_info(
                                    entry, run_id
                                ),
                                span_id=entry.phoenix_span_info.span_id_hex,
                                phoenix_graphql_endpoint=replace_host_with_localhost(
                                    run_configs[run_id].execution.phoenix_server_url
                                    + "/graphql"
                                ),
                            )
                        )
                        total_put_in_queue += 1
                    if in_q.qsize() / in_q.maxsize > 0.1:
                        drain_results_nonblocking(out_q, buffer)
                        buffer = try_flush(buffer)

            in_q.put(SENTINEL)

            while (msg := out_q.get()) != SENTINEL:
                if isinstance(msg, Exception):
                    raise msg

                buffer.append(msg)
                buffer = try_flush(buffer)

            if runs:
                writers[Run].write_table(Run.create_parquet_table(runs))
            if experiments:
                writers[Experiment].write_table(
                    Experiment.create_parquet_table(experiments)
                )

            buffer = try_flush(buffer, force=True)

        except Exception:
            in_q.put(EXCEPTION_SENTINEL)
            raise

        finally:
            live.update(make_process_status_table())
            thread.join()

            for writer in writers.values():
                writer.close()

    if buffer:
        raise RuntimeError(f"Buffer was not completely emptied. ({len(buffer)})")

    if (
        pl.read_parquet(writers[Question].where)
        .drop("question_id")
        .is_duplicated()
        .any()
    ):
        logger.error(
            "There was a hash collision in the Questions. You have to save the individual ones, "
            "not just hashes and handle collisions properly."
        )
        raise NotImplementedError()


@app.command("parse", help="Parse results to produce parquet files.")
@run_async
async def produce_parquet():
    print("Producing Parquet files ...")

    os.makedirs(PARQUET_ANALYSIS_DIR, exist_ok=True)
    await build_parquet_files(PARQUET_ANALYSIS_DIR)


@app.command(name="sync-tex", help="Sync DAGster reports → university LaTeX project")
def sync_tex(
    from_dir: Path = typer.Argument(
        DAGSTER_REPORT_DIR, help="Source directory (DAGSTER_REPORT_DIR)"
    ),
    to_dir: Path = typer.Argument(
        TEX_PROJECT_REPORT_DIR, help="Target directory (tex.cloud project folder)"
    ),
):
    typer.secho(f"Syncing {from_dir} → {to_dir}", bold=True)
    sync_exact_with_confirmation(from_dir, to_dir)

    typer.secho("\nGit operations in target directory:", bold=True)
    git_commit_and_push(to_dir)


def main():
    app()


if __name__ == "__main__":
    main()
