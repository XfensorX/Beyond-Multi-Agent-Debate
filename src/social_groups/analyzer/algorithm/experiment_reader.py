import asyncio
import concurrent.futures
import logging
import queue
import threading
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import count
from pathlib import Path
from typing import List

import polars as pl
from rich.live import Live
from rich.table import Table

from social_groups.analyzer.algorithm.phoenix_connector import main_process_loop
from social_groups.analyzer.algorithm.utils.answer_info_cache import AnswerInfoCache
from social_groups.analyzer.algorithm.utils.common import (
    EXCEPTION_SENTINEL,
    SENTINEL,
    ExperimentName,
    logger,
)
from social_groups.analyzer.algorithm.utils.io_operations import (
    create_parquet_writer,
    read_hydra_config,
    read_meta_config,
)
from social_groups.analyzer.algorithm.utils.lazy_dictionary import LazyDict
from social_groups.analyzer.algorithm.utils.package_handling import (
    ReceivePackage,
    SendPackage,
)
from social_groups.analyzer.config import (
    CHUNK_SIZE,
    IN_QUEUE_MAXSIZE,
    OUT_QUEUE_MAXSIZE,
    PARALLEL_FILE_WRITES,
)
from social_groups.analyzer.models import Answer, Experiment, Question, Run
from social_groups.directories import (
    MULTIRUN_FINAL_RESULTS_DIR,
    RUNS_FINAL_RESULTS_DIR,
    TRACK_FILE_NAME_COMPRESSED,
)
from social_groups.general.tracking import TrackEntry, iter_jsonl_zst
from social_groups.general.utils.urls import replace_host_with_localhost

logger.setLevel(logging.DEBUG)


def drain_results_nonblocking(out_q: queue.Queue, buffer: List[ReceivePackage]) -> None:
    for _ in range(OUT_QUEUE_MAXSIZE):
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


async def handle_jsonl_file(
    project_path: Path,
    run_id: int,
    in_q: queue.Queue,
    graphql_endpoint: str,
    global_answer_infos_cache: AnswerInfoCache,
) -> int:
    """Returns answer_infos"""
    answer_infos_cache = AnswerInfoCache()
    put_in_queue = 0
    packages = []
    for item in iter_jsonl_zst(project_path / TRACK_FILE_NAME_COMPRESSED):
        entry = TrackEntry.model_validate(item)

        if entry.phoenix_span_info.span_id_hex is None:
            raise NotImplementedError("The SpanID should be always set.")

        packages.append(
            SendPackage(
                id=await answer_infos_cache.register_answer_info(entry, run_id),
                run_id=run_id,
                span_id=entry.phoenix_span_info.span_id_hex,
                phoenix_graphql_endpoint=graphql_endpoint,
            )
        )

    await global_answer_infos_cache.update(answer_infos_cache)

    for p in packages:
        in_q.put(p)
        put_in_queue += 1

    return put_in_queue


ExecutingFileFuture = asyncio.Future[int]


async def build_parquet_files(output_directory: Path):
    logger.info("Building parquet files")

    total_put_in_queue = 0
    total_flushed = 0

    models = [Question, Answer, Run, Experiment]

    id_generators = {model: count() for model in models}

    run_configs = LazyDict(read_hydra_config, max_cached=50)
    run_meta_infos = LazyDict(read_meta_config, max_cached=50)

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

    in_q: queue.Queue = queue.Queue(maxsize=IN_QUEUE_MAXSIZE)
    out_q: queue.Queue = queue.Queue(maxsize=OUT_QUEUE_MAXSIZE)

    async def try_flush(
        buf: list[ReceivePackage], force: bool = False
    ) -> list[ReceivePackage]:
        nonlocal total_flushed

        while len(buf) >= CHUNK_SIZE or (len(buf) > 0 and force):
            flushed = 0

            size = min(len(buf), CHUNK_SIZE)
            chunk = buf[:size]
            buf = buf[size:]

            new_questions = []

            question_ids = []
            entries = []
            run_ids = []

            for b in chunk:
                b_entry = await answer_infos_cache.retrieve_answer_info(b.id, b.run_id)

                q = Question.from_raw_data(
                    assigned_id=next(id_generators[Question]),
                    entry=b_entry,
                    hydra_config=run_configs[b.run_id],
                    span_attributes=b.span_info,
                    meta_info=run_meta_infos[b.run_id],
                )

                q_hash = q.get_id_independent_hash()

                if q_hash in seen_questions:
                    q_id = seen_questions[q_hash]
                else:
                    q_id = q.question_id
                    seen_questions[q_hash] = q_id
                    new_questions.append(q)

                entries.append(b_entry)
                run_ids.append(b.run_id)
                question_ids.append(q_id)
                flushed += 1

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

            total_flushed += flushed

            if flushed == 0:
                raise RuntimeError(
                    "Could not flush any of the buffer contents. The answer infos are not aligned."
                )
        return buf

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

    executing_files: set[ExecutingFileFuture] = set()

    async def handle_done_files(
        _executing_files: set[ExecutingFileFuture], wait_for_all: bool = False
    ) -> set[ExecutingFileFuture]:
        nonlocal total_put_in_queue

        if wait_for_all:
            done_file, new_executing_files = await asyncio.wait(_executing_files)
        else:
            done_file, new_executing_files = await asyncio.wait(
                _executing_files, timeout=0.01, return_when=asyncio.FIRST_COMPLETED
            )

        for future in done_file:
            put_in_q = future.result()
            total_put_in_queue += put_in_q

        return new_executing_files

    handle_jsonl_file_semaphore = asyncio.Semaphore(PARALLEL_FILE_WRITES)

    with Live(get_renderable=make_process_status_table) as live:
        try:
            for (
                experiment_name,
                project_paths,
            ) in project_paths_per_experiment.items():
                print(f"Reading: {experiment_name}")

                experiment_id = next(id_generators[Experiment])

                experiments.append(
                    Experiment.from_raw_data(
                        assigned_id=experiment_id, experiment_name=experiment_name
                    )
                )

                for project_path in project_paths:
                    run_id = next(id_generators[Run])

                    run_configs.register(run_id, project_path)
                    run_meta_infos.register(run_id, project_path)

                    runs.append(
                        Run.from_raw_data(
                            assigned_id=run_id,
                            hydra_config=run_configs[run_id],
                            experiment_id=experiment_id,
                            meta_info=run_meta_infos[run_id],
                        )
                    )

                    async def _wrapped_handle_jsonl_file(
                        _project_path: Path,
                        _run_id: int,
                        _in_q: queue.Queue,
                        _graphql_endpoint: str,
                        _global_answer_infos_cache: AnswerInfoCache,
                    ):
                        async with handle_jsonl_file_semaphore:
                            return await handle_jsonl_file(
                                _project_path,
                                _run_id,
                                _in_q,
                                _graphql_endpoint,
                                _global_answer_infos_cache,
                            )

                    executing_files.add(
                        asyncio.create_task(
                            _wrapped_handle_jsonl_file(
                                project_path,
                                run_id,
                                in_q,
                                replace_host_with_localhost(
                                    run_configs[run_id].execution.phoenix_server_url
                                    + "/graphql"
                                ),
                                answer_infos_cache,
                            )
                        )
                    )

                    executing_files = await handle_done_files(executing_files)
                    drain_results_nonblocking(out_q, buffer)
                    buffer = await try_flush(buffer)

            if runs:
                writers[Run].write_table(Run.create_parquet_table(runs))

            if experiments:
                writers[Experiment].write_table(
                    Experiment.create_parquet_table(experiments)
                )

            sentinel_sent = False
            print("Reached the End")
            while True:
                if executing_files:
                    executing_files = await handle_done_files(executing_files)
                elif not sentinel_sent:
                    in_q.put(SENTINEL)
                    sentinel_sent = True

                drain_results_nonblocking(out_q, buffer)

                if len(buffer) > CHUNK_SIZE:
                    buffer = await try_flush(buffer)
                else:
                    msg = out_q.get()

                    if msg == SENTINEL:
                        break
                    if isinstance(msg, Exception):  # This has to be removed
                        raise msg

                    buffer.append(msg)

            buffer = await try_flush(buffer, force=True)

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
        print(
            "There was a hash collision in the Questions. You have to save the individual ones, "
            "not just hashes and handle collisions properly."
        )
        raise NotImplementedError()
