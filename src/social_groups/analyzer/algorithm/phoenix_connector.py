import queue
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from social_groups.analyzer.algorithm.utils.common import (
    EXCEPTION_SENTINEL,
    SENTINEL,
    TRANSIENT_ERRORS,
    SpanAttributesFuture,
    logger,
)
from social_groups.analyzer.algorithm.utils.io_operations import (
    get_span_attributes,
    worker_initializer,
)
from social_groups.analyzer.algorithm.utils.package_handling import (
    PendingCollection,
    ReceivePackage,
    SendPackage,
)
from social_groups.analyzer.config import (
    MAX_IDS_PER_REQUEST,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    QUEUE_TIMEOUT,
)


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
