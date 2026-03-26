import functools
import logging
import os
from pathlib import Path
from typing import Any

from diskcache import Cache

from social_groups.general.types import SpanId

logger = logging.getLogger(__name__)


def make_cache_key(sid: str, endpoint: str):
    return f"{endpoint}:{sid}"


def with_per_span_cache(path: Path):
    """Decorator that caches *individual* SpanId results on disk.

    - The wrapped function still takes a list of span_ids (batch API).
    - Each span_id is cached separately.
    - Invalidate everything: just delete the folder `path/`.
    - Cache hits are served without calling the original function at all.
    """

    os.makedirs(path, exist_ok=True)
    logger.warning("Using Phoenix Cache located at %s.", path)

    with Cache(str(path.absolute())) as cache:

        def decorator(func):
            @functools.wraps(func)
            def wrapper(
                *, span_ids: list[SpanId], phoenix_graphql_endpoint: str
            ) -> dict[SpanId, dict[str, Any]]:
                result: dict[SpanId, dict[str, Any]] = {
                    sid: cache.get(make_cache_key(sid, phoenix_graphql_endpoint))
                    for sid in span_ids
                }

                missing: list[SpanId] = [
                    sid for sid, data in result.items() if data is None
                ]

                # 2. Only call the expensive API for missing spans
                if missing:
                    batch_result = func(
                        span_ids=missing,
                        phoenix_graphql_endpoint=phoenix_graphql_endpoint,
                    )

                    # 3. Store new results in cache + add to final result
                    for sid, data in batch_result.items():
                        cache.set(make_cache_key(sid, phoenix_graphql_endpoint), data)
                        result[sid] = data

                return result

            return wrapper

    return decorator
