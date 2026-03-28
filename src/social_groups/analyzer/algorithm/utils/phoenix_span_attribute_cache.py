import functools
import logging
import os
from typing import Any

from diskcache import Cache

from social_groups.directories import PHOENIX_CACHE_DIR
from social_groups.general.types import SpanId

logger = logging.getLogger(__name__)


def make_cache_key(sid: str, endpoint: str):
    return f"{endpoint}:{sid}"


os.makedirs(PHOENIX_CACHE_DIR.absolute(), exist_ok=True)
PHOENIX_DISK_CACHE = Cache(
    str(PHOENIX_CACHE_DIR.absolute()),
    # === Durability & reliability changes ===
    sqlite_journal_mode="wal",
    sqlite_synchronous=1,
    size_limit=50 * (2**30),  # 50GB Maximum
    sqlite_cache_size=2**14,  # (~64-128 MB page cache)
)


def with_per_span_cache():
    """Decorator that caches *individual* SpanId results on disk.

    - The wrapped function still takes a list of span_ids (batch API).
    - Each span_id is cached separately.
    - Invalidate everything: just delete the folder `path/`.
    - Cache hits are served without calling the original function at all.
    """

    logger.warning("Using Phoenix Cache located at %s.", PHOENIX_CACHE_DIR.absolute())

    def decorator(func):
        @functools.wraps(func)
        def wrapper(
            *, span_ids: list[SpanId], phoenix_graphql_endpoint: str
        ) -> dict[SpanId, dict[str, Any]]:
            result: dict[SpanId, dict[str, Any]] = {}
            missing: list[SpanId] = []

            for sid in span_ids:
                key = make_cache_key(sid, phoenix_graphql_endpoint)
                data = PHOENIX_DISK_CACHE.get(key)
                if data is None:
                    missing.append(sid)
                else:
                    result[sid] = data

            if missing:
                batch_result = func(
                    span_ids=missing,
                    phoenix_graphql_endpoint=phoenix_graphql_endpoint,
                )
                for sid, data in batch_result.items():
                    key = make_cache_key(sid, phoenix_graphql_endpoint)
                    PHOENIX_DISK_CACHE.set(key, data)  # direct set
                    result[sid] = data

            return result

        return wrapper

    return decorator
