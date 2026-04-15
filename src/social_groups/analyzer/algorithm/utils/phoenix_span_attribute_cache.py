import functools
import logging
import os
from typing import Any

from diskcache import Cache

from social_groups.analyzer.config import KNOWN_FAULTY_SPAN_IDS
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
        async def wrapper(
            *, span_ids: list[SpanId], phoenix_graphql_endpoint: str
        ) -> dict[SpanId, dict[str, Any]]:
            result: dict[SpanId, dict[str, Any]] = {}
            missing: list[SpanId] = []

            used_sid_backward_mapping: dict[SpanId, SpanId] = {}

            for original_sid in span_ids:
                used_sid = original_sid
                if original_sid in KNOWN_FAULTY_SPAN_IDS:
                    used_sid = KNOWN_FAULTY_SPAN_IDS[original_sid]
                    used_sid_backward_mapping[used_sid] = original_sid

                key = make_cache_key(used_sid, phoenix_graphql_endpoint)
                data = PHOENIX_DISK_CACHE.get(key)
                if data is None:
                    missing.append(used_sid)
                else:
                    result[original_sid] = data

            if missing:
                batch_result = await func(
                    span_ids=missing,
                    phoenix_graphql_endpoint=phoenix_graphql_endpoint,
                )

                for used_sid, data in batch_result.items():
                    PHOENIX_DISK_CACHE.set(
                        make_cache_key(used_sid, phoenix_graphql_endpoint), data
                    )
                    original_sid = used_sid_backward_mapping.get(used_sid, used_sid)
                    result[original_sid] = data

            return result

        return wrapper

    return decorator
