import logging
from concurrent.futures import Future
from typing import Any

import httpx

SENTINEL = "___SENTINEL___"
EXCEPTION_SENTINEL = "___EXCEPTION_SENTINEL___"
logger = logging.getLogger("PARQUET BUILDER")

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
ExperimentName = str
