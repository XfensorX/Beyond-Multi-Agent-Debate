from __future__ import annotations

import logging
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import field
from functools import cache
from typing import Any, Dict, Iterator, Optional

import httpx
from openinference.instrumentation import capture_span_context
from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.trace.span import format_span_id, format_trace_id
from phoenix.otel import register
from pydantic import BaseModel

from social_groups.trialrunner.utils.general import flatten_dict

log = logging.getLogger(__name__)


def phoenix_server_is_up(
    url: str,
    timeout_s: float = 1.5,
) -> bool:
    """
    Returns True if the Phoenix GraphQL server responds successfully to a simple query.

    Assumes a GraphQL HTTP endpoint at http://host:port/path that accepts JSON POST.
    """

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        ValueError,
    ) as e:
        log.error(repr(e))

        return False


def setup_phoenix(endpoint: str, project_name: str) -> None:
    register(
        project_name=project_name,
        batch=True,
        endpoint=endpoint,
        auto_instrument=True,
        verbose=False,
    )


_PROJECTS_QUERY = """
query ($after: String = null) {
  projects(after: $after) {
    edges { project: node { id name } }
    pageInfo { hasNextPage endCursor }
  }
}
"""

_URL_INFOS_QUERY = """
query ($spanId: String!) {
  s0: getSpanByOtelId(spanId: $spanId) {
    id
    project { id }
    context { traceId }
  }
}
"""


def get_phoenix_project_id(
    *,
    phoenix_base_url: str,
    project_name: str,
    timeout_s: float = 10.0,
) -> str:
    base = phoenix_base_url.rstrip("/")
    after: Optional[str] = None

    with httpx.Client(base_url=base, timeout=timeout_s) as client:
        while True:
            r = client.post(
                "/graphql",
                json={"query": _PROJECTS_QUERY, "variables": {"after": after}},
            )
            r.raise_for_status()
            payload = r.json()["data"]["projects"]

            for edge in payload["edges"]:
                proj = edge["project"]
                if proj["name"] == project_name:
                    return proj["id"]

            page_info = payload["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            after = page_info["endCursor"]

    raise KeyError(
        f"Phoenix project '{project_name}' not found at {phoenix_base_url}. "
        f"Tip: make sure at least one trace/span has been sent to that project."
    )


def retrieve_phoenix_url_from_span_id(
    span_id: str,
    *,
    phoenix_base_url: str,
    timeout_s: float = 10.0,
) -> str:
    """
    Returns Phoenix's GraphQL 'project id' for a given project name.
    Works for local Phoenix (default http://localhost:6006) and hosted Phoenix, as long as /graphql is reachable.
    """
    base = phoenix_base_url.rstrip("/")

    with httpx.Client(base_url=base, timeout=timeout_s) as client:
        r = client.post(
            "/graphql",
            json={"query": _URL_INFOS_QUERY, "variables": {"spanId": span_id}},
        )
        r.raise_for_status()
        payload = r.json()["data"]["s0"]

        return f"{phoenix_base_url}/projects/{payload['project']['id']}/spans/{payload['context']['traceId']}?selectedNoteSpanId={payload['id']}"


class PhoenixExampleHandle(BaseModel):
    trace_id_hex: Optional[str] = None
    span_id_hex: Optional[str] = None
    captured_span_ids: list[str] = field(default_factory=list)

    @staticmethod
    @cache
    def resolve_project_id(phoenix_base_url: str, project_name: str) -> str:
        return get_phoenix_project_id(
            phoenix_base_url=phoenix_base_url, project_name=project_name
        )

    def span_url(self, phoenix_base_url: str, project_name: str) -> str:
        if not self.span_id_hex:
            raise ValueError("span_id_hex not set (did the span finish?)")
        return f"{phoenix_base_url.rstrip('/')}/projects/{self.resolve_project_id(phoenix_base_url, project_name)}/spans/{self.span_id_hex}"

    def trace_url(self, phoenix_base_url: str, project_name: str) -> str:
        if not self.trace_id_hex:
            raise ValueError("trace_id_hex not set (did the span finish?)")
        return f"{phoenix_base_url.rstrip('/')}/projects/{self.resolve_project_id(phoenix_base_url, project_name)}/traces/{self.trace_id_hex}"


@contextmanager
def phoenix_example_span(
    *,
    example_id: str,
    tracer_name: str = "experiment",
    span_name: str = "example",
    span_kind: str = "chain",
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[tuple[Span, PhoenixExampleHandle]]:
    """
    Creates a *single parent span* for one example, and ensures everything inside becomes its child
    (auto-instrumented OpenAI/LangChain spans included) by making it the current span.

    Yields: (span, handle)
      - span: the live OpenTelemetry span (you can set attributes, add events, record exceptions, etc.)
      - handle: has span_id/trace_id + helpers to build Phoenix deep links

    IMPORTANT:
      - Make sure Phoenix tracing is registered already (phoenix.otel.register(...)).
    """
    tracer = trace.get_tracer(tracer_name)

    handle = PhoenixExampleHandle()

    # Capture all OpenInference spans created within the block (useful to persist/inspect).
    with capture_span_context() as capture:
        # Create the example parent span and make it current.
        with tracer.start_as_current_span(
            span_name,
            attributes={
                **(flatten_dict(attributes, map_to_basic_types=True) or {}),
                "experiment.example_id": example_id,
                # OpenInference convention: set openinference span kind
                "openinference.span.kind": span_kind,
            },
        ) as span:
            try:
                yield span, handle
            finally:
                # Capture IDs for deep-linking
                ctx = span.get_span_context()
                handle.span_id_hex = format_span_id(ctx.span_id)
                handle.trace_id_hex = format_trace_id(ctx.trace_id)

                # Also capture all span IDs created within this context (optional)
                # capture.get_span_contexts() exists; we store span IDs as hex strings when available.
                span_contexts = capture.get_span_contexts() or []
                handle.captured_span_ids = [
                    format_span_id(sc.span_id)
                    for sc in span_contexts
                    if getattr(sc, "span_id", None) is not None
                ]


@contextmanager
def phoenix_log_span(
    message: str, *, tracer_name: str = "experiment", **attrs: Any
) -> Iterator[Span]:
    """
    Creates a small child span for a human-readable log/message that will show up in Phoenix.
    Call this *inside* phoenix_example_span(...) so it's nested under the example.
    """
    tracer = trace.get_tracer(tracer_name)
    with tracer.start_as_current_span(
        "log",
        attributes={"log.message": message, "openinference.span.kind": "tool", **attrs},
    ) as span:
        yield span
