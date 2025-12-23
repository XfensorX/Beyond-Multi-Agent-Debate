import logging

from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
import urllib.request
import urllib.error

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
    tracer_provider = register(
        project_name=project_name,
        batch=True,
        endpoint=endpoint,
        verbose=False,
    )
    LangChainInstrumentor(tracer_provider=tracer_provider).instrument(
        skip_dep_check=True
    )
