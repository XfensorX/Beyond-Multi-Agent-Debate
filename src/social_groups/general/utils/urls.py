from urllib.parse import urlparse, urlunparse


def replace_host_with_localhost(url: str, new_host: str = "localhost") -> str:
    parsed = urlparse(url)

    if parsed.port is not None:
        new_netloc = f"{new_host}:{parsed.port}"
    else:
        new_netloc = new_host

    new_url = urlunparse(
        (
            parsed.scheme,
            new_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return new_url
