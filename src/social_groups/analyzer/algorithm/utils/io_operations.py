from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import httpx
import polars as pl
import pyarrow
import pyarrow.parquet as pq
import yaml
from omegaconf import OmegaConf
from pyarrow.parquet import ParquetWriter

from social_groups.analyzer.algorithm.utils.phoenix_span_attribute_cache import (
    with_per_span_cache,
)
from social_groups.directories import META_FILE_NAME
from social_groups.general.types import SpanId
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo
from social_groups.trialrunner.utils.phoenix import build_span_url


def polars_schema_to_arrow_schema(polars_schema: pl.Schema) -> pyarrow.Schema:
    return pl.DataFrame(schema=polars_schema).to_arrow().schema


def create_parquet_writer(location: Path, schema: pl.Schema) -> ParquetWriter:
    return pq.ParquetWriter(
        location,
        polars_schema_to_arrow_schema(schema),
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


ATTRIBUTE_KEY_SPAN_URL = "custom_phoenix_span_url_attribute"

CLIENT: None | httpx.Client = None


def worker_initializer():
    """Runs once per worker thread"""
    global CLIENT
    CLIENT = httpx.Client(
        transport=httpx.HTTPTransport(retries=3),
        timeout=httpx.Timeout(120.0, connect=30.0),  # adjust to your connection
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
    return CLIENT


@with_per_span_cache()
def get_span_attributes(
    *, span_ids: list[SpanId], phoenix_graphql_endpoint: str
) -> dict[SpanId, dict[str, Any]]:
    global CLIENT

    if CLIENT is None:
        CLIENT = worker_initializer()

    fields = "\n".join(
        f's_{oid}: getSpanByOtelId(spanId: "{oid}") {{ attributes id project {{ id }} context {{ traceId }} }}'
        for oid in span_ids
    )
    query = f"query GetSpans {{\n{fields}\n}}"

    response = CLIENT.post(
        phoenix_graphql_endpoint,
        json={
            "query": query,
        },
        headers={"Content-Type": "application/json"},
    )

    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise RuntimeError(data["errors"])

    out: dict[SpanId, dict[str, Any]] = {}
    for alias, node in data["data"].items():
        original_oid = alias[2:]
        if node is None:
            raise RuntimeError(f"Span {original_oid} has no node")

        attrs = node["attributes"]
        span = json.loads(attrs) if isinstance(attrs, str) else attrs
        span[ATTRIBUTE_KEY_SPAN_URL] = build_span_url(
            phoenix_graphql_endpoint.rstrip("/graphql"),
            node["project"]["id"],
            node["context"]["traceId"],
            node["id"],
        )
        out[original_oid] = span

    return out


def read_hydra_config(run_dir: Path) -> MainConfig:
    """
    Common Hydra output: <run_dir>/.hydra/config.yaml (and overrides.yaml).
    Adjust paths for your setup.
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    return MainConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.create(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))),
            resolve=True,
        ),
        strict=False,
    )


def read_meta_config(run_dir: Path) -> ExperimentMetaInfo:
    """
    Common Hydra output: <run_dir>/.hydra/config.yaml (and overrides.yaml).
    Adjust paths for your setup.
    """
    cfg_path = run_dir / META_FILE_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config = ExperimentMetaInfo.model_validate(config, strict=False)

    return config


def read_log_text(run_dir: Path, log_filename: str = "main.log") -> Optional[str]:
    p = run_dir / log_filename
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")
