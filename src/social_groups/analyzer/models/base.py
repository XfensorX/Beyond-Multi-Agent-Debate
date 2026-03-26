from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import patito as pt
import polars as pl
from pyarrow.parquet import ParquetWriter
from pydantic import ConfigDict

from social_groups.analyzer.utils import polars_schema_to_arrow_schema


class PolarsBaseModel(pt.Model, ABC):
    model_config = ConfigDict(use_enum_values=True)

    @classmethod
    def _compute_schemas(cls):
        if not hasattr(cls, "_polars_schema"):
            cls._polars_schema = pl.Schema(cls.examples().schema)
            cls._arrow_schema = polars_schema_to_arrow_schema(cls._polars_schema)

    @classmethod
    @abstractmethod
    def from_raw_data(cls, *args, **kwargs) -> Self: ...

    @classmethod
    def get_polars_schema(cls):
        if not hasattr(cls, "_polars_schema"):
            cls._compute_schemas()
        return cls._polars_schema

    @classmethod
    def create_parquet_table(cls, items: list[Self]):
        if not hasattr(cls, "_polars_schema"):
            cls._compute_schemas()

        return (
            cls.DataFrame(
                [item.model_dump() for item in items],
                orient="row",
                schema=cls._polars_schema,
            )
            .cast(strict=True)
            .validate()
            .to_arrow()
            .cast(cls._arrow_schema)
        )


def append_parquet_row(writer: ParquetWriter, element: PolarsBaseModel):
    writer.write_table(element.__class__.create_parquet_table([element]))
