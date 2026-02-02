from __future__ import annotations

from abc import ABC, abstractmethod

import patito as pt
import polars as pl
from pyarrow.parquet import ParquetWriter
from pydantic import ConfigDict

from social_groups.analyzer.utils import polars_schema_to_arrow_schema


class PolarsBaseModel(pt.Model, ABC):
    model_config = ConfigDict(use_enum_values=True)

    @classmethod
    @abstractmethod
    def from_raw_data(cls, *args, **kwargs) -> PolarsBaseModel: ...

    @classmethod
    def get_polars_schema(cls) -> pl.Schema:
        return pl.Schema(cls.examples().schema)

    @classmethod
    def create_parquet_table(cls, items: list[PolarsBaseModel]):
        schema = cls.get_polars_schema()

        return (
            cls.DataFrame(
                [item.model_dump() for item in items], orient="row", schema=schema
            )
            .cast(strict=True)
            .validate()
            .to_arrow()
            .cast(polars_schema_to_arrow_schema(schema))
        )


def append_parquet_row(writer: ParquetWriter, element: PolarsBaseModel):
    writer.write_table(element.__class__.create_parquet_table([element]))
