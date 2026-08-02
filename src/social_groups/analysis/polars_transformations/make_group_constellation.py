import polars as pl

import social_groups.polars_columns as plc
import social_groups.polars_values as plv


def make_group_constellation():
    return (
        pl.col(plc.model_names)
        .list.eval(pl.element().replace(plv.MODEL_NAME_TO_LETTER_MAPPING))
        .list.join("")
        .alias(plc.group_constellation)
    )


def make_model_family():
    return (
        pl.col(plc.model_names)
        .list.eval(parse_model_family(pl.element()))
        .list.unique()
        .list.item()
        .alias(plc.model_family)
    )


def parse_model_family(expr: pl.Expr):
    return expr.str.split("/").list.last().str.split("-").list.first()


def parse_parameters(expr: pl.Expr):
    return expr.str.split("B").list.first().str.split("-").list.last().cast(float)
