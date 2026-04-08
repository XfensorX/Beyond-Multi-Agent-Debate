from itertools import chain, combinations_with_replacement

import dagster as dg
import polars as pl

import social_groups.polars_columns as plc
from social_groups.config import GROUP_SIZES, MODEL_LETTERS


def check_standard_group_constellations(frame: pl.DataFrame):
    group_constellations = set(frame[plc.group_constellation].unique())

    standard_groupings = set(
        chain(
            *[
                [
                    "".join(sorted(c))
                    for c in combinations_with_replacement(MODEL_LETTERS, i)
                ]
                for i in GROUP_SIZES
            ]
        )
    )

    return dg.AssetCheckResult(
        check_name="check_standard_group_constellations",
        passed=bool((group_constellations - standard_groupings) == set()),
        metadata={
            "constellations": list(group_constellations),
            "wrong_constellations": list(group_constellations - standard_groupings),
        },
    )


def check_unique_data_connector(frame: pl.DataFrame):
    data_conns = set(frame["data_connector"].unique())

    return dg.AssetCheckResult(
        check_name="check_unique_data_connector",
        passed=bool(len(data_conns) == 1),
        metadata={"data_connectors": list(data_conns)},
    )


def check_single_model_only(frame: pl.DataFrame):
    try:
        frame[plc.model_names].list.item()
        valid = True
    except pl.exceptions.ComputeError:
        valid = False

    return dg.AssetCheckResult(
        check_name="check_single_model_only",
        passed=valid,
        metadata={"Given Model Names": frame[plc.model_names].unique().to_list()},
    )
