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
