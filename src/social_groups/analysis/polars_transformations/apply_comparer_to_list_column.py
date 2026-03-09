import polars as pl
from polars.testing import assert_frame_equal

import src.social_groups.polars_columns as plc
from social_groups.reporting.parsing import AnswerComparer

INDEX_COL = "_____custom____index_col"
INDIVIDUAL_COMPARISON = "____custom_individual_comparison_col"


def apply_comparer_to_list_column(
    df: pl.DataFrame, comparer: AnswerComparer, column: str
):
    new = (
        df.with_columns(pl.row_index().alias(INDEX_COL))
        .explode(column)
        .with_columns(
            comparer(
                pl.col(column),
                pl.col(plc.answer_string),
            ).alias(INDIVIDUAL_COMPARISON)
        )
        .group_by(INDEX_COL)
        .agg(
            pl.exclude(INDEX_COL, INDIVIDUAL_COMPARISON, column)
            .implode()
            .list.unique()
            .list.item(),
            pl.col(INDIVIDUAL_COMPARISON).implode().alias(column),
        )
        .drop(INDEX_COL)
    )

    assert_frame_equal(df.drop(column), new.drop(column), check_column_order=False)
    return new
