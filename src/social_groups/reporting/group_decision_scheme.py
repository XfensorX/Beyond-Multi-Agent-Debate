import re

import polars as pl

from social_groups.reporting.group_reply import GroupReplyAggregator
from social_groups.reporting.parsing import AnswerComparer

CORRECT = "______custom________correct_column"
ROW_IDX = "______custom________row_index"


def count_equal_elements(
    df: pl.DataFrame, list_col: str, compare_col: str, *, new_col: str
):
    return (
        df.with_row_index(ROW_IDX)
        .explode(list_col)
        .with_columns((pl.col(list_col) == pl.col(compare_col)).alias(CORRECT))
        .group_by(ROW_IDX, maintain_order=True)
        .agg(pl.col(CORRECT).sum().alias(new_col))
        .join(df.with_row_index(ROW_IDX), on=ROW_IDX, how="left")
        .drop(ROW_IDX)
    )


def count_unequal_elements(
    df: pl.DataFrame, list_col: str, compare_col: str, *, new_col: str
):
    return (
        df.with_row_index(ROW_IDX)
        .explode(list_col)
        .with_columns((pl.col(list_col) != pl.col(compare_col)).alias(CORRECT))
        .group_by(ROW_IDX, maintain_order=True)
        .agg(pl.col(CORRECT).sum().alias(new_col))
        .join(df.with_row_index(ROW_IDX), on=ROW_IDX, how="left")
        .drop(ROW_IDX)
    )


BEGINNING_GROUP_VOTE = "___calculate_decision_scheme__beginning_group_vote"
END_GROUP_VOTE = "___calculate_decision_scheme__end_group_vote"
MEMBERS_CORRECT_BEGINNING = "___calculate_decision_scheme__members_correct_beginning"
MEMBERS_CORRECT_END = "___calculate_decision_scheme__members_correct_end"
END_GROUP_VOTE_CORRECT = "___calculate_decision_scheme__end_group_vote_correct"


def extend_by_group_info_before_and_after(
    df: pl.DataFrame,
    answers_before_col: str,
    answers_after_col: str,
    target_answer_col: str,
    group_reply_strategy: GroupReplyAggregator,
    comparison_strategy: AnswerComparer,
) -> pl.DataFrame:
    df = df.with_columns(
        (
            (group_reply_strategy(pl.col(answers_before_col))).alias(
                BEGINNING_GROUP_VOTE
            )
        ),
        ((group_reply_strategy(pl.col(answers_after_col))).alias(END_GROUP_VOTE)),
    )

    df = df.with_columns(
        (comparison_strategy(pl.col(END_GROUP_VOTE), pl.col(target_answer_col))).alias(
            END_GROUP_VOTE_CORRECT
        )
    )

    df = count_equal_elements(
        df, answers_before_col, target_answer_col, new_col=MEMBERS_CORRECT_BEGINNING
    )

    df = count_equal_elements(
        df, answers_after_col, target_answer_col, new_col=MEMBERS_CORRECT_END
    )

    return df


def calculate_extended_decision_scheme(
    df: pl.DataFrame,
    answers_before_col: str,
    answers_after_col: str,
    target_answer_col: str,
    group_reply_strategy: GroupReplyAggregator,
    comparison_strategy: AnswerComparer,
) -> pl.DataFrame:
    df = extend_by_group_info_before_and_after(
        df,
        answers_before_col,
        answers_after_col,
        target_answer_col,
        group_reply_strategy,
        comparison_strategy,
    )
    return (
        df.group_by(MEMBERS_CORRECT_BEGINNING, MEMBERS_CORRECT_END)
        .agg(
            correct=(pl.col(END_GROUP_VOTE_CORRECT).drop_nulls().mean()),
            occurrences=(pl.col(END_GROUP_VOTE_CORRECT).drop_nulls().len()),
        )
        .with_columns(incorrect=pl.lit(1) - pl.col("correct"))
        .sort(MEMBERS_CORRECT_BEGINNING, MEMBERS_CORRECT_END, descending=True)
        .rename(
            {
                MEMBERS_CORRECT_BEGINNING: "Correct Members Beginning",
                MEMBERS_CORRECT_END: "Correct Members End",
            }
        )
    )


def calculate_decision_scheme(
    df: pl.DataFrame,
    answers_before_col: str,
    answers_after_col: str,
    target_answer_col: str,
    group_reply_strategy: GroupReplyAggregator,
    comparison_strategy: AnswerComparer,
) -> pl.DataFrame:
    return (
        calculate_extended_decision_scheme(
            df=df,
            answers_before_col=answers_before_col,
            answers_after_col=answers_after_col,
            target_answer_col=target_answer_col,
            group_reply_strategy=group_reply_strategy,
            comparison_strategy=comparison_strategy,
        )
        .group_by("Correct Members Beginning")
        .agg(
            correct=(
                pl.col("correct").dot(pl.col("occurrences")) / pl.sum("occurrences")
            ),
            incorrect=(
                pl.col("incorrect").dot(pl.col("occurrences")) / pl.sum("occurrences")
            ),
        )
        .select("correct", "incorrect", "Correct Members Beginning")
    )


def calculate_empirical_decision_scheme(
    df: pl.DataFrame,
    answers_before_col: str,
    answers_after_col: str,
    target_answer_col: str,
    group_reply_strategy: GroupReplyAggregator,
    comparison_strategy: AnswerComparer,
) -> pl.DataFrame:
    return (
        calculate_extended_decision_scheme(
            df=df,
            answers_before_col=answers_before_col,
            answers_after_col=answers_after_col,
            target_answer_col=target_answer_col,
            group_reply_strategy=group_reply_strategy,
            comparison_strategy=comparison_strategy,
        )
        .group_by("Correct Members Beginning")
        .agg(
            correct=(pl.col("correct").dot(pl.col("occurrences"))).cast(
                int, strict=True
            ),
            incorrect=(pl.col("incorrect").dot(pl.col("occurrences"))).cast(
                int, strict=True
            ),
        )
        .select("correct", "incorrect", "Correct Members Beginning")
        .sort("Correct Members Beginning")
    )


def calculate_empirical_decision_scheme_matrix(
    df: pl.DataFrame,
    answers_before_col: str,
    answers_after_col: str,
    target_answer_col: str,
    group_reply_strategy: GroupReplyAggregator,
    comparison_strategy: AnswerComparer,
) -> pl.DataFrame:
    df = (
        calculate_extended_decision_scheme(
            df=df,
            answers_before_col=answers_before_col,
            answers_after_col=answers_after_col,
            target_answer_col=target_answer_col,
            group_reply_strategy=group_reply_strategy,
            comparison_strategy=comparison_strategy,
        )
        .pivot(
            "Correct Members Beginning",
            index="Correct Members End",
            values="occurrences",
        )
        .select(
            pl.lit("to ").alias("_") + pl.col("Correct Members End").cast(str),
            pl.exclude("Correct Members End"),
        )
        .rename(lambda x: f"from {x}" if x != "_" else x)
        .fill_null(0)
    )

    row_ids = [int(re.search(r"\d+", x).group()) for x in df["_"].to_list()]
    col_ids = [int(re.search(r"\d+", c).group()) for c in df.columns if c != "_"]

    max_id = max(row_ids + col_ids)
    all_ids = list(range(max_id + 1))

    for i in all_ids:
        col = f"from {i}"
        if col not in df.columns:
            df = df.with_columns(pl.lit(0).alias(col))

    existing_rows = set(df["_"].to_list())
    missing_rows = [
        {"_": f"to {i}", **{f"from {j}": 0 for j in all_ids}}
        for i in all_ids
        if f"to {i}" not in existing_rows
    ]

    if missing_rows:
        missing_df = (
            pl.DataFrame(missing_rows)
            .select(df.columns)
            .with_columns(pl.selectors.numeric().cast(pl.UInt32))
        )
        df = pl.concat([df, missing_df], how="vertical")

    df = (
        df.with_columns(
            pl.col("_").str.extract(r"(\d+)").cast(pl.Int64).alias("__order")
        )
        .sort("__order", descending=True)
        .drop("__order")
    )

    df = df.select(["_"] + [f"from {i}" for i in reversed(all_ids)])  # order columns

    columns_as_int = (
        pl.Series(df.select(pl.exclude("_")).columns).str.replace("from ", "").cast(int)
    )

    rows_as_int = df["_"].str.replace("to ", "").cast(int)

    assert columns_as_int.is_sorted(descending=True)
    assert rows_as_int.is_sorted(descending=True)

    assert columns_as_int.n_unique() == columns_as_int.max() + 1
    assert rows_as_int.n_unique() == rows_as_int.max() + 1

    assert columns_as_int.min() == 0
    assert rows_as_int.min() == 0

    return df


def calculate_decision_scheme_matrix(
    df: pl.DataFrame,
    answers_before_col: str,
    answers_after_col: str,
    target_answer_col: str,
    group_reply_strategy: GroupReplyAggregator,
    comparison_strategy: AnswerComparer,
) -> pl.DataFrame:
    return calculate_empirical_decision_scheme_matrix(
        df=df,
        answers_before_col=answers_before_col,
        answers_after_col=answers_after_col,
        target_answer_col=target_answer_col,
        group_reply_strategy=group_reply_strategy,
        comparison_strategy=comparison_strategy,
    ).select("_", pl.exclude("_") / pl.exclude("_").sum())
