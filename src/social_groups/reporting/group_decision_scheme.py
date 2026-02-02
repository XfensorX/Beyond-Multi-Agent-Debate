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


def calculate_extended_decision_scheme(
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
        df, answers_before_col, "answer_string", new_col=MEMBERS_CORRECT_BEGINNING
    )

    df = count_equal_elements(
        df, answers_after_col, "answer_string", new_col=MEMBERS_CORRECT_END
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
            ).mean(),
            incorrect=(
                pl.col("incorrect").dot(pl.col("occurrences")) / pl.sum("occurrences")
            ).mean(),
        )
        .select("correct", "incorrect", "Correct Members Beginning")
    )
