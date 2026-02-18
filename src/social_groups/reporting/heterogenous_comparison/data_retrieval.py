import polars as pl

from social_groups.reporting.analysis_columns import AnalysisColumn
from social_groups.reporting.group_reply import GroupReplyAggregator
from social_groups.reporting.parsing import AnswerComparer, AnswerParser
from social_groups.reporting.plots.config import MODEL_NAME_TO_LETTER_MAPPING


def apply_parsing_and_group_decision(
    df: pl.DataFrame,
    parser: AnswerParser,
    comparer: AnswerComparer,
    group_reply: GroupReplyAggregator,
) -> pl.DataFrame:
    return (
        df.with_columns(
            (
                pl.col("answers_at_beginning")
                .list.eval(parser(pl.element()))
                .alias(AnalysisColumn.parsed_individual_answers_before.value)
            ),
            (
                pl.col("answers_at_end")
                .list.eval(parser(pl.element()))
                .alias(AnalysisColumn.parsed_individual_answers_after.value)
            ),
            group_constellation=(
                pl.col("model_names")
                .list.eval(pl.element().replace(MODEL_NAME_TO_LETTER_MAPPING))
                .list.sort()
                .list.join("")
            ),
        )
        .with_columns(
            group_reply(
                pl.col(AnalysisColumn.parsed_individual_answers_before.value)
            ).alias(AnalysisColumn.parsed_combined_answers_before.value),
            group_reply(
                pl.col(AnalysisColumn.parsed_individual_answers_after.value)
            ).alias(AnalysisColumn.parsed_combined_answers_after.value),
        )
        .with_columns(
            is_correct=comparer(
                pl.col(AnalysisColumn.parsed_combined_answers_after.value),
                pl.col("answer_string"),
            )
        )
    )
