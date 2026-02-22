import polars as pl

import social_groups.polars_columns as plc
from social_groups.reporting.analysis_columns import AnalysisColumn
from social_groups.reporting.group_reply import GroupReplyAggregator
from social_groups.reporting.parsing import AnswerComparer, AnswerParser


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
            comparer(
                pl.col(AnalysisColumn.parsed_combined_answers_after.value),
                pl.col(plc.answer_string),
            ).alias(plc.is_correct)
        )
    )
