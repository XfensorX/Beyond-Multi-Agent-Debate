import json

import polars as pl

from social_groups.reporting.analysis_columns import AnalysisColumn
from social_groups.reporting.group_reply import GroupReplyAggregator
from social_groups.reporting.loading import build_analysis_frame
from social_groups.reporting.parsing import AnswerComparer, AnswerParser
from social_groups.reporting.plots.config import MODEL_NAME_TO_LETTER_MAPPING
from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


def get_baseline_frame():
    b_frame = (
        build_analysis_frame()
        .filter(pl.col("run_identifier").str.contains("heterogeneous_group"))
        .filter(pl.col("run_identifier").str.contains("baseline"))
    )

    assert set(b_frame["data_connector"].unique()) == {"mmlu-pro-subset"}
    assert len(set(b_frame["experiment_id"].unique())) == 1
    assert len(set(b_frame["name"].unique())) == 1

    b_frame = b_frame.with_columns(
        model_name=pl.col("experiment_configuration_json").map_elements(
            lambda x: ExperimentConfig.model_validate(
                json.loads(x)
            ).strategy.configuration.backend.model_name,
            return_dtype=pl.Utf8,
        )
    )

    b_frame = b_frame.drop(
        [
            "answers_at_beginning",  # empty
            "answers_at_end",  # empty
            "experiment_id",  # just one
            "name",  # just one
            "message_ids",  # not filled
            # "phoenix_span_id",  # not interesting
            "data_connector",  # all "mmlu-pro-subset"
            "original_source",  # not interesting
            "experiment_configuration_json",  # already used
            "meta_info_json",
            "execution_config_json",
            "answer_index",
            "answer_options",
        ]
    )

    return b_frame


def get_mad_frame():
    m_frame = (
        build_analysis_frame()
        .filter(pl.col("run_identifier").str.contains("heterogeneous_group"))
        .filter(pl.col("run_identifier").str.contains("mad"))
    )
    #
    assert set(m_frame["data_connector"].unique()) == {"mmlu-pro-subset"}
    # assert len(set(b_frame["experiment_id"].unique())) == 1
    # assert len(set(b_frame["name"].unique())) == 1
    #
    m_frame = m_frame.with_columns(
        model_names=pl.col("experiment_configuration_json").map_elements(
            lambda x: list(
                map(
                    lambda d: d.backend.model_name,
                    ExperimentConfig.model_validate(
                        json.loads(x)
                    ).strategy.configuration.debate_agents,
                )
            ),
            return_dtype=pl.List(pl.Utf8),
        )
    )

    m_frame = m_frame.drop(
        [
            "final_answer",  # is empty in group case
            "message_ids",  # not filled
            "data_connector",  # all "mmlu-pro-subset"
            "original_source",  # not interesting
            # "experiment_configuration_json",  # already used
            # "meta_info_json",
            "execution_config_json",
            "answer_index",
            "answer_options",
        ]
    )

    return m_frame


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
                # TODO: should probably be a parameter instead
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
