import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

from social_groups.analysis.asset_checks import (
    check_correct_data_length,
    check_unique_data_connector,
)
from social_groups.analysis.polars_transformations.deserialize_experiment_configuration import (
    deserialize_experiment_configuration,
)


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="final_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector",
            asset="final_gemma4_no_discussion_voting_base",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_correct_data_length",
            asset="final_gemma4_no_discussion_voting_base",
            blocking=True,
        ),
    ],
)
def final_gemma4_no_discussion_voting_base(combined_data: pl.DataFrame):
    final_gemma4_no_discussion_voting_base = (
        combined_data.filter(
            (pl.col("name") == "gemma4_no_discussion_voting_baseline")
            & (pl.col("data_connector") == "gpqa-diamond")
        )
        .with_columns(
            deserialize_experiment_configuration(
                pl.col("experiment_configuration_json")
            ).alias("_experiment_configuration")
        )
        .with_columns(
            with_few_shot_prompting=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("use_few_shot_prompting"),
            use_thinking=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("use_thinking"),
            model_name=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("backend")
            .struct.field("model_name"),
            max_new_tokens=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("llm")
            .struct.field("max_new_tokens"),
            temperature=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("llm")
            .struct.field("temperature"),
            top_p=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("llm")
            .struct.field("top_p"),
            seed=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("llm")
            .struct.field("seed"),
            generation_method=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("name")
            .replace({"single-agent": "standard"}),
        )
        .with_columns(
            model_size=pl.col("model_name").str.extract(r"gemma-4-(.+?)-it", 1),
        )
        .drop("_experiment_configuration")
    )

    yield check_unique_data_connector(final_gemma4_no_discussion_voting_base)
    yield check_correct_data_length(
        final_gemma4_no_discussion_voting_base, 198 * 2 * 5 * 9
    )

    yield dg.Output(
        final_gemma4_no_discussion_voting_base.drop(
            [
                "answers_at_beginning",  # empty
                "answers_at_end",  # empty
                "experiment_id",
                "name",
                "message_ids",  # not filled
                "phoenix_span_id",  # not interesting
                "data_connector",  # all "gpqa-diamond"
                "original_source",  # not interesting
                "experiment_configuration_json",  # already used
                "meta_info_json",
                "execution_config_json",
                "answer_index",
                "answer_options",
            ]
        )
    )


defs = dg.Definitions(
    assets=dg.with_source_code_references([final_gemma4_no_discussion_voting_base])
)
