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
            name="check_unique_data_connector", asset="final_baseline", blocking=True
        ),
        AssetCheckSpec(
            name="check_correct_data_length", asset="final_baseline", blocking=True
        ),
    ],
)
def final_baseline(combined_data: pl.DataFrame):
    final_baseline = (
        combined_data.filter(
            (
                pl.col("name").is_in(
                    {
                        "heterogeneous_group_baseline",
                        "heterogeneous_group_baseline_tool",
                        "heterogeneous_group_baseline_structured_output",
                    }
                )
            )
            & (pl.col("data_connector") == "mmlu-pro-big-subset")
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
            generation_method=pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("name")
            .replace(
                {
                    "single-agent": "standard",
                    "single-agent-structured-output": "structured-output",
                    "single-agent-tool": "tool",
                }
            ),
        )
        .drop("_experiment_configuration")
    )

    yield check_correct_data_length(final_baseline, 108000)
    yield check_unique_data_connector(final_baseline)

    yield dg.Output(
        final_baseline.drop(
            [
                "answers_at_beginning",  # empty
                "answers_at_end",  # empty
                "experiment_id",  #
                "name",  #
                "message_ids",  # not filled
                "phoenix_span_id",  # not interesting
                "data_connector",  # all "mmlu-pro-big-subset"
                "original_source",  # not interesting
                "experiment_configuration_json",  # already used
                "meta_info_json",
                "execution_config_json",
                "answer_index",
                "answer_options",
            ]
        )
    )


defs = dg.Definitions(assets=dg.with_source_code_references([final_baseline]))
