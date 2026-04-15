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
            asset="final_no_discussion_voting",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_correct_data_length",
            asset="final_no_discussion_voting",
            blocking=True,
        ),
    ],
)
def final_no_discussion_voting(combined_data: pl.DataFrame):
    final_no_discussion_voting = (
        combined_data.filter(
            (pl.col("name").is_in({"no_discussion_voting"}))
            & (pl.col("data_connector") == "mmlu-pro-big-subset")
        )
        .with_columns(
            deserialize_experiment_configuration(
                pl.col("experiment_configuration_json")
            ).alias("_experiment_configuration")
        )
        .with_columns(
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("debate_agents")
            .list.eval(pl.element().struct.field("backend").struct.field("model_name"))
            .list.item()
            .alias("model_name"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("debate_agents")
            .list.eval(pl.element().struct.field("params").struct.field("temperature"))
            .list.item()
            .alias("temperature"),
        )
        .drop("_experiment_configuration")
    )

    yield check_unique_data_connector(final_no_discussion_voting)
    yield check_correct_data_length(final_no_discussion_voting, 18000)

    yield dg.Output(
        final_no_discussion_voting.drop(
            [
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
                "final_answer",
            ]
        )
    )


defs = dg.Definitions(
    assets=dg.with_source_code_references([final_no_discussion_voting])
)
