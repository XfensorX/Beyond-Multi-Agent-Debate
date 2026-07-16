import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

import social_groups.polars_columns as plc
from social_groups.analysis.asset_checks import (
    check_correct_data_length,
    check_unique_data_connector,
)
from social_groups.analysis.polars_transformations.deserialize_experiment_configuration import (
    deserialize_experiment_configuration,
)
from social_groups.analysis.polars_transformations.make_group_constellation import (
    parse_model_family,
    parse_parameters,
)
from social_groups.polars_values import MODEL_NAME_TO_LETTER_MAPPING


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="final_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector",
            asset="final_diversity_params",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_correct_data_length",
            asset="final_diversity_params",
            blocking=True,
        ),
    ],
)
def final_diversity_params(combined_data: pl.DataFrame):


    final_diversity_params = (
        combined_data.filter(
            pl.col("name").is_in(
                {"diversity_parameter_sweep_mad2", "diversity_parameter_sweep_mad3"}
            )
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
            .alias(plc.model_names),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("number_of_rounds")
            .alias("number_of_mad_rounds"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("debate_agents")
            .list.eval(pl.element().struct.field("params").struct.field("temperature"))
            .list.unique()
            .list.item()
            .alias("temperature"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("debate_agents")
            .list.eval(pl.element().struct.field("params").struct.field("top_p"))
            .list.unique()
            .list.item()
            .alias("top_p"),
        )
        .with_columns(
            pl.col(plc.model_names)
            .list.eval(parse_model_family(pl.element()))
            .list.unique()
            .list.item()
            .alias(plc.model_family),
            pl.col(plc.model_names)
            .list.eval(parse_parameters(pl.element()))
            .alias("model_parameters"),
            pl.col(plc.model_names)
            .list.eval(pl.element().replace(MODEL_NAME_TO_LETTER_MAPPING))
            .alias("sizes"),
        )
        .drop("_experiment_configuration")
    )

    yield check_correct_data_length(final_diversity_params, 172800)
    yield check_unique_data_connector(final_diversity_params)

    yield dg.Output(
        final_diversity_params.drop(
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


defs = dg.Definitions(assets=dg.with_source_code_references([final_diversity_params]))
