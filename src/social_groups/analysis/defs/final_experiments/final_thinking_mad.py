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
            asset="final_thinking_mad",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_correct_data_length",
            asset="final_thinking_mad",
            blocking=True,
        ),
    ],
)
def final_thinking_mad(combined_data: pl.DataFrame):

    final_thinking_mad = (
        combined_data.filter(
            pl.col("name").is_in({"het_group_thinking_mad2", "het_group_thinking_mad3"})
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
            .struct.field("openly_thinking_models")
            .list.eval(pl.element().struct.field("model_name"))
            .alias("open_thinking_models_config"),
        )
        .with_columns(
            pl.struct(["model_names", "open_thinking_models_config"])
            .map_elements(
                lambda x: [
                    m if m in x["open_thinking_models_config"] else None
                    for m in x["model_names"]
                ],
                return_dtype=pl.List(pl.String),
            )
            .alias("open_thinking_models"),
            pl.struct(["model_names", "open_thinking_models_config"])
            .map_elements(
                lambda x: [
                    m if m not in x["open_thinking_models_config"] else None
                    for m in x["model_names"]
                ],
                return_dtype=pl.List(pl.String),
            )
            .alias("closed_thinking_models"),
        )
        .with_columns(
            pl.col(plc.model_names)
            .list.eval(parse_model_family(pl.element()))
            .list.unique()
            .list.item()
            .alias(plc.model_family),
            pl.col("closed_thinking_models")
            .list.eval(
                pl.when(pl.element() == pl.lit(None))
                .then(pl.element())
                .otherwise(parse_parameters(pl.element()))
            )
            .cast(pl.List(pl.Float64))
            .alias("closed_thinking_models_parameter"),
            pl.col("open_thinking_models")
            .list.eval(
                pl.when(pl.element() == pl.lit(None))
                .then(pl.element())
                .otherwise(parse_parameters(pl.element()))
            )
            .cast(pl.List(pl.Float64))
            .alias("open_thinking_models_parameter"),
            pl.col("open_thinking_models_config")
            .list.eval(pl.element().replace(MODEL_NAME_TO_LETTER_MAPPING))
            .alias("who_thinks_openly"),
        )
        .with_columns(
            pl.when(pl.col("who_thinks_openly").list.len() == 0)
            .then(pl.lit("none"))
            .otherwise(
                pl.col("who_thinks_openly")
                .list.first()
                .replace(MODEL_NAME_TO_LETTER_MAPPING)
            )
            .alias("who_thinks_openly")
        )
        .drop("_experiment_configuration")
    )

    yield check_correct_data_length(final_thinking_mad, 97200)
    yield check_unique_data_connector(final_thinking_mad)

    yield dg.Output(
        final_thinking_mad.drop(
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
                "open_thinking_models_config",
            ]
        )
    )


defs = dg.Definitions(assets=dg.with_source_code_references([final_thinking_mad]))
