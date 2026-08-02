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
            asset="final_different_families",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_correct_data_length",
            asset="final_different_families",
            blocking=True,
        ),
    ],
)
def final_different_families(combined_data: pl.DataFrame):
    def create_sizes(model_names: list[str]) -> list[str]:
        def transform_model_name(x: str):
            if "Ministral" in x:
                i = "m"
            elif "Qwen3.5" in x:
                i = "Q"
            elif "Qwen3" in x:
                i = "q"
            else:
                raise NotImplementedError(x)

            return i + MODEL_NAME_TO_LETTER_MAPPING[x]

        return [transform_model_name(m) for m in model_names]

    final_different_families = (
        combined_data.filter(
            pl.col("name").is_in({"different_families_mad2", "different_families_mad3"})
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
        )
        .with_columns(
            pl.col(plc.model_names)
            .list.eval(parse_model_family(pl.element()))
            .alias("model_families"),
            pl.col(plc.model_names)
            .list.eval(parse_parameters(pl.element()))
            .alias("model_parameters"),
            pl.col(plc.model_names)
            .map_elements(create_sizes, return_dtype=pl.List(pl.String))
            .alias("sizes"),
        )
        .drop("_experiment_configuration")
    )

    yield check_correct_data_length(final_different_families, 36000)
    yield check_unique_data_connector(final_different_families)

    yield dg.Output(
        final_different_families.drop(
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


defs = dg.Definitions(assets=dg.with_source_code_references([final_different_families]))
