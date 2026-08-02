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

GEMMA4_MODEL_NAME_TO_LETTER_MAPPING = {
    "google/gemma-4-E2B-it": "L",
    "google/gemma-4-E4B-it": "H",
}


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="final_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector",
            asset="final_gemma4_multi_agent_debateV2",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_correct_data_length",
            asset="final_gemma4_multi_agent_debateV2",
            blocking=True,
        ),
    ],
)
def final_gemma4_multi_agent_debateV2(combined_data: pl.DataFrame):
    final_gemma4_multi_agent_debateV2 = (
        combined_data.filter(
            (
                pl.col("name").is_in(
                    {
                        "gemma4_multi_agent_debate_3part",
                        "gemma4_multi_agent_debate_2part",
                    }
                )
            )
            & (pl.col("data_connector") == "gpqa-diamond")
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
            .struct.field("openly_thinking_models")
            .list.eval(pl.element().struct.field("model_name"))
            .alias("openly_thinking_model_names"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("number_of_rounds")
            .alias("number_of_mad_rounds"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("use_few_shot_prompting")
            .alias("with_few_shot_prompting"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("debate_agents")
            .list.len()
            .alias("number_of_agents"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("debate_agents")
            .list.eval(pl.element().struct.field("params").struct.field("seed"))
            .list.unique()
            .list.item()
            .alias("seed"),
            pl.col("_experiment_configuration")
            .struct.field("strategy")
            .struct.field("configuration")
            .struct.field("debate_agents")
            .list.eval(
                pl.element().struct.field("params").struct.field("max_new_tokens")
            )
            .list.unique()
            .list.item()
            .alias("max_new_tokens"),
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
            .list.eval(pl.element().replace(GEMMA4_MODEL_NAME_TO_LETTER_MAPPING))
            .list.join("")
            .alias(plc.group_constellation),
            pl.col(plc.model_names)
            .list.eval(pl.element().str.extract(r"gemma-4-(.+?)-it", 1))
            .alias("model_sizes"),
            pl.when(pl.col("openly_thinking_model_names").is_null())
            .then(pl.lit("nobody"))
            .when(
                pl.col("openly_thinking_model_names").list.contains(
                    "google/gemma-4-E2B-it"
                )
                & pl.col("openly_thinking_model_names").list.contains(
                    "google/gemma-4-E4B-it"
                )
            )
            .then(pl.lit("all"))
            .when(
                pl.col("openly_thinking_model_names").list.contains(
                    "google/gemma-4-E2B-it"
                )
            )
            .then(pl.lit("only_L"))
            .when(
                pl.col("openly_thinking_model_names").list.contains(
                    "google/gemma-4-E4B-it"
                )
            )
            .then(pl.lit("only_H"))
            .otherwise(pl.lit("nobody"))
            .alias("thinking_openly"),
        )
        .drop("_experiment_configuration")
    )

    yield check_unique_data_connector(final_gemma4_multi_agent_debateV2)
    yield check_correct_data_length(
        final_gemma4_multi_agent_debateV2, 198 * 5 * 4 * (4 + 7)
    )

    yield dg.Output(
        final_gemma4_multi_agent_debateV2.drop(
            [
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
                "final_answer",  # empty in this MAD strategy
                "openly_thinking_model_names",  # already summarized
            ]
        )
    )


defs = dg.Definitions(
    assets=dg.with_source_code_references([final_gemma4_multi_agent_debateV2])
)
