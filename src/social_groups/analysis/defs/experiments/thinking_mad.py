import json

import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

import social_groups.polars_columns as plc
import social_groups.polars_values as plv
from social_groups.analysis.asset_checks import (
    check_standard_group_constellations,
    check_unique_data_connector,
)
from social_groups.analysis.polars_transformations import make_group_constellation
from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="creative_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector", asset="thinking_mad", blocking=True
        ),
        AssetCheckSpec(
            name="check_standard_group_constellations",
            asset="thinking_mad",
            blocking=True,
        ),
    ],
)
def thinking_mad(combined_data: pl.DataFrame):
    frame = combined_data.filter(
        pl.col("name").is_in({"het_group_thinking_mad2", "het_group_thinking_mad3"})
    ).with_columns(
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
        ),
        thinking_models=pl.col("experiment_configuration_json").map_elements(
            lambda x: list(
                map(
                    lambda d: d.model_name,
                    ExperimentConfig.model_validate(
                        json.loads(x)
                    ).strategy.configuration.openly_thinking_models,
                )
            ),
            return_dtype=pl.List(pl.Utf8),
        ),
    )

    frame = frame.extend(
        combined_data.filter(
            pl.col("name").is_in(
                {"heterogeneous_group_mad2", "heterogeneous_group_mad3"}
            )
        )
        .with_columns(
            pl.col(plc.experiment_configuration_json)
            .map_elements(
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
            .alias(plc.model_names)
        )
        .with_columns(pl.col(plc.model_names).alias(plc.thinking_models)),
    )

    yield check_unique_data_connector(frame)

    # format the "thinking" models column
    frame = frame.with_columns(
        (
            pl.when(pl.col(plc.thinking_models) == pl.col(plc.model_names))
            .then(pl.lit(plv.thinking_models.everyone))
            .otherwise(
                pl.when(pl.col(plc.thinking_models).list.len() == 0)
                .then(pl.lit(plv.thinking_models.nobody))
                .otherwise(pl.col(plc.thinking_models).list.join(" "))
            )
            .replace(plv.MODEL_NAME_TO_LETTER_MAPPING)
            .alias(plc.thinking_models)
        ),
        make_group_constellation(),
    )

    yield check_standard_group_constellations(frame)

    yield dg.Output(
        frame.drop(
            [
                plc.model_names,  # reformatted to group_constellation
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
    )


defs = dg.Definitions(assets=dg.with_source_code_references([thinking_mad]))
