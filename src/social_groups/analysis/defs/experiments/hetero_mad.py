import json

import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

import src.social_groups.polars_columns as plc
from social_groups.analysis.asset_checks import (
    check_standard_group_constellations,
    check_unique_data_connector,
)
from social_groups.analysis.polars_transformations import make_group_constellation
from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector", asset="hetero_mad", blocking=True
        ),
        AssetCheckSpec(
            name="check_standard_group_constellations",
            asset="hetero_mad",
            blocking=True,
        ),
    ],
)
def hetero_mad(combined_data: pl.DataFrame):
    frame = combined_data.filter(
        pl.col("name").is_in(
            {
                "heterogeneous_group_mad1",
                "heterogeneous_group_mad2",
                "heterogeneous_group_mad3",
                "heterogeneous_group_mad4",
            }
        )
    ).with_columns(
        model_names=pl.col(plc.experiment_configuration_json)
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

    yield check_standard_group_constellations(
        frame.with_columns(make_group_constellation())
    )
    yield check_unique_data_connector(frame)

    yield dg.Output(
        frame.drop(
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
    )


defs = dg.Definitions(assets=dg.with_source_code_references([hetero_mad]))
