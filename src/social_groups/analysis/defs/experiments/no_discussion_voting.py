import json

import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

import src.social_groups.polars_columns as plc
from social_groups.analysis.asset_checks import (
    check_single_model_only,
    check_unique_data_connector,
)
from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="baseline_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector",
            asset="no_discussion_voting",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_single_model_only",
            asset="no_discussion_voting",
            blocking=True,
        ),
    ],
)
def no_discussion_voting(combined_data: pl.DataFrame):
    frame = combined_data.filter(
        pl.col("name").is_in({"no_discussion_voting"})
        & pl.col("data_connector").is_in({"mmlu-pro-subset"})
    ).with_columns(
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
        .alias(plc.model_names),
        temperature=pl.col(plc.experiment_configuration_json)
        .map_elements(
            lambda x: list(
                map(
                    lambda d: d.params.temperature,
                    ExperimentConfig.model_validate(
                        json.loads(x)
                    ).strategy.configuration.debate_agents,
                )
            ),
            return_dtype=pl.List(pl.Utf8),
        )
        .list.item(),
    )

    yield check_unique_data_connector(frame)
    yield check_single_model_only(frame)

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


defs = dg.Definitions(assets=dg.with_source_code_references([no_discussion_voting]))
