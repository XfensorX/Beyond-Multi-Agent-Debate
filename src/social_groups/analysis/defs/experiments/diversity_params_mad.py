import json

import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

import social_groups.polars_columns as plc
from social_groups.analysis.asset_checks import check_unique_data_connector
from social_groups.analysis.polars_transformations import make_group_constellation
from social_groups.general.utils.standard_library import unique_item
from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="mad_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector",
            asset="diversity_params_mad",
            blocking=True,
        )
    ],
)
def diversity_params_mad(combined_data: pl.DataFrame):
    frame = (
        combined_data.filter(
            pl.col("name").is_in(
                {
                    "diversity_parameter_sweep_mad2_step1",
                    "diversity_parameter_sweep_mad2_step2",
                    "diversity_parameter_sweep_mad3_step1",
                    "diversity_parameter_sweep_mad3_step2",
                    "diversity_parameter_sweep_mad3_step3",
                    "diversity_parameter_sweep_mad3_step4",
                }
            )
        )
        .with_columns(
            model_names=pl.col(plc.experiment_configuration_json).map_elements(
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
            temperature=pl.col(plc.experiment_configuration_json).map_elements(
                lambda x: unique_item(
                    map(
                        lambda d: d.params.temperature,
                        ExperimentConfig.model_validate(
                            json.loads(x)
                        ).strategy.configuration.debate_agents,
                    )
                ),
                return_dtype=pl.Float64,
            ),
            top_p=pl.col(plc.experiment_configuration_json).map_elements(
                lambda x: unique_item(
                    map(
                        lambda d: d.params.top_p,
                        ExperimentConfig.model_validate(
                            json.loads(x)
                        ).strategy.configuration.debate_agents,
                    )
                ),
                return_dtype=pl.Float64,
            ),
        )
        .with_columns(make_group_constellation())
    )

    yield check_unique_data_connector(frame)

    yield dg.Output(
        frame.drop(
            [
                "final_answer",  # is empty in group case
                plc.model_names,  # group constellation is better
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


defs = dg.Definitions(assets=dg.with_source_code_references([diversity_params_mad]))
