import json

import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(name="has_correct_dataset", asset="hetero_mad", blocking=True),
        AssetCheckSpec(
            name="not_more_runs_than_needed", asset="hetero_mad", blocking=True
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
        )
    )

    run_idfs = frame["run_identifier"].unique()
    data_conns = set(frame["data_connector"].unique())

    yield dg.AssetCheckResult(
        check_name="has_correct_dataset",
        passed=bool(data_conns == {"mmlu-pro-subset"}),
        metadata={"data_connectors": list(data_conns)},
    )

    yield dg.AssetCheckResult(
        check_name="not_more_runs_than_needed",
        passed=bool(4 == run_idfs.len()),
        metadata={"run_idfs": run_idfs.to_list()},
    )

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
