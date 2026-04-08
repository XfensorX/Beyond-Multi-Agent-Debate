import json

import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="baseline_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(name="has_correct_dataset", asset="baseline", blocking=True),
        AssetCheckSpec(name="only_one_exp_id", asset="baseline", blocking=True),
        AssetCheckSpec(name="only_one_exp_name", asset="baseline", blocking=True),
    ],
)
def baseline(combined_data: pl.DataFrame):
    frame = combined_data.filter(
        pl.col("name").is_in({"heterogeneous_group_baseline"})
        & pl.col("data_connector").is_in({"mmlu-pro-subset"})
    ).with_columns(
        model_name=pl.col("experiment_configuration_json").map_elements(
            lambda x: (
                ExperimentConfig.model_validate(
                    json.loads(x)
                ).strategy.configuration.backend.model_name
            ),
            return_dtype=pl.Utf8,
        )
    )

    exp_ids = frame["experiment_id"].unique()
    exp_names = frame["name"].unique()
    data_conns = set(frame["data_connector"].unique())

    yield dg.AssetCheckResult(
        check_name="has_correct_dataset",
        passed=bool(data_conns == {"mmlu-pro-subset"}),
        metadata={"data_connectors": list(data_conns)},
    )

    yield dg.AssetCheckResult(
        check_name="only_one_exp_id",
        passed=bool(1 == exp_ids.len()),
        metadata={"exp_ids": exp_ids.to_list()},
    )

    yield dg.AssetCheckResult(
        check_name="only_one_exp_name",
        passed=bool(1 == exp_names.len()),
        metadata={"exp_names": exp_names.to_list()},
    )

    yield dg.Output(
        frame.drop(
            [
                "answers_at_beginning",  # empty
                "answers_at_end",  # empty
                "experiment_id",  # just one
                "name",  # just one
                "message_ids",  # not filled
                "phoenix_span_id",  # not interesting
                "data_connector",  # all "mmlu-pro-subset"
                "original_source",  # not interesting
                "experiment_configuration_json",  # already used
                "meta_info_json",
                "execution_config_json",
                "answer_index",
                "answer_options",
            ]
        )
    )


defs = dg.Definitions(assets=dg.with_source_code_references([baseline]))
