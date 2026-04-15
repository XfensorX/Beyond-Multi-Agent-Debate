import dagster as dg
import polars as pl
from dagster import AssetCheckSpec

import social_groups.polars_columns as plc
from social_groups.analysis.asset_checks import (
    check_correct_data_length,
    check_unique_data_connector,
)
from social_groups.analysis.polars_transformations import make_group_constellation
from social_groups.analysis.polars_transformations.deserialize_experiment_configuration import (
    deserialize_experiment_configuration,
)
from social_groups.analysis.polars_transformations.make_group_constellation import (
    make_model_family,
)


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="final_experiments",
    deps=["combined_data"],
    check_specs=[
        AssetCheckSpec(
            name="check_unique_data_connector",
            asset="final_changed_order_mad",
            blocking=True,
        ),
        AssetCheckSpec(
            name="check_correct_data_length",
            asset="final_changed_order_mad",
            blocking=True,
        ),
    ],
)
def final_changed_order_mad(combined_data: pl.DataFrame):
    final_changed_order_mad = (
        combined_data.filter(
            (
                pl.col("name").is_in(
                    {
                        "changed_order_mad2",
                        "changed_order_mad3",
                        # TODO: what to do with Mad4?
                    }
                )
            )
            & (pl.col("data_connector") == "mmlu-pro-big-subset")
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
        )
        .with_columns(
            make_group_constellation(),
            make_model_family(),
        )
        .drop("_experiment_configuration")
    )

    yield check_unique_data_connector(final_changed_order_mad)
    yield check_correct_data_length(
        final_changed_order_mad, 54000
    )  # TODO: change when mad3s are done

    yield dg.Output(
        final_changed_order_mad.drop(
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


defs = dg.Definitions(assets=dg.with_source_code_references([final_changed_order_mad]))
