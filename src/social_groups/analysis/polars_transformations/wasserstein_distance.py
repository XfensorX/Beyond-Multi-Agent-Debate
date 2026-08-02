import traceback
from functools import partial

import polars as pl

import social_groups.polars_columns as plc
from social_groups.reporting.decision_scheme_comparison import (
    wasserstein_transition_distance_to_optimal,
)
from social_groups.reporting.group_decision_scheme import (
    calculate_decision_scheme_matrix,
)


def calculate_wasserstein_distance_withd_different_biases(
    group_reply, comparer, filtered_frame: pl.DataFrame
):
    try:
        matrix = calculate_decision_scheme_matrix(
            filtered_frame,
            plc.parsed_individual_answers_before,
            plc.parsed_individual_answers_after,
            plc.answer_string,
            group_reply,
            comparer,
        )

        filtered_frame = filtered_frame.with_columns(
            comparer(
                pl.col(plc.parsed_combined_answers_before),
                pl.col(plc.answer_string),
            ).alias("is_correct_before")
        )

        metadata = (
            filtered_frame.select(
                [
                    name
                    for name, is_constant in filtered_frame.select(
                        pl.all().n_unique() == 1
                    )
                    .to_dicts()[0]
                    .items()
                    if is_constant
                ]
            )
            .unique()
            .to_dicts()[0]
        )

        for bias in ["good proposals", "bad proposals", "weight on one", False]:
            metadata[f"wd_bias_{bias}"] = wasserstein_transition_distance_to_optimal(
                matrix, biased=bias
            )

        metadata[plc.accuracy] = filtered_frame[plc.is_correct].mean()
        metadata["accuracy_before"] = filtered_frame["is_correct_before"].mean()
        metadata["accuracy_increase"] = (
            metadata[plc.accuracy] - metadata["accuracy_before"]
        )
        metadata["accuracy_increase_percent"] = (
            metadata[plc.accuracy] / metadata["accuracy_before"]
        ) - 1

        return pl.DataFrame([metadata])
    except Exception as e:
        print("FAILED GROUP:")
        print(filtered_frame)
        traceback.print_exc()
        raise e


def calculate_wasserstein_distance(
    frame: pl.DataFrame, group_reply, comparer, unique_identifier_column_name: str
):
    wasserstein_mad = frame.group_by(unique_identifier_column_name).map_groups(
        partial(
            calculate_wasserstein_distance_withd_different_biases, group_reply, comparer
        )
    )

    return wasserstein_mad
