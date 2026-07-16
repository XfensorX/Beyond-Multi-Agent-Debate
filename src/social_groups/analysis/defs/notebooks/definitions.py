import os
import warnings
from pathlib import Path
from typing import Any, TypedDict

import dagster as dg
import dagstermill

from social_groups.analysis.notebook_assets import (
    SUPPORTED_EXTENSION,
    ExtraNotebookAsset,
    create_notebook_asset,
)


class NotebookEntry(TypedDict, total=False):
    ins: dict[str, dg.AssetIn]
    extra_assets: dict[str, SUPPORTED_EXTENSION]


global_notebook_registry: dict[str, NotebookEntry] = {
    "baseline_analysis": {
        "ins": {"baseline_frame": dg.AssetIn("baseline")},
        "extra_assets": {"baseline_full_evaluation_table": "tex"},
    },
    "heterogeneous_comparison": {
        "ins": {
            "mad_frame": dg.AssetIn("hetero_mad"),
            "baseline_frame": dg.AssetIn("baseline"),
            "original_table_page46": dg.AssetIn(
                ["report", "external", "group_problem_solving_page_46"]
            ),
        },
        "extra_assets": {"heterogeneous_groups_evaluation_table": "tex"},
    },
    "thinking_mad_analysis": {
        "ins": {"thinking_frame": dg.AssetIn("thinking_mad")},
        "extra_assets": {
            "thinking_mad_evaluation_table": "tex",
            "mad_evaluation_table_per_group_constellation_with_different_thinking_models": "tex",
            "thinking_mad_comparison_colored": "svg",
        },
    },
    "diversity_params_analysis": {
        "ins": {"diversity_frame": dg.AssetIn("diversity_params_mad")},
        "extra_assets": {
            "diversity_params_mad_evaluation_table": "tex",
            "diversity_params_mad_evaluation_colored_table_figure": "svg",
        },
    },
    "changed_order_mad_analysis": {
        "ins": {"frame": dg.AssetIn("changed_order_mad")},
        "extra_assets": {"changed_order_mad_evaluation_table": "tex"},
    },
    "changed_prompt_mad_analysis": {
        "ins": {
            "changed_prompt": dg.AssetIn("changed_prompt_mad"),
            "original_prompt": dg.AssetIn("hetero_mad"),
        },
        "extra_assets": {"changed_prompts_mad_evaluation_table": "tex"},
    },
    "no_discussion_voting_analysis": {
        "ins": {"frame": dg.AssetIn("no_discussion_voting")},
        "extra_assets": {
            "no_discussion_voting_evaluation_table": "tex",
            "participants_number_influence_plot": "svg",
        },
    },
    "baseline_output_comparison_analysis": {
        "ins": {"baseline_output_frame": dg.AssetIn("baseline_output_comparison")},
        "extra_assets": {"baseline_approach_comparison_table": "tex"},
    },
    "final_no_discussion_voting": {
        "ins": {
            "baseline": dg.AssetIn("final_baseline"),
            "no_discussion_data": dg.AssetIn("final_no_discussion_voting"),
        },
        "extra_assets": {
            "no_discussion_voting_by_number_of_participants": "png",
            "entropy_accuracy_curve_for_different_participant_numbers_general_case": "png",
            "no_discussion_voting_mixed_model_sizes": "png",
            "no_discussion_voting_mixed_model_families": "png",
            "mixed_methods_vs_best_constituent_models_temp07": "png",
            "mixed_methods_vs_best_constituent_models_temp00": "png",
            "mixed_methods_vs_best_constituent_models_equal_or_less_cost_temp00": "png",
            "mixed_methods_vs_best_constituent_models_equal_or_less_cost_temp07": "png",
            "knowledge_blackhole_heatmap": "png",
            "no_discussion_voting_main_result_table": "parquet",
            "all_group_combinations_majority_voting": "parquet",
            "all_group_combinations_majority_voting_mixed_families": "parquet",
        },
    },
    "final_baseline_analysis": {
        "ins": {
            "final_baseline": dg.AssetIn("final_baseline"),
            "baseline_paper_reported": dg.AssetIn(
                ["report", "external", "baseline_paper_reported"]
            ),
        },
        "extra_assets": {
            "baseline_accuracy_overview_all_with_parsing": "tex",
            "baseline_method_comparison_plot": "png",
            "used_parsing_regular_expressions": "tex",
            "baseline_few_shot_comparison_plot": "png",
            "baseline_structural_black_hole_plot": "png",
            "baseline_comparison_with_reported_numbers": "tex",
            "baseline_standard_ordering_into_LMH_groups": "tex",
        },
    },
    "final_mad_baseline": {
        "ins": {
            "frame": dg.AssetIn("final_changed_order_mad"),
            "baseline": dg.AssetIn("final_baseline"),
            "no_discussion_baseline": dg.AssetIn(
                [
                    "report",
                    "final_no_discussion_voting",
                    "all_group_combinations_majority_voting",
                ]
            ),
        },
        "extra_assets": {
            "improvement_per_family_constellation_heatmap_plot_homogeneous_mad": "png",
            "main_results_homog_mad_formatted": "tex",
            "homogeneous_mad_cost_tradeoff_paretofront_plot": "png",
            "rho_matrix_group_constellation_correlation": "tex",
            "heterogeneous_group_composition_performance_against_baselines": "png",
            "mode_compositions_that_can_be_better_than_baselines": "tex",
            "baseline_pareto_frontiers_with_heterogeneous_points": "png",
            "accuracy_spread_by_group_ordering_per_composition": "tex",
            "wasserstein_distance_correlation_heterogeneous_mad": "png",
            "best_fitting_decision_scheme_across_heterogeneous_mad": "png",
            "decisionscheme_correlation_with_improvement_over_baselines": "tex",
            "homogeneous_mad_big_subset_result": "parquet",
            "size_heterogeneous_mad_big_subset_result": "parquet",
        },
    },
    "final_medium_subset_mad": {
        "ins": {
            "medium_mad": dg.AssetIn("final_medium_subset_mad"),
            "big_homo_mad": dg.AssetIn(
                ["report", "final_mad_baseline", "homogeneous_mad_big_subset_result"]
            ),
            "big_size_hetero_mad": dg.AssetIn(
                [
                    "report",
                    "final_mad_baseline",
                    "size_heterogeneous_mad_big_subset_result",
                ]
            ),
        },
        "extra_assets": {
            "accuracy_correlation_medium_and_big_subset": "tex",
            "accuracy_difference_from_big_to_medium_subset": "tex",
            "medium_subset_pareto_friend_by_group_size": "png",
            "accuracy_gain_vs_wasserstein_distance": "png",
            "medium_basic_mad_size_heterogeneity_result": "parquet",
        },
    },
    "final_changing_prompt": {
        "ins": {
            "changed_prompt_mad": dg.AssetIn("final_changed_prompt_mad"),
            "medium_mad": dg.AssetIn(
                [
                    "report",
                    "final_medium_subset_mad",
                    "medium_basic_mad_size_heterogeneity_result",
                ]
            ),
        },
        "extra_assets": {
            "accuracy_change_by_group_constellation": "png",
            "permutation_effect_within_each_model_family": "png",
            "changed_prompt_vs_standard_prompt": "png",
        },
    },
    "final_different_families": {
        "ins": {
            "different_families_mad": dg.AssetIn("final_different_families"),
            "baseline": dg.AssetIn("final_baseline"),
            "homogeneous_mad": dg.AssetIn(
                ["report", "final_mad_baseline", "homogeneous_mad_big_subset_result"]
            ),
            "size_heterogeneous_mad": dg.AssetIn(
                [
                    "report",
                    "final_mad_baseline",
                    "size_heterogeneous_mad_big_subset_result",
                ]
            ),
            "no_discussion_voting_same_size_mixed_families": dg.AssetIn(
                [
                    "report",
                    "final_no_discussion_voting",
                    "all_group_combinations_majority_voting_mixed_families",
                ]
            ),
            "no_discussion_voting_mixed_size_same_family": dg.AssetIn(
                [
                    "report",
                    "final_no_discussion_voting",
                    "all_group_combinations_majority_voting",
                ]
            ),
            "changed_order_mad": dg.AssetIn("final_changed_order_mad"),
        },
        "extra_assets": {
            "pareto_frontier_all_methods": "png",
            "plot_size_matched_performance": "png",
            "plot_correlation_heatmaps": "png",
            "plot_size_matched_wd": "png",
        },
    },
    "final_diversity_params": {
        "ins": {"diversity_mad_frame": dg.AssetIn("final_diversity_params")},
        "extra_assets": {
            "accuracy_vs_accuracy_increase_by_temperature_and_top_p": "png",
            "all_Results_diversity_parameter_mad": "tex",
            "delta_heatmap_by_group_constellation_and_model_family": "png",
            "delta_accuracy_vs_delta_accuracy_increase_by_temperature_and_top_p": "png",
        },
    },
    "final_thinking_mad": {
        "ins": {
            "thinking_mad_frame": dg.AssetIn("final_thinking_mad"),
            "diversity_mad": dg.AssetIn("final_diversity_params"),
        },
        "extra_assets": {
            "sensitivity_to_open_thinking_configuration": "png",
            "influence_of_open_thinking_configuration_unordered_configuration": "png",
            "influence_of_open_thinking_config_for_LH_groups": "png",
        },
    },
    "final_gemma4": {
        "ins": {
            "gemma4_single_model_baseline": dg.AssetIn(
                "final_gemma4_single_model_baseline"
            ),
            "gemma4_no_discussion_voting": dg.AssetIn(
                "final_gemma4_no_discussion_voting_base"
            ),
            "gemma4_multi_agent_debate": dg.AssetIn("final_gemma4_multi_agent_debate"),
        },
    },
}


def register_materialization(
    name: str,
    obj: Any,
    description: str,
    force: bool = False,
):

    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        context = dagstermill.get_context()
        is_in_dagster = isinstance(
            context,
            dagstermill.context.DagstermillRuntimeExecutionContext,
        )

        if not force and not is_in_dagster:
            print("Skipping Materialization because in interactive mode.")
            return
        if force:
            notebook_name = Path(os.getenv("JPY_SESSION_NAME") or "___INVALID___").stem
        else:
            notebook_name = context.op_name.split("__")[-1]  # Take away group names

        ext = global_notebook_registry[notebook_name]["extra_assets"][name]
        ExtraNotebookAsset(
            name=name, extension=ext, notebook_name=notebook_name
        ).register_materialization(
            obj, description=description, print_output_path=force
        )


defs = dg.Definitions(
    assets=[
        asset
        for file, info in global_notebook_registry.items()
        for asset in create_notebook_asset(
            f"{file}.ipynb",
            ins=info["ins"] if "ins" in info else {},
            extra_assets=[
                ExtraNotebookAsset(name=name, extension=ext, notebook_name=file)
                for name, ext in info["extra_assets"].items()
            ]
            if "extra_assets" in info
            else None,
        )
    ],
    executor=dg.multiprocess_executor.configured({"max_concurrent": 2}),
)
