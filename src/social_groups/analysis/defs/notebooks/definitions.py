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
}


def register_materialization(
    name: str,
    obj: Any,
    description: str,
):
    context = dagstermill.get_context()
    is_in_dagster = isinstance(
        context,
        dagstermill.context.DagstermillRuntimeExecutionContext,
    )

    if not is_in_dagster:
        print("Skipping Materialization because in interactive mode.")
        return

    notebook_name = context.op_name.split("__")[-1]  # Take away group names
    ext = global_notebook_registry[notebook_name]["extra_assets"][name]
    ExtraNotebookAsset(
        name=name, extension=ext, notebook_name=notebook_name
    ).register_materialization(
        obj,
        description=description,
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
)
