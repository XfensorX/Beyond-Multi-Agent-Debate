import dagster as dg

from social_groups.analysis.notebook_assets import (
    ExtraNotebookAsset,
    create_notebook_asset,
)

defs = dg.Definitions(
    assets=[
        *create_notebook_asset(
            "baseline_analysis.ipynb",
            ins={"baseline_frame": dg.AssetIn("baseline")},
        ),
        *create_notebook_asset(
            "heterogeneous_comparison.ipynb",
            ins={
                "mad_frame": dg.AssetIn("hetero_mad"),
                "baseline_frame": dg.AssetIn("baseline"),
            },
        ),
        *create_notebook_asset(
            "thinking_mad_analysis.ipynb",
            ins={"thinking_frame": dg.AssetIn("thinking_mad")},
        ),
        *create_notebook_asset(
            "diversity_params_analysis.ipynb",
            ins={"diversity_frame": dg.AssetIn("diversity_params_mad")},
        ),
        *create_notebook_asset(
            "changed_order_mad_analysis.ipynb",
            ins={"frame": dg.AssetIn("changed_order_mad")},
        ),
        *create_notebook_asset(
            "changed_prompt_mad_analysis.ipynb",
            ins={
                "changed_prompt": dg.AssetIn("changed_prompt_mad"),
                "original_prompt": dg.AssetIn("hetero_mad"),
            },
        ),
        *create_notebook_asset(
            "no_discussion_voting_analysis.ipynb",
            ins={"frame": dg.AssetIn("no_discussion_voting")},
        ),
        *create_notebook_asset(
            "baseline_output_comparison_analysis.ipynb",
            ins={"baseline_output_frame": dg.AssetIn("baseline_output_comparison")},
            extra_assets=[
                ExtraNotebookAsset(
                    name="baseline_approach_comparison_table", extension="tex"
                )
            ],
        ),
    ],
)
