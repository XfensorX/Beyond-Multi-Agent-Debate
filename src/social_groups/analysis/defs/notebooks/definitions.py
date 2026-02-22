import dagster as dg
from dagster import (
    file_relative_path,
)
from dagstermill import define_dagstermill_asset

from social_groups.directories import DAGSTER_BASE_DIR


def create_notebook_asset(file_name: str, deps: list[str], ins: dict[str, dg.AssetIn]):
    return define_dagstermill_asset(
        name=file_name.split(".")[0],
        notebook_path=file_relative_path(__file__, file_name),
        group_name="notebooks",
        deps=deps,
        ins=ins,
        metadata={
            "Original Notebook": dg.MetadataValue.notebook(
                file_relative_path(__file__, file_name)
            ),
            "dagster/code_references": dg.CodeReferencesMetadataValue(
                code_references=[
                    dg.LocalFileCodeReference(
                        file_path=file_relative_path(__file__, file_name),
                        line_number=0,
                        label="Original Notebook",
                    ),
                    dg.LocalFileCodeReference(
                        file_path=str(DAGSTER_BASE_DIR / file_name),
                        line_number=0,
                        label="Run Notebook",
                    ),
                ]
            ),
        },
    )


defs = dg.Definitions(
    assets=[
        create_notebook_asset(
            "baseline_analysis.ipynb",
            deps=["baseline"],
            ins={"baseline_frame": dg.AssetIn("baseline")},
        ),
        create_notebook_asset(
            "heterogeneous_comparison.ipynb",
            deps=["hetero_mad", "baseline"],
            ins={
                "mad_frame": dg.AssetIn("hetero_mad"),
                "baseline_frame": dg.AssetIn("baseline"),
            },
        ),
        create_notebook_asset(
            "thinking_mad_analysis.ipynb",
            deps=["thinking_mad"],
            ins={"thinking_frame": dg.AssetIn("thinking_mad")},
        ),
        create_notebook_asset(
            "diversity_params.ipynb",
            deps=["diversity_params_mad"],
            ins={"diversity_frame": dg.AssetIn("diversity_params_mad")},
        ),
    ]
)
