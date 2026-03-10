import dagster as dg
import polars as pl

from social_groups.analysis.notebook_assets import ExtraNotebookAsset


@dg.asset(
    key=dg.AssetKey(["report", "external", "group_problem_solving_page_46"]),
    group_name="external",
    metadata={"file_extension": "tex"},
    description="The group performance of different group constellation on a group problem solving task, from the book Group Problem Solving.",
)
def group_problem_solving_page_46():
    original_table_page46 = pl.DataFrame(
        {
            "group_constellation": [
                "HHH",
                "HHM",
                "HHL",
                "HML",
                "HMM",
                "H",
                "HLL",
                "MMM",
                "M",
                "MML",
                "MLL",
                "L",
                "LLL",
            ],
            "score (-115 to 115)": [80, 74, 67, 64, 61, 60, 56, 48, 42, 39, 37, 25, 21],
        }
    )

    asset = ExtraNotebookAsset(
        name=group_problem_solving_page_46.key.parts[-1],
        extension="tex",
        notebook_name="external",
    )

    asset.register_materialization(original_table_page46, "Dropped Description")

    return dg.Output(
        None,  # side effect asset
        metadata={
            "path": asset._get_path(),
            "file_name": asset.name,
            "file_extension": asset.extension,
            "preview": asset._get_preview(original_table_page46),
        },
    )


defs = dg.Definitions(
    assets=[group_problem_solving_page_46],
)
