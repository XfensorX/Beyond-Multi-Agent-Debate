import dagster as dg
import polars as pl

from social_groups.analysis.notebook_assets import ExtraNotebookAsset


@dg.asset(
    key=dg.AssetKey(["report", "external", "baseline_paper_reported"]),
    group_name="external",
    metadata={"file_extension": "tex"},
    description="The reported performance of different models reported in their respective papers/technical reports.",
    io_manager_key="polars_parquet_io_manager",
)
def baseline_paper_reported():
    numbers = pl.DataFrame(
        [
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3-14B Base",
                "Accuracy": "61.03",
                "Source": "Qwen-3 Technical Report Table 5 (p. 7)",
            },
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3-8B Base",
                "Accuracy": "56.73",
                "Source": "Qwen-3 Technical Report Table 6 (p. 7)",
            },
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3-235B-A22B Base",
                "Accuracy": "68.18",
                "Source": "Qwen-3 Technical Report Table 3 (p. 6)",
            },
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3-30B-A3B",
                "Accuracy": "61.49",
                "Source": "Qwen-3 Technical Report Table 5 (p. 7)",
            },
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3-32B Base",
                "Accuracy": "65.54",
                "Source": "Qwen-3 Technical Report Table 4 (p. 6)",
            },
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3-4B Base",
                "Accuracy": "50.58",
                "Source": "Qwen-3 Technical Report Table 7 (p. 8)",
            },
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3-0.6B Base",
                "Accuracy": "24.74",
                "Source": "Qwen-3 Technical Report Table 8 (p. 8)",
            },
            {
                "Dataset": "MMLUPro",
                "Model": "Qwen3.5-397B-A17B",
                "Accuracy": "87.8",
                "Source": "Qwen 3.5 Release Blog",
            },
        ]
    )

    asset = ExtraNotebookAsset(
        name=baseline_paper_reported.key.parts[-1],
        extension="tex",
        notebook_name="external",
    )

    asset.register_materialization(numbers, "Dropped Description")

    return dg.Output(
        numbers,
        metadata={
            "path": asset._get_path(),
            "file_name": asset.name,
            "file_extension": asset.extension,
            "preview": asset._get_preview(numbers),
        },
    )


defs = dg.Definitions(
    assets=[baseline_paper_reported],
)
