import dagster as dg
import polars as pl
from dagster import AssetKey, FilesystemIOManager, asset
from dagster_polars import PolarsParquetIOManager
from dagstermill import ConfigurableLocalOutputNotebookIOManager

from social_groups.directories import DAGSTER_BASE_DIR, PARQUET_ANALYSIS_DIR


@asset(key_prefix=["raw"], io_manager_key="polars_parquet_io_manager", group_name="raw")
def answers() -> pl.LazyFrame:
    return pl.scan_parquet(PARQUET_ANALYSIS_DIR / "Answer.parquet")


@asset(key_prefix=["raw"], io_manager_key="polars_parquet_io_manager", group_name="raw")
def experiments() -> pl.LazyFrame:
    return pl.scan_parquet(PARQUET_ANALYSIS_DIR / "Experiment.parquet")


@asset(key_prefix=["raw"], io_manager_key="polars_parquet_io_manager", group_name="raw")
def questions() -> pl.LazyFrame:
    return pl.scan_parquet(PARQUET_ANALYSIS_DIR / "Question.parquet")


@asset(key_prefix=["raw"], io_manager_key="polars_parquet_io_manager", group_name="raw")
def runs() -> pl.LazyFrame:
    return pl.scan_parquet(PARQUET_ANALYSIS_DIR / "Run.parquet")


@dg.asset(
    io_manager_key="polars_parquet_io_manager",
    group_name="raw",
    deps=[
        AssetKey(["raw", "answers"]),
        AssetKey(["raw", "runs"]),
        AssetKey(["raw", "experiments"]),
        AssetKey(["raw", "questions"]),
    ],
)
def combined_data(
    answers: pl.LazyFrame,
    runs: pl.LazyFrame,
    experiments: pl.LazyFrame,
    questions: pl.LazyFrame,
) -> pl.DataFrame:
    return (
        answers.join(runs, on="run_id", how="left")
        .join(experiments, on="experiment_id", how="left")
        .join(questions, on="question_id", how="left")
    ).collect()


defs = dg.Definitions(
    assets=dg.with_source_code_references(
        [answers, experiments, questions, runs, combined_data]
    ),
    resources={
        "polars_parquet_io_manager": PolarsParquetIOManager(
            base_dir=str(DAGSTER_BASE_DIR)
        ),
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(
            base_dir=str(DAGSTER_BASE_DIR)
        ),
        "io_manager": FilesystemIOManager(base_dir=str(DAGSTER_BASE_DIR)),
    },
)
