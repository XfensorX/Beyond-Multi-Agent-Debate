from dataclasses import dataclass

import polars as pl

from social_groups.directories import PARQUET_ANALYSIS_DIR


@dataclass
class Data:
    questions: pl.DataFrame
    answers: pl.DataFrame
    experiments: pl.DataFrame
    runs: pl.DataFrame


def load_tables():
    return Data(
        questions=pl.read_parquet(PARQUET_ANALYSIS_DIR / "Question.parquet"),
        answers=pl.read_parquet(PARQUET_ANALYSIS_DIR / "Answer.parquet"),
        experiments=pl.read_parquet(PARQUET_ANALYSIS_DIR / "Experiment.parquet"),
        runs=pl.read_parquet(PARQUET_ANALYSIS_DIR / "Run.parquet"),
    )


def build_analysis_frame():
    data = load_tables()

    df = (
        data.answers.join(data.runs, on="run_id", how="left")
        .join(data.experiments, on="experiment_id", how="left")
        .join(data.questions, on="question_id", how="left")
    )
    return df
