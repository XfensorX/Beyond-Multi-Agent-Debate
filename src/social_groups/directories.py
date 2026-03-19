from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent.parent

TEX_PROJECT_REPORT_DIR = PROJECT_DIR.parent / "TexProject" / "data" / "report"


RESULTS_DIR = PROJECT_DIR / "results"
PHOENIX_CACHE_DIR = RESULTS_DIR / "phoenix" / "CACHE"
MULTIRUN_FINAL_RESULTS_DIR = RESULTS_DIR / "multirun" / "final"
RUNS_FINAL_RESULTS_DIR = RESULTS_DIR / "runs" / "final"

ANALYSIS_DIR = RESULTS_DIR / "analysis"
PARQUET_ANALYSIS_DIR = ANALYSIS_DIR / "parquet"
DAGSTER_BASE_DIR = ANALYSIS_DIR / "dagster"
DAGSTER_REPORT_DIR = DAGSTER_BASE_DIR / "report"
REPORTING_DIR = ANALYSIS_DIR / "reporting"


CONFIGS_DIR = PROJECT_DIR / "configs"
ORCHESTRATION_CONFIGS_DIR = CONFIGS_DIR / "orchestration"


LOG_FILE_NAME = "stdout.log"
META_FILE_NAME = "meta.yaml"
TRACK_FILE_NAME = "experiment_result.jsonl"
TRACK_FILE_NAME_COMPRESSED = TRACK_FILE_NAME + ".zst"
