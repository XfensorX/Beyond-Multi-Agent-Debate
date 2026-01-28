from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent.parent

RESULTS_DIR = PROJECT_DIR / "results"
MULTIRUN_FINAL_RESULTS_DIR = RESULTS_DIR / "multirun" / "final"
RUNS_FINAL_RESULTS_DIR = RESULTS_DIR / "runs" / "final"

CONFIGS_DIR = PROJECT_DIR / "configs"
ORCHESTRATION_CONFIGS_DIR = CONFIGS_DIR / "orchestration"


LOG_FILE_NAME = "stdout.log"
META_FILE_NAME = "meta.yaml"
TRACK_FILE_NAME = "experiment_result.jsonl"
