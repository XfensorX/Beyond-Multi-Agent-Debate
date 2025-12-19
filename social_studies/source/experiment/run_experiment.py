from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from config import results_dir
from data_connectors.mmlu_pro import run_test_set
from main_registry import DECISION_SCHEMES, DecisionSchemeName


class DataSource(Enum):
    MMLUPro = "MMLUPro"


class MetaInformation(BaseModel):
    description: str


class ExperimentConfig(BaseModel):
    name: str
    results_subdir: Path
    data: DataSource
    strategy: DecisionSchemeName
    meta_information: MetaInformation
    # TODO: add parameters of decision_scheme


class ExperimentInfo(BaseModel):
    results_dir: Path


def run_experiment(experiment_config: ExperimentConfig) -> ExperimentInfo:
    try:
        decision_scheme = DECISION_SCHEMES.get(experiment_config.strategy)
    except KeyError:
        raise NotImplementedError(
            f"Strategy '{experiment_config.strategy}' not implemented or registered."
        )

    match experiment_config.data:
        case DataSource.MMLUPro:
            run_test_set(decision_scheme(), experiment_config.name)
        case _:
            raise NotImplementedError(f"{experiment_config.data} is not implemented.")

    return ExperimentInfo(
        results_dir=results_dir() / Path(experiment_config.results_subdir)
    )
