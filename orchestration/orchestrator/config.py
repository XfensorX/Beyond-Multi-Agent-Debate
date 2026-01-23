from __future__ import annotations

from pathlib import Path

import yaml

from orchestration.orchestrator.models.execution_environment import (
    ExecutionConfig,
    ExecutionEnvironment,
)
from orchestration.orchestrator.services.base import (
    SLURM_SERVICE,
    SlurmService,
    SlurmServiceName,
)

YAML_ENDING = ".yaml"
EXECUTION_ENVIRONMENT_CONFIG_NAME = "general" + YAML_ENDING
CONFIGURATIONS_PATH = Path(__file__).parent.parent / "configurations"


def load_config(service: SlurmServiceName, where: ExecutionEnvironment) -> SlurmService:
    config_path = CONFIGURATIONS_PATH / where.value / f"{service.value}{YAML_ENDING}"
    with open(config_path) as f:
        return SLURM_SERVICE.get(service).model_validate(yaml.safe_load(f))


def load_execution_config(where: ExecutionEnvironment) -> ExecutionConfig:
    file_path = CONFIGURATIONS_PATH / where.value / EXECUTION_ENVIRONMENT_CONFIG_NAME
    with open(file_path) as f:
        return ExecutionConfig.model_validate(yaml.safe_load(f))
