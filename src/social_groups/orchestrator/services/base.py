from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, computed_field

from social_groups.directories import ORCHESTRATION_CONFIGS_DIR
from social_groups.general.utils.registry import Registry
from social_groups.general.utils.standard_library import (
    import_all_submodules,
    make_enum,
)
from social_groups.orchestrator import services
from social_groups.orchestrator.config import YAML_ENDING
from social_groups.orchestrator.models.execution_environment import (
    ExecutionLocation,
    ExecutionLocationConfig,
)
from social_groups.orchestrator.models.slurm_config import SlurmConfiguration
from social_groups.orchestrator.utils.general import make_exported_variables_block


class SlurmService(BaseModel, ABC):
    slurm_config: SlurmConfiguration

    @computed_field
    @property
    def used_job_name(self) -> str | None:
        return None

    @staticmethod
    @abstractmethod
    def job_name_is_matching_this_service(given_job_name: str) -> bool: ...

    @abstractmethod
    def create_env_dict(
        self, exec_config: ExecutionLocationConfig
    ) -> dict[str, Any]: ...

    @abstractmethod
    def create_run_command(self, exec_config: ExecutionLocationConfig) -> str: ...

    def create_job_file_content(self, exec_config: ExecutionLocationConfig) -> str:
        if self.used_job_name is None:
            raise ValueError("Must specifiy a jobname for the service.")

        return (
            self.slurm_config.create_batch_file_header(self.used_job_name)
            + "\n\n\n"
            + make_exported_variables_block(self.create_env_dict(exec_config))
            + "\n\n\n"
            + self.create_run_command(exec_config)
        )


SLURM_SERVICE: Registry[type[SlurmService]] = Registry("slurm service")


def _validate_slurm_service(cls: type[SlurmService]):
    if not isinstance(cls, type) or not issubclass(cls, SlurmService):
        raise TypeError("Only SlurmService subclasses can be registered")


def register_slurm_service(name):
    return SLURM_SERVICE.register(name, validate=_validate_slurm_service)


import_all_submodules(services, ignore_prefix="base")
SlurmServiceName: type[Enum] = make_enum("SlurmServiceName", set(SLURM_SERVICE.names()))


def load_config(service: SlurmServiceName, where: ExecutionLocation) -> SlurmService:
    config_path = (
        ORCHESTRATION_CONFIGS_DIR / where.value / f"{service.value}{YAML_ENDING}"
    )
    with open(config_path) as f:
        return SLURM_SERVICE.get(service).model_validate(yaml.safe_load(f))
