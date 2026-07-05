from abc import ABC
from typing import Optional

from pydantic import BaseModel, field_validator

from social_groups.orchestrator.models.slurm_config import SlurmConfiguration
from social_groups.orchestrator.services.base import (
    SlurmService,
)
from social_groups.orchestrator.utils.types import ModelId


class ModelConfiguration(BaseModel):
    port: int

    tool_call_parser: Optional[str] = "hermes"

    max_batch_prefill_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None

    tokenizer_mode: Optional[str] = None
    config_format: Optional[str] = None
    load_format: Optional[str] = None
    reasoning_parser: Optional[str] = None


class BaseInferenceService(SlurmService, ABC):
    llm_models: dict[ModelId, ModelConfiguration]
    _chosen_model_id: ModelId | None = None  # The model to actually run

    _chosen_gpus_per_model_instance: int = 1

    @field_validator("slurm_config")
    @classmethod
    def require_partition_specification(
        cls, v: SlurmConfiguration
    ) -> SlurmConfiguration:
        if v.partition is None:
            raise ValueError(
                "Please specify partition parameter for Inference Service."
            )
        return v

    def check_model_config_exists(self, to_test: list[ModelId]):
        all_model_ids = set(self.llm_models.keys())
        not_available = set(to_test) - all_model_ids
        if not_available:
            raise ValueError(
                "\nInvalid Model Ids: \n - "
                + "\n - ".join(sorted(not_available))
                + "\n\n>> Please add them to the yaml-config first."
                "\n\nAvailable models: \n - "
                + "\n - ".join(sorted(self.llm_models.keys()))
            )

    def set_used_model(
        self,
        modelid: ModelId,
        used_gpus: int | None = None,
        gpus_per_model_instance: int = 1,
    ):
        self.check_model_config_exists([modelid])
        self._chosen_model_id = modelid
        if used_gpus is not None:
            self.slurm_config.gres = f"gpu:{used_gpus}"
        if gpus_per_model_instance is not None:
            self.slurm_config.gpus_per_model_instance = gpus_per_model_instance
