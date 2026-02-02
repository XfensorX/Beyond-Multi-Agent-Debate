from __future__ import annotations

from social_groups.analyzer.models.base import PolarsBaseModel
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo


class Run(PolarsBaseModel):
    run_id: int
    experiment_id: int

    experiment_configuration_json: str

    execution_config_json: str
    meta_info_json: str

    @classmethod
    def from_raw_data(
        cls,
        *,
        assigned_id: int,
        hydra_config: MainConfig,
        meta_info: ExperimentMetaInfo,
        experiment_id: int,
    ) -> Run:
        return cls(
            run_id=assigned_id,
            experiment_id=experiment_id,
            execution_config_json=hydra_config.execution.model_dump_json(),
            meta_info_json=meta_info.model_dump_json(),
            experiment_configuration_json=hydra_config.experiment.model_dump_json(),
        )
