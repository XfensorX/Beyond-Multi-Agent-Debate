from __future__ import annotations

from social_groups.analyzer.models.base import PolarsBaseModel


class Experiment(PolarsBaseModel):
    experiment_id: int
    name: str

    @classmethod
    def from_raw_data(cls, *, assigned_id: int, experiment_name: str) -> Experiment:
        return cls(experiment_id=assigned_id, name=experiment_name)
