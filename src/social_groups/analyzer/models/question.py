from __future__ import annotations

import hashlib
import json
from typing import Any

from social_groups.analyzer.models.base import PolarsBaseModel
from social_groups.trialrunner.data_connectors.mmlu_pro import (
    MMLUProExample,
)
from social_groups.trialrunner.experiment.main_registry import DataConnectorName
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo
from social_groups.trialrunner.utils.tracking import TrackEntry


class Question(PolarsBaseModel):
    question_id: int
    original_question_id: int

    data_connector: DataConnectorName
    original_source: str
    category: str
    question: str

    # TODO: Rename this to "correct_ ..."
    answer_options: list[str]
    answer_index: int
    answer_string: str

    @classmethod
    def from_raw_data(
        cls,
        *,
        assigned_id: int,
        entry: TrackEntry,
        hydra_config: MainConfig,
        meta_info: ExperimentMetaInfo,
        span_attributes: dict[str, Any],
    ) -> Question:
        match hydra_config.experiment.data.value:
            case "mmlu-pro" | "mmlu-pro-subset":
                question = MMLUProExample.model_validate(
                    json.loads(span_attributes["question"])
                )

                return cls(
                    question_id=assigned_id,
                    original_question_id=question.question_id,
                    data_connector=hydra_config.experiment.data,
                    original_source=question.src,
                    category=question.category.value,
                    question=entry.input.question,
                    answer_options=question.options,
                    answer_index=question.answer_index,
                    answer_string=question.answer,
                )
            case _:
                raise NotImplementedError("Data Connector not Implemented yet")

    def get_id_independent_hash(self) -> bytes:
        payload = json.dumps(
            self.model_dump(exclude={"question_id"}, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).digest()
