from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from social_groups.analyzer.models.base import PolarsBaseModel
from social_groups.general.tracking import TrackEntry
from social_groups.trialrunner.data_connectors.gpqa_diamond import GPQADiamondExample
from social_groups.trialrunner.data_connectors.mmlu_pro import (
    MMLUProExample,
)
from social_groups.trialrunner.experiment.main_registry import (
    DATA_CONNECTORS,
    DataConnectorName,
)
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo

mmlu_pro_connector = DATA_CONNECTORS.get("mmlu-pro")()
mmlu_pro_cached = {
    e.question_id: (e, e.model_dump_json()) for e in mmlu_pro_connector.iterate_data()
}

mmlu_pro_reversed_cache = {
    mmlu_pro_connector.prepare_example(mmlu_pro_cached[e.question_id][0]).question: e
    for e in mmlu_pro_connector.iterate_data()
}


logger = logging.getLogger(__name__)


class Question(PolarsBaseModel):
    question_id: int
    original_question_id: int

    data_connector: DataConnectorName
    original_source: str
    category: str
    question: str

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
        span_attributes: dict[str, Any] | None,
    ) -> Question:
        match hydra_config.experiment.data.value:
            case (
                "mmlu-pro"
                | "mmlu-pro-subset"
                | "mmlu-pro-big-subset"
                | "mmlu-pro-medium-subset"
            ):
                if span_attributes is not None:
                    question = MMLUProExample.model_validate(
                        json.loads(span_attributes["question"])
                    )
                else:
                    logger.warning(
                        f"Did not find phoenix data for span: {entry.phoenix_span_info.span_id_hex}"
                    )
                    question = mmlu_pro_reversed_cache[entry.input.question]

                if (
                    mmlu_pro_connector.prepare_example(
                        mmlu_pro_cached[question.question_id][0]
                    ).question
                    != entry.input.question
                ):
                    c = mmlu_pro_connector.prepare_example(
                        mmlu_pro_cached[question.question_id][0]
                    ).question
                    print(f"Entry: {entry.input.question}")
                    print(f"Connector: {c}")
                    print(meta_info)
                    raise RuntimeError(
                        f"Invalid Question Mapping Detected for span {entry.phoenix_span_info.span_id_hex}."
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
            case "gpqa-diamond":
                question = GPQADiamondExample.model_validate(
                    json.loads(span_attributes["question"])
                )
                return cls(
                    question_id=assigned_id,
                    original_question_id=question.question_id,
                    data_connector=hydra_config.experiment.data,
                    original_source="",
                    category="",
                    question=entry.input.question,
                    answer_options=["A", "B", "C", "D"],
                    answer_index=-1,
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
