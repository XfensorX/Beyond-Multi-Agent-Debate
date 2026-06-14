from __future__ import annotations

from typing import Any, List, Optional

from social_groups.analyzer.config import ATTRIBUTE_KEY_SPAN_URL
from social_groups.analyzer.models.base import PolarsBaseModel
from social_groups.general.tracking import TrackEntry
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo

MessageId = int


class Answer(PolarsBaseModel):
    id: int
    run_id: int
    message_ids: List[MessageId]
    question_id: int
    phoenix_span_id: str
    phoenix_span_url: str
    run_identifier: str
    answers_at_beginning: List[str]
    answers_at_end: List[str]
    final_answer: Optional[str]
    used_input_tokens: int
    used_output_tokens: int

    @classmethod
    def from_raw_data(
        cls,
        *,
        assigned_id: int,
        run_id: int,
        entry: TrackEntry,
        hydra_config: MainConfig,
        meta_info: ExperimentMetaInfo,
        question_id: int,
        span_attributes: dict[str, Any] | None,
    ) -> Answer:

        return cls(
            id=assigned_id,
            run_id=run_id,
            message_ids=[],  # TODO
            question_id=question_id,
            phoenix_span_id=entry.phoenix_span_info.span_id_hex or "NOT-AVAILABLE",
            run_identifier=meta_info.phoenix_project_name,
            answers_at_beginning=(
                (entry.output.answers_at_beginning or []) if entry.output else []
            ),
            answers_at_end=(
                (entry.output.answers_at_end or []) if entry.output else []
            ),
            final_answer=entry.output.final_answer if entry.output else None,
            phoenix_span_url=span_attributes[ATTRIBUTE_KEY_SPAN_URL]
            if span_attributes
            else "unavailable",
            used_input_tokens=entry.output.used_input_tokens,
            used_output_tokens=entry.output.used_output_tokens,
        )
