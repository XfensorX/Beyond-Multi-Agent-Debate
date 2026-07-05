from enum import Enum
from typing import Any, Iterable

import datasets
from pydantic import ConfigDict

from social_groups.trialrunner.data_connectors.base import DataConnector, ExampleBase
from social_groups.trialrunner.decision_schemes.base import ExampleInput
from social_groups.trialrunner.experiment.main_registry import register_data_connector


class GPQADiamondExample(ExampleBase):
    model_config = ConfigDict(frozen=True)
    question: str
    answer: str


@register_data_connector("gpqa-diamond")
class GPQADiamondConnector(DataConnector[GPQADiamondExample]):
    prompts = None

    def __init__(self):
        self.dataset = datasets.load_dataset("fingertap/GPQA-Diamond")

    def prepare_example(self, example: GPQADiamondExample) -> ExampleInput:
        return ExampleInput(presolved_questions="", question=example.question)

    def iterate_data(self) -> Iterable[GPQADiamondExample]:
        test_ds = self.dataset["test"]

        def to_structured(id_entry: tuple[int, dict[str, Any]]):
            id_, entry = id_entry
            return GPQADiamondExample(
                question_id=id_, question=entry["question"], answer=entry["answer"]
            )

        return iter(map(to_structured, enumerate(test_ds)))

    def data_length(self):
        return len(self.dataset["test"])