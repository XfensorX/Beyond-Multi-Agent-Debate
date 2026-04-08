from enum import Enum
from typing import Any, Iterable

import datasets
from pydantic import ConfigDict

from social_groups.trialrunner.data_connectors.base import DataConnector, ExampleBase
from social_groups.trialrunner.decision_schemes.base import ExampleInput
from social_groups.trialrunner.experiment.main_registry import register_data_connector

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class MMLUProCategory(Enum):
    COMPUTER_SCIENCE = "computer science"
    MATH = "math"
    CHEMISTRY = "chemistry"
    ENGINEERING = "engineering"
    LAW = "law"
    BIOLOGY = "biology"
    HEALTH = "health"
    PHYSICS = "physics"
    BUSINESS = "business"
    PHILOSOPHY = "philosophy"
    ECONOMICS = "economics"
    OTHER = "other"
    PSYCHOLOGY = "psychology"
    HISTORY = "history"


def form_options(options: list):
    option_str = "Options are:\n"
    for opt, o in zip(options, OPTION_LETTERS):
        option_str += f"({o}): {opt}" + "\n"
    return option_str


def get_example_questions(validation_data) -> dict[MMLUProCategory, str]:
    prompts = {c: "" for c in MMLUProCategory}

    for d in validation_data:
        prompts[MMLUProCategory(d["category"])] += (
            "Q:"
            + " "
            + d["question"]
            + "\n"
            + form_options(d["options"])
            + "\n"
            + d["cot_content"]
            + "\n\n"
        )
    return prompts


class MMLUProExample(ExampleBase):
    model_config = ConfigDict(frozen=True)

    question: str
    src: str
    category: MMLUProCategory
    cot_content: str
    answer_index: int
    answer: str
    options: list[str]


@register_data_connector("mmlu-pro")
class MMLUProConnector(DataConnector[MMLUProExample]):
    prompts = None

    def __init__(self):
        self.dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro")

    def prepare_example(self, example: MMLUProExample) -> ExampleInput:
        query = "Q: " + example.question + "\n" + form_options(example.options) + "\n"
        return ExampleInput(
            presolved_questions=self.prompts[example.category],
            question=query,
        )

    def iterate_data(self) -> Iterable[MMLUProExample]:
        self.prompts = get_example_questions(self.dataset["validation"])
        test_ds = self.dataset["test"]

        def to_structured(entry: dict[str, Any]):
            entry = dict(entry)
            entry["category"] = MMLUProCategory(entry["category"])
            return MMLUProExample(**entry)

        return iter(map(to_structured, test_ds))

    def data_length(self):
        return len(self.dataset["test"])
