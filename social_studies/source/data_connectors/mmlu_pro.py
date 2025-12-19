from concurrent.futures import ThreadPoolExecutor
from enum import Enum


import datasets
from pydantic import BaseModel
from tqdm import tqdm

from decision_schemes.base import DecisionScheme, ExampleInput, ExampleOutput

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


class ExperimentQuestion(BaseModel):
    output: ExampleOutput

    question_id: int
    question: str
    src: str
    category: MMLUProCategory
    cot_content: str
    answer_index: int
    answer: str
    options: list[str]


def run_test_set(
    scheme: DecisionScheme,
    experiment_name: str,
    num_workers: int = 8,
):
    dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    prompts = get_example_questions(dataset["validation"])
    test_ds = dataset["test"]

    def process_entry(entry):
        query = "Q: " + entry["question"] + "\n" + form_options(entry["options"]) + "\n"

        output = scheme.run_example(
            ExampleInput(
                example_questions=prompts[MMLUProCategory(entry["category"])],
                question=query,
            )
        )

        entry = dict(entry)
        entry["category"] = MMLUProCategory(entry["category"])
        return ExperimentQuestion(**entry, output=output)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # TODO: move experiment tracking to different class
        with open(f"{experiment_name}.json", "w") as output_file:
            for processed_entry in tqdm(
                executor.map(process_entry, test_ds),
                total=len(test_ds),
            ):
                json_string = processed_entry.model_dump_json(indent=None)
                output_file.write(json_string + "\n")
