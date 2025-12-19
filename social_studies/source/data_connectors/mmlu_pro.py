from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Callable, Any

import datasets
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from tqdm import tqdm

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


class ExperimentInput(BaseModel):
    example_questions: str
    question: str


class HistoryMessage(BaseModel):
    input_context: list[BaseMessage]
    answer: str
    agent_id: int
    model_name: str
    options: dict[str, Any]


class ExperimentOutput(BaseModel):
    number_of_agents: int
    used_input_tokens: int
    used_output_tokens: int
    used_rounds: int
    answers_at_beginning: list[str] | None = None
    answers_at_end: list[str] | None = None
    final_answer: str
    history: list[HistoryMessage]


class ExperimentQuestion(BaseModel):
    output: ExperimentOutput

    question_id: int
    question: str
    src: str
    category: MMLUProCategory
    cot_content: str
    answer_index: int
    answer: str
    options: list[str]


def run_test_set(
    run_one_question: Callable[[ExperimentInput], ExperimentOutput],
    experiment_name: str,
    num_workers: int = 8,
    chunksize: int = 1,
):
    dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    prompts = get_example_questions(dataset["validation"])
    test_ds = dataset["test"]

    def process_entry(entry):
        query = "Q: " + entry["question"] + "\n" + form_options(entry["options"]) + "\n"

        output = run_one_question(
            ExperimentInput(
                example_questions=prompts[MMLUProCategory(entry["category"])],
                question=query,
            )
        )

        entry = dict(entry)
        entry["category"] = MMLUProCategory(entry["category"])
        return ExperimentQuestion(**entry, output=output)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with open(f"{experiment_name}.json", "w") as output_file:
            for processed_entry in tqdm(
                executor.map(process_entry, test_ds, chunksize=chunksize),
                total=len(test_ds),
            ):
                json_string = processed_entry.model_dump_json(indent=None)
                output_file.write(json_string + "\n")
