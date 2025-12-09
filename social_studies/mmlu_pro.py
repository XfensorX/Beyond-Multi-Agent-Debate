# Adapted from https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/run_gpt4o.py


import json
import random
import re
from dataclasses import dataclass
from typing import Callable

import datasets
from enum import Enum

from tqdm import tqdm

from social_studies.utils import AccuracyTracker


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


def parse_prediction(output):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    else:
        print(f"extraction failed {output=}, do a random guess:")
        return random.choice(OPTION_LETTERS)


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


@dataclass
class ExperimentInput:
    example_questions: str
    question: str


def run_test_set(
    run_one_question: Callable[[ExperimentInput], str], experiment_name: str
):
    dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    prompts = get_example_questions(dataset["validation"])

    per_category_accuracy = {c: AccuracyTracker() for c in MMLUProCategory}
    total_accuracy = AccuracyTracker()
    answers = []

    print("----------------- Start Answering -------------------")
    with open(f"{experiment_name}.json", "w") as output_file:
        for entry in tqdm(dataset["test"]):
            query = (
                "Q: " + entry["question"] + "\n" + form_options(entry["options"]) + "\n"
            )
            answer = run_one_question(
                ExperimentInput(
                    example_questions=prompts[MMLUProCategory(entry["category"])],
                    question=query,
                )
            )

            entry["solution"] = answer
            answers.append(entry)
            prediction = parse_prediction(answer)
            if entry["answer"] == prediction:
                total_accuracy.register_success()
                per_category_accuracy[entry["category"]].register_success()
            else:
                total_accuracy.register_failed()
                per_category_accuracy[entry["category"]].register_failed()

            json_string = json.dumps(entry)
            output_file.write(json_string + "\n")
