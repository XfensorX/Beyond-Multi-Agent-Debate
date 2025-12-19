import random
import re

from tqdm import tqdm

from social_studies.source.result_analysis.MultipleChoiceAnalyzer import (
    MultipleChoiceTracker,
)

from social_studies.source.data_connectors.mmlu_pro import (
    OPTION_LETTERS,
    ExperimentQuestion,
    MMLUProCategory,
)

_ANSWER_PATTERNS = [
    r"answer\s+is\s*\(?([A-J])\)?",  # ...answer is (C)... | ...answer is C ...
    r"^\s*([A-J])\s*$",  # single letter: C
    r"^\s*\(?\s*([A-J])\s*\)?\s*$",  # single letter in braces: (C)
]


def parse_prediction(output: str) -> str:
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, output, flags=re.IGNORECASE)
        if m and ((answer := m.group(1).upper()) in OPTION_LETTERS):
            return answer

    print(f"extraction failed {output=}, do a random guess:")
    return random.choice(OPTION_LETTERS)


# TODO:
# - make this take an iterator instead of a file
# - instead of depending on ExperimentQuestion it should have it's own datatype


def analyse_results(file):
    with open(file, "r") as input_file:
        lines = input_file.readlines()

    per_category_tracker = {c: MultipleChoiceTracker() for c in MMLUProCategory}
    total_tracker = MultipleChoiceTracker()

    for entry in tqdm(lines):
        entry = ExperimentQuestion.model_validate_json(entry)
        prediction = parse_prediction(entry.output.final_answer)

        total_tracker.register(entry.answer, prediction)
        per_category_tracker[entry.category].register(entry.answer, prediction)

    return {
        "total": total_tracker,
        "categories": per_category_tracker,
    }
