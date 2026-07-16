from __future__ import annotations

import dataclasses
import enum
import random
import re
from functools import lru_cache
from typing import Callable, Literal

import polars as pl


class ParsingResultError(enum.Enum):
    EMPTY_INPUT = "___empty_input___"
    NOT_PARSABLE = "___not_parsable___"


class AnswerOptions(enum.Enum):
    letters_A_to_J = "letters_A_to_J"
    letters_A_to_D = "letters_A_to_D"
    letters_A_to_J_naive = "letters_A_to_J_naive"


_ANSWER_OPTIONS: dict[AnswerOptions, set[str]] = {
    AnswerOptions.letters_A_to_J: {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
    AnswerOptions.letters_A_to_D: {"A", "B", "C", "D"},
    AnswerOptions.letters_A_to_J_naive: {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
    },
}


_ANSWER_PATTERNS: dict[AnswerOptions, list[str]] = {
    AnswerOptions.letters_A_to_J: [
        r"answer\s+is\s*:?\s*\(?([A-J])\)?",  # ...answer is (C)... | ...answer is C ...
        r"answer\s*:\s*\(?\s*([A-J])\s*\)?\s*[\.\!\?]*\s*$",
        # ... Answer: (C) | ... Answer: C  |# at the end of the string with optional punctuation
        r"\s*\(\s*([A-J])\s*\)\s*$",  # ... (C) at the end of the string
        r"^\s*\(?\s*([A-J])\s*\)?\s*$",  # single letter in braces: (C) at the end
        r"^\s*([A-J])\s*$",  # single letter: C at the end
        r"answer\s*:\s*\(?\s*([A-J])\s*\)?\s*[\.\!\?]*\s*",  # ... Answer: (C) | ... Answer: C  | # Somewhere in the string
        r"^\s*\(\s*([A-J])\s*\)\s?\:?\s*",
        r"^([A-J])\:",
        r"^\(([A-J])\)$",
        r"\(([A-J])\)\s*(?:is the correct answer|is the answer|should be the correct answer|matches|must be the correct answer|is correct)",
    ],
    AnswerOptions.letters_A_to_D: [
        r"answer\s+is\s*:?\s*\(?([A-D])\)?",  # ...answer is (C)... | ...answer is C ...
        r"answer\s*:\s*\(?\s*([A-D])\s*\)?\s*[\.\!\?]*\s*$",
        # ... Answer: (C) | ... Answer: C  |# at the end of the string with optional punctuation
        r"\s*\(\s*([A-D])\s*\)\s*$",  # ... (C) at the end of the string
        r"^\s*\(?\s*([A-D])\s*\)?\s*$",  # single letter in braces: (C) at the end
        r"^\s*([A-D])\s*$",  # single letter: C at the end
        r"answer\s*:\s*\(?\s*([A-D])\s*\)?\s*[\.\!\?]*\s*",  # ... Answer: (C) | ... Answer: C  | # Somewhere in the string
        r"^\s*\(\s*([A-D])\s*\)\s?\:?\s*",
        r"^([A-D])\:",
        r"^\(([A-D])\)$",
        r"\(([A-D])\)\s*(?:is the correct answer|is the answer|should be the correct answer|matches|must be the correct answer|is correct)",
    ],
    AnswerOptions.letters_A_to_J_naive: [
        r"The answer is \(([A-J])\).?\s*$",
        r"^\(([A-J])\)$",
    ],
}


def get_string_parser(
    patterns_to_check: list[str], possible_answers: set[str]
) -> Callable[[str | None], str | ParsingResultError]:
    @lru_cache
    def parse(answer: str | None):
        if answer is None:
            return ParsingResultError.EMPTY_INPUT.value

        answer = answer.replace("*", "")

        matches: list[tuple[int, str]] = []

        for pat in patterns_to_check:
            for m in re.finditer(pat, answer, flags=re.IGNORECASE):
                final_letter = m.group(1).upper()
                if final_letter in possible_answers:
                    matches.append((m.start(), final_letter))

        if not matches:
            return ParsingResultError.NOT_PARSABLE.value

        return max(matches, key=lambda x: (x[0], len(x[1]), x[1]))[1]  # take last match

    return parse


@dataclasses.dataclass
class AnswerParser:
    """
    Parses answers from a given string. (in polars)
    Used to extract multiple choice answers from LLM return expressions.
    """

    option: AnswerOptions

    def __post_init__(self):
        match self.option:
            case (
                AnswerOptions.letters_A_to_J
                | AnswerOptions.letters_A_to_J_naive
                | AnswerOptions.letters_A_to_D
            ):
                self._parser = get_string_parser(
                    _ANSWER_PATTERNS[self.option], _ANSWER_OPTIONS[self.option]
                )
            case _:
                raise NotImplementedError()

    def __call__(self, expr: pl.Expr) -> pl.Expr:
        return expr.map_elements(self._parser, return_dtype=pl.Utf8)


@dataclasses.dataclass
class AnswerComparer:
    option: AnswerOptions

    triple_underscore_handling: Literal["null", "wrong", "random"]

    fill_nulls_with: bool | None = None

    def __call__(self, given: pl.Expr, target: pl.Expr) -> pl.Expr:
        match self.option:
            case AnswerOptions.letters_A_to_J | AnswerOptions.letters_A_to_D:
                match self.triple_underscore_handling:
                    case "null":
                        t = (
                            pl.when(given.str.starts_with("___"))
                            .then(None)
                            .otherwise(given == target)
                        )
                    case "wrong":
                        t = (
                            pl.when(given.str.starts_with("___"))
                            .then(False)
                            .otherwise(given == target)
                            .fill_null(False)
                        )
                    case "random":
                        random.seed(0)
                        p_correct = 1 / len(_ANSWER_OPTIONS[self.option])
                        t = (
                            pl.when(given.str.starts_with("___"))
                            .then(
                                given.map_elements(
                                    lambda _: random.random() < p_correct
                                )
                            )
                            .otherwise(given == target)
                        )

                    case _:
                        raise NotImplementedError()

            case _:
                raise NotImplementedError()

        if self.fill_nulls_with is not None:
            return t.fill_null(self.fill_nulls_with)
        else:
            return t
