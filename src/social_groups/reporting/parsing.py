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


_ANSWER_OPTIONS: dict[AnswerOptions, set[str]] = {
    AnswerOptions.letters_A_to_J: {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}
}


_ANSWER_PATTERNS: dict[AnswerOptions, set[str]] = {
    AnswerOptions.letters_A_to_J: {
        r"answer\s+is\s*:?\s*\(?([A-J])\)?",  # ...answer is (C)... | ...answer is C ...
        r"^\s*([A-J])\s*$",  # single letter: C
        r"^\s*\(?\s*([A-J])\s*\)?\s*$",  # single letter in braces: (C)
        r"answer\s*:\s*\(?\s*([A-J])\s*\)?\s*[\.\!\?]*\s*$",
        # ... Answer: (C) | ... Answer: C  |# at the end of the string with optional punctuation
        r"^\s*\(\s*([A-J])\s*\)\s*:?.*$",  # (C): ... | (C) .... |# at the beginning of the string
    }
}


def get_string_parser(
    patterns_to_check: set[str], possible_answers: set[str]
) -> Callable[[str | None], str | ParsingResultError]:
    @lru_cache
    def parse(answer: str | None):
        if answer is None:
            return ParsingResultError.EMPTY_INPUT.value

        answer = answer.replace("*", "")
        for pat in patterns_to_check:
            m = re.search(pat, answer, flags=re.IGNORECASE)
            if m and ((final_letter := m.group(1).upper()) in possible_answers):
                return final_letter

        return ParsingResultError.NOT_PARSABLE.value

    return parse


@dataclasses.dataclass
class AnswerParser:
    """
    Parses answers from a given string. (in polars)

    Used to extract multiple choice answers from LLM return expressions.

    """

    option: AnswerOptions

    def __call__(self, expr: pl.Expr) -> pl.Expr:
        match self.option:
            case AnswerOptions.letters_A_to_J:
                parser = get_string_parser(
                    _ANSWER_PATTERNS[self.option], _ANSWER_OPTIONS[self.option]
                )
                return expr.map_elements(parser, return_dtype=pl.Utf8)

            case _:
                raise NotImplementedError()


@dataclasses.dataclass
class AnswerComparer:
    option: AnswerOptions

    triple_underscore_handling: Literal["null", "wrong", "random"]

    @staticmethod
    def __post_init__():
        random.seed(0)

    def __call__(self, given: pl.Expr, target: pl.Expr) -> pl.Expr:
        match self.option:
            case AnswerOptions.letters_A_to_J:
                match self.triple_underscore_handling:
                    case "null":
                        return (
                            pl.when(given.str.starts_with("___"))
                            .then(None)
                            .otherwise(given == target)
                        )
                    case "wrong":
                        return (
                            pl.when(given.str.starts_with("___"))
                            .then(False)
                            .otherwise(given == target)
                        )
                    case "random":
                        p_correct = 1 / len(_ANSWER_OPTIONS[self.option])
                        return (
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
