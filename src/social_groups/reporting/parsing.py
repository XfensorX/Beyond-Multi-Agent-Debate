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
        r"answer\s*:\s*\(?\s*([A-J])\s*\)?\s*[\.\!\?]*\s*$",
        # ... Answer: (C) | ... Answer: C  |# at the end of the string with optional punctuation
        r"\s*\(\s*([A-J])\s*\)\s*$",  # ... (C) at the end of the string
        r"^\s*\(?\s*([A-J])\s*\)?\s*$",  # single letter in braces: (C)
        r"^\s*([A-J])\s*$",  # single letter: C
        r"answer\s*:\s*\(?\s*([A-J])\s*\)?\s*[\.\!\?]*\s*",  # ... Answer: (C) | ... Answer: C  | # Somewhere in the string
        r"^\s*\(\s*([A-J])\s*\)\s?\:?\s*",
        r"^([A-J])\:",
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
            last_match = None
            for m in re.finditer(pat, answer, flags=re.IGNORECASE):
                if not last_match or m.start() > last_match.start():
                    last_match = m

            if last_match and (
                (final_letter := last_match.group(1).upper()) in possible_answers
            ):
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

    fill_nulls_with: bool | None = None

    @staticmethod
    def __post_init__():
        random.seed(0)

    def __call__(self, given: pl.Expr, target: pl.Expr) -> pl.Expr:
        match self.option:
            case AnswerOptions.letters_A_to_J:
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
