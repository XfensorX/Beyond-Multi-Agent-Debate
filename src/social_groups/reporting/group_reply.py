import abc
import collections
import dataclasses
import enum

import polars as pl
from cachetools.func import lru_cache
from more_itertools.recipes import all_equal

from social_groups.reporting.parsing import ParsingResultError


class GroupReplyError(enum.Enum):
    ALL_VOTES_INVALID = "___all_votes_invalid___"
    DIFFERENT_VOTES = "___different_votes___"


@lru_cache
def is_invalid_response(response: str) -> bool:
    return response in {x.value for x in ParsingResultError}


class GroupReplyStrategy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, votes: list[str]): ...


class SingularityVote(GroupReplyStrategy):
    """
    all have to vote the same, otherwise invalid, ignores invalid results
    """

    def __call__(self, votes: list[str]) -> str:
        valid_votes = [v for v in votes if not is_invalid_response(v)]

        if not valid_votes:
            return GroupReplyError.ALL_VOTES_INVALID.value

        if all_equal(valid_votes):
            return votes[0]

        return GroupReplyError.DIFFERENT_VOTES.value


class MajorityVote(GroupReplyStrategy):
    """the majority vote >50%"""

    def __call__(self, votes: list[str]) -> str:
        valid_votes = [v for v in votes if not is_invalid_response(v)]

        if not valid_votes:
            return GroupReplyError.ALL_VOTES_INVALID.value

        m = collections.Counter(valid_votes).most_common(2)
        if len(m) == 1:
            return m[0][0]

        most_common, second_most_common = m

        if most_common[1] == second_most_common[1]:
            return GroupReplyError.DIFFERENT_VOTES.value
        else:
            return most_common[0]


@dataclasses.dataclass
class GroupReplyAggregator:
    strategy: GroupReplyStrategy

    def __call__(self, expr: pl.Expr) -> pl.Expr:
        return expr.map_elements(self.strategy)
