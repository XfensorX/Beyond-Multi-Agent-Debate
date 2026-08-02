import math
from collections import defaultdict
from typing import Callable

import polars as pl


def entropy(xs: list[str]) -> float:
    n = len(xs)

    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1

    probs = [c / n for c in counts.values()]

    return -sum(p * math.log2(p) for p in probs)


def normalized_entropy(K: int) -> Callable[[list[str]], float]:
    def normalize(xs: list[str]):
        H = entropy(xs)
        return H / math.log2(K)

    return normalize


def list_entropy(expr: pl.Expr):
    return expr.map_elements(entropy, return_dtype=pl.Float64)


def list_normalized_entropy(expr: pl.Expr, K: int):

    return expr.map_elements(normalized_entropy(K), return_dtype=pl.Float64)
