from math import isnan
from typing import Callable, Literal

import numpy as np
import polars as pl
from scipy.stats import binom


def log_likelihood_of_decision_scheme(
    obs: pl.DataFrame, scheme: Callable[[int], np.ndarray], epsilon: float = 1e-12
) -> float:
    """
    Log-likelihood of observed group decisions under an SDS model.

    Parameters
    ----------
    obs : DataFrame with len N+1
        Correct Members Beginning: int
        correct: #cases where (group correct | correct Members at beginning)
        incorrect: # cases where (group incorrect | correct Members at beginning)

    scheme : Callable
        function taking N and returning array of shape (N+1,)
        scheme(N)[i] = P(group correct | i correct members)

    epsilon : float
        Small value to clip probabilities to avoid log(0).
    """
    assert obs["Correct Members Beginning"].is_sorted()
    assert obs["Correct Members Beginning"].min() == 0
    assert (
        obs["Correct Members Beginning"].len()
        == obs["Correct Members Beginning"].max() + 1
    )

    obs = (
        obs.drop("Correct Members Beginning").select("incorrect", "correct").to_numpy()
    )

    N = obs.shape[0] - 1
    p_correct = scheme(N)
    p_adj = np.clip(p_correct, epsilon, 1 - epsilon)
    k = obs[:, 1]
    n = obs.sum(axis=1)
    # Calculate log-likelihood for each row (i) and sum them
    # binom.logpmf(k, n, p) calculates: log( (n choose k) * p^k * (1-p)^(n-k) )
    row_log_likelihoods = binom.logpmf(k, n, p_adj).sum()

    return float(row_log_likelihoods)


def wasserstein_transition_distance_to_optimal(
    decision_scheme_matrix: pl.DataFrame,
    biased: Literal["good proposals", "bad proposals", "weight on one", False] = False,
) -> float:
    assert (
        decision_scheme_matrix.select(pl.exclude("_"))
        .select(pl.all_horizontal(pl.all() <= 1).all())
        .item()
    ), decision_scheme_matrix

    assert (
        decision_scheme_matrix.select("_")
        .to_series()
        .str.replace("to ", "")
        .cast(int)
        .is_sorted(descending=True)
    )

    assert (
        pl.Series(decision_scheme_matrix.select(pl.exclude("_")).columns)
        .str.replace("from ", "")
        .cast(int)
        .is_sorted(descending=True)
    )

    decision_scheme_matrix = decision_scheme_matrix.drop("_").to_numpy()
    rows, cols = decision_scheme_matrix.shape

    match biased:
        case False:
            weights = np.ones(cols) / cols
        case "good proposals":
            weights = np.arange(1, cols + 1)[::-1] / ((cols * cols + cols) / 2)
        case "bad proposals":
            weights = np.arange(1, cols + 1) / ((cols * cols + cols) / 2)
        case "weight on one":
            one_weight = 0.8
            weights = np.full(cols, (1 - one_weight) / (cols - 1))
            weights[-2] = one_weight
        case _:
            raise NotImplementedError

    distance = float(
        weights @ np.abs(1 - np.cumsum(decision_scheme_matrix, axis=0)).sum(axis=0)
    )

    distance /= cols

    if isnan(distance):
        raise RuntimeError(
            "Some former calculation has an error. This should not happen."
        )

    return distance
