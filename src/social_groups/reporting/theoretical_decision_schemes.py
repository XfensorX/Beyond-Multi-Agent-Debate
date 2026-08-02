import numpy as np


def proportionality(N):
    """
    Probability equals proportion of correct members.
    """
    return np.array([i / N for i in range(N + 1)])


def equiprobability(N):
    """
    always the same probability
    """
    return np.array([0.5 for i in range(N + 1)])


def majority_rule(N):
    """
    Majority rule with tie = 0.5
    """
    scheme = np.zeros(N + 1)
    for i in range(N + 1):
        if i > N / 2:
            scheme[i] = 1.0
        elif i == N / 2:
            scheme[i] = 0.5
        else:
            scheme[i] = 0.0
    return scheme


def truth_wins(N):
    """
    At least one correct member guarantees correct group decision.
    """
    scheme = np.zeros(N + 1)
    scheme[1:] = 1.0
    return scheme


def truth_supported(N):
    """
    At least two correct members required.
    """
    scheme = np.zeros(N + 1)
    if N >= 2:
        scheme[2:] = 1.0
    return scheme


def unanimity(N):
    """
    All members must be correct.
    """
    scheme = np.zeros(N + 1)
    scheme[N] = 1.0
    return scheme


def random_rule(N):
    """
    Random responding (baseline).
    """
    return np.full(N + 1, 0.5)


def incorrect_wins(N):
    """
    Mirror of truth-wins: any incorrect member leads to incorrect decision.
    """
    scheme = np.ones(N + 1)
    scheme[1:] = 0.0
    return scheme


def minority_wins(N):
    """
    Mirror-type rule: group tends toward minority (contrast model).
    """
    scheme = np.ones(N + 1)
    scheme[N] = 0.0
    return scheme


THEORETICAL_DECISION_SCHEMES = {
    "proportionality": proportionality,
    "majority_rule": majority_rule,
    "truth_wins": truth_wins,
    "truth_supported": truth_supported,
    "unanimity": unanimity,
    "random_rule": random_rule,
    "incorrect_wins": incorrect_wins,
    "minority_wins": minority_wins,
}
