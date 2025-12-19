from collections import Counter
from typing import Hashable, Iterable


class MultipleChoiceTracker:
    """
    Tracker for multiple-choice questions where:
    - each question has exactly one correct answer
    - you may give a real answer, or a special "no clue" answer

    You call `register(correct, given)` for each question.

    Parameters
    ----------
    none_label : hashable, default None
        The value used to represent "no clue" / "no answer".
    """

    def __init__(self, none_label: Hashable = None):
        self.none_label = none_label
        self._confusion = Counter()  # (true, pred) -> count
        self._total = 0

    # ---------------------------------------------------------------------
    # Registration
    # ---------------------------------------------------------------------
    def register(self, correct_answer: Hashable, given_answer: Hashable):
        """
        Register a single question result.

        Use `given_answer == self.none_label` to indicate "no clue".
        """
        self._confusion[(correct_answer, given_answer)] += 1
        self._total += 1

    def register_many(
        self, correct_answers: Iterable[Hashable], given_answers: Iterable[Hashable]
    ):
        """Convenience: add a batch of results."""
        for c, g in zip(correct_answers, given_answers):
            self.register(c, g)

    # ---------------------------------------------------------------------
    # Basic counts
    # ---------------------------------------------------------------------
    @property
    def total_all(self) -> int:
        """Total number of questions (including 'no clue')."""
        return self._total

    @property
    def total_no_answer(self) -> int:
        """Number of questions where the given answer was none_label."""
        if self.none_label is None and self.none_label not in [
            k[1] for k in self._confusion
        ]:
            # early exit when none_label is not used at all
            return 0
        return sum(
            count
            for (true, pred), count in self._confusion.items()
            if pred == self.none_label
        )

    @property
    def total_answered(self) -> int:
        """Number of questions where an actual answer was given."""
        return self.total_all - self.total_no_answer

    @property
    def labels(self):
        """
        All labels seen as correct or predicted, excluding the none_label.
        """
        s = set()
        for true, pred in self._confusion.keys():
            if true != self.none_label:
                s.add(true)
            if pred != self.none_label:
                s.add(pred)
        return tuple(sorted(s))

    # ---------------------------------------------------------------------
    # Internal helpers (answered-only confusion)
    # ---------------------------------------------------------------------
    def _answered_confusion_items(self):
        """Yield (true, pred, count) for entries where pred != none_label."""
        for (true, pred), count in self._confusion.items():
            if pred == self.none_label:
                continue
            yield true, pred, count

    def _tp_fp_fn_answered(self, label: Hashable):
        """
        Compute TP, FP, FN for a label using only answered questions
        (ignoring none_label predictions).
        """
        tp = fp = fn = 0
        for true, pred, count in self._answered_confusion_items():
            if true == label and pred == label:
                tp += count
            elif true != label and pred == label:
                fp += count
            elif true == label and pred != label:
                fn += count
        return tp, fp, fn

    # ---------------------------------------------------------------------
    # Accuracy and micro metrics
    # ---------------------------------------------------------------------
    @property
    def accuracy_answered(self) -> float:
        """
        Accuracy using only questions where an actual answer was given.
        """
        if self.total_answered == 0:
            return 0.0
        correct_answered = sum(
            count
            for (true, pred), count in self._confusion.items()
            if pred != self.none_label and true == pred
        )
        return correct_answered / self.total_answered

    @property
    def accuracy_all(self) -> float:
        """
        Accuracy on ALL questions, where "no clue" is treated as if a
        random answer had been given.

        Expected correct from 'no clue' = total_no_answer * (1 / len(self.labels)).
        """
        if self.total_all == 0:
            return 0.0

        correct_answered = sum(
            count
            for (true, pred), count in self._confusion.items()
            if pred != self.none_label and true == pred
        )

        print(f"Assuming there are {len(self.labels)} possible answers: {self.labels}")
        expected_correct_from_none = self.total_no_answer * (1.0 / len(self.labels))

        return (correct_answered + expected_correct_from_none) / self.total_all

    # In single-label multi-class:
    #   micro_precision = micro_recall = micro_f1 = accuracy
    @property
    def micro_precision_answered(self) -> float:
        return self.accuracy_answered

    @property
    def micro_recall_answered(self) -> float:
        return self.accuracy_answered

    @property
    def micro_f1_answered(self) -> float:
        return self.accuracy_answered

    @property
    def micro_precision_all(self) -> float:
        return self.accuracy_all

    @property
    def micro_recall_all(self) -> float:
        return self.accuracy_all

    @property
    def micro_f1_all(self) -> float:
        return self.accuracy_all

    # ---------------------------------------------------------------------
    # Per-label metrics (answered-only)
    # ---------------------------------------------------------------------
    def precision_answered(self, label: Hashable) -> float:
        tp, fp, _ = self._tp_fp_fn_answered(label)
        denom = tp + fp
        return tp / denom if denom else 0.0

    def recall_answered(self, label: Hashable) -> float:
        tp, _, fn = self._tp_fp_fn_answered(label)
        denom = tp + fn
        return tp / denom if denom else 0.0

    def f1_answered(self, label: Hashable) -> float:
        p = self.precision(label)
        r = self.recall(label)
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def macro_precision_answered(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.precision(l) for l in self.labels) / len(self.labels)

    @property
    def macro_recall_answered(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.recall(l) for l in self.labels) / len(self.labels)

    @property
    def macro_f1_answered(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.f1(l) for l in self.labels) / len(self.labels)

    # ---------------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------------
    def per_label_summary_answered(self):
        """Per-label stats (TP, FP, FN, precision, recall, F1) for answered-only."""
        summary = {}
        for l in self.labels:
            tp, fp, fn = self._tp_fp_fn_answered(l)
            summary[l] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": self.precision(l),
                "recall": self.recall(l),
                "f1": self.f1(l),
            }
        return summary

    def confusion_dict(self):
        """
        Return the raw confusion info as a nested dict (including none_label):
        result[true][pred] = count
        """
        result = {}
        for (true, pred), count in self._confusion.items():
            result.setdefault(true, {})[pred] = count
        return result

    # ---------------------------------------------------------------------
    # Representation
    # ---------------------------------------------------------------------
    def __repr__(self):
        return (
            f"MultipleChoiceTracker("
            f"total_all={self.total_all}, "
            f"answered={self.total_answered}, "
            f"no_answer={self.total_no_answer}, "
            f"acc_answered={self.accuracy_answered:.3f}, "
            f"acc_all_random={self.accuracy_all:.3f})"
        )
