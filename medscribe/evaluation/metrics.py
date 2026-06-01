"""Simple evaluation metric placeholders."""

from __future__ import annotations


def precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from counts."""
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
