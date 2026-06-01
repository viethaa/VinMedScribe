"""Basic import tests for the scaffold."""

from __future__ import annotations

from medscribe.evaluation.metrics import precision_recall_f1
from medscribe.pipeline import run_text_pipeline


def test_text_pipeline_returns_note() -> None:
    result = run_text_pipeline("Benh nhan dau dau hai ngay.")

    assert result.cleaned_transcript
    assert "# SOAP Note" in result.note


def test_precision_recall_f1() -> None:
    metrics = precision_recall_f1(true_positive=2, false_positive=1, false_negative=1)

    assert round(metrics["precision"], 2) == 0.67
    assert round(metrics["recall"], 2) == 0.67
    assert round(metrics["f1"], 2) == 0.67
