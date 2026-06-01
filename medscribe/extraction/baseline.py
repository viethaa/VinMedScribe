"""Baseline clinical field extraction.

This starts as a conservative scaffold. The real extraction contribution can
replace this module with rules, local LLM prompting, or PhoBERT fine-tuning.
"""

from __future__ import annotations

from medscribe.extraction.schema import ClinicalFields


def extract_clinical_fields(transcript: str) -> ClinicalFields:
    """Return an empty structured note shell with transcript evidence attached."""
    evidence = [transcript] if transcript else []
    return ClinicalFields(raw_evidence=evidence)
