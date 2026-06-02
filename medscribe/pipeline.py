"""High-level orchestration for the text-first MedScribe pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from medscribe.cleaning.normalizer import clean_transcript
from medscribe.extraction.baseline import extract_clinical_fields
from medscribe.extraction.schema import ClinicalFields
from medscribe.notes.soap import render_soap_note


@dataclass(frozen=True)
class PipelineResult:
    """Container for the intermediate and final pipeline outputs."""

    raw_transcript: str
    cleaned_transcript: str
    fields: ClinicalFields
    note: str


def run_text_pipeline(transcript: str):
    """Run the current transcript-to-note skeleton."""
    cleaned = clean_transcript(transcript)
    fields = extract_clinical_fields(cleaned)
    note = render_soap_note(fields)
    return PipelineResult(
        raw_transcript=transcript,
        cleaned_transcript=cleaned,
        fields=fields,
        note=note,
    )
