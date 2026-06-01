"""SOAP note rendering."""

from __future__ import annotations

from medscribe.extraction.schema import ClinicalFields


def _list_or_placeholder(values: list[str]) -> str:
    if not values:
        return "- Not extracted"
    return "\n".join(f"- {value}" for value in values)


def render_soap_note(fields: ClinicalFields) -> str:
    """Render extracted fields into a deterministic SOAP-style note."""
    subjective_items = []
    if fields.chief_complaint:
        subjective_items.append(f"Chief complaint: {fields.chief_complaint}")
    if fields.duration:
        subjective_items.append(f"Duration: {fields.duration}")
    subjective_items.extend(f"Symptom: {symptom}" for symptom in fields.symptoms)

    medications = [
        " - ".join(part for part in (med.name, med.dosage, med.frequency, med.duration) if part)
        for med in fields.medications
    ]

    sections = [
        "# SOAP Note",
        "",
        "## Subjective",
        _list_or_placeholder(subjective_items),
        "",
        "## Objective",
        "- Not extracted",
        "",
        "## Assessment",
        _list_or_placeholder(fields.diagnosis),
        "",
        "## Plan",
        _list_or_placeholder(fields.plan),
        "",
        "## Medications",
        _list_or_placeholder(medications),
        "",
        "## Follow-up",
        fields.follow_up or "- Not extracted",
    ]
    return "\n".join(sections)
