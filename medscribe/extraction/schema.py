"""Shared data structures for extracted clinical fields."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Medication:
    """Medication mention with optional dosing details."""

    name: str
    dosage: str | None = None
    frequency: str | None = None
    duration: str | None = None


@dataclass(frozen=True)
class ClinicalFields:
    """Structured fields used to render a clinical note."""

    chief_complaint: str | None = None
    symptoms: list[str] = field(default_factory=list)
    duration: str | None = None
    history: list[str] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    diagnosis: list[str] = field(default_factory=list)
    follow_up: str | None = None
    plan: list[str] = field(default_factory=list)
    raw_evidence: list[str] = field(default_factory=list)
