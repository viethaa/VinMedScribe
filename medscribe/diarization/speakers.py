"""Speaker diarization placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerSegment:
    """One diarized transcript span."""

    speaker: str
    start_seconds: float | None
    end_seconds: float | None
    text: str


def diarize_transcript(transcript: str) -> list[SpeakerSegment]:
    """Return a single-speaker fallback until diarization is added."""
    cleaned = transcript.strip()
    if not cleaned:
        return []
    return [
        SpeakerSegment(
            speaker="unknown",
            start_seconds=None,
            end_seconds=None,
            text=cleaned,
        )
    ]
