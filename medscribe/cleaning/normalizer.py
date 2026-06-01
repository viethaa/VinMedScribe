"""Transcript cleaning utilities."""

from __future__ import annotations

import re

FILLERS = (
    "a",
    "ah",
    "um",
    "uh",
    "ờ",
    "ừ",
    "à",
    "ừm",
)


def clean_transcript(transcript: str) -> str:
    """Apply a first-pass cleanup to a raw transcript."""
    text = transcript.strip()
    text = re.sub(r"\s+", " ", text)

    for filler in FILLERS:
        text = re.sub(rf"\b{re.escape(filler)}\b[, ]*", "", text, flags=re.IGNORECASE)

    return text.strip()
