"""File IO helpers for pipeline inputs and outputs."""

from __future__ import annotations

from pathlib import Path


def read_text(path: Path) -> str:
    """Read UTF-8 text from disk."""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
