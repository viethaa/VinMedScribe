"""Audio preprocessing helpers for ASR input."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    limit_seconds: float | None = None,
) -> Path:
    """Convert audio to a Whisper-compatible WAV file."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg before preprocessing audio.")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-vn",
    ]
    if limit_seconds is not None:
        command.extend(["-t", str(limit_seconds)])
    command.append(str(output_path))

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path
