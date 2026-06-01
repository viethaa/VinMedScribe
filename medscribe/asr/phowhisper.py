"""PhoWhisper ASR adapter.

The existing top-level ``test_phowhisper.py`` script remains the smoke test.
This module is the future integration point for the end-to-end pipeline.
"""

from __future__ import annotations

from pathlib import Path


def transcribe_audio(audio_path: Path, model_size: str = "small", device: str = "auto") -> str:
    """Transcribe audio with PhoWhisper and return plain text.

    This is intentionally a thin placeholder until the smoke-test script is
    refactored into a reusable ASR service.
    """
    raise NotImplementedError(
        "PhoWhisper pipeline integration is pending. Use test_phowhisper.py for current ASR smoke tests. "
        f"Requested audio={audio_path}, model={model_size}, device={device}."
    )
