"""Diarize audio, run timestamped PhoWhisper ASR, and label transcript turns.

Setup:
  python3 -m pip install torch transformers huggingface_hub pyannote.audio
  brew install ffmpeg
  cp .env.example .env
  # Paste your read-only Hugging Face token into .env.

You must also accept the access conditions for pyannote/speaker-diarization-3.1
on Hugging Face before the diarization model can be downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DEFAULT_ASR_MODEL_ID = "vinai/PhoWhisper-small"
DEFAULT_ASR_CHUNK_LENGTH_SECONDS = 30


def convert_to_wav(input_path: str) -> str:
    """Convert input audio to a temporary 16 kHz mono WAV file."""
    source = Path(input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg before running diarization.")

    with tempfile.NamedTemporaryFile(prefix=f"{source.stem}_", suffix="_16khz_mono.wav", delete=False) as tmp:
        output = Path(tmp.name)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output),
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        output.unlink(missing_ok=True)
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed while converting {source}: {stderr}") from exc

    return str(output)


def diarize_audio(wav_path: str) -> list[dict[str, Any]]:
    """Run pyannote diarization and return speaker timestamp segments."""
    token = _get_huggingface_token()
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN is not set. Add it to .env or export it before running diarization.")

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing diarization dependencies. Install them with: "
            "python3 -m pip install torch pyannote.audio"
        ) from exc

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_ID, use_auth_token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    diarization = pipeline(wav_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(speaker),
            }
        )

    return sorted(segments, key=lambda segment: (segment["start"], segment["end"]))


def transcribe_audio_with_timestamps(wav_path: str) -> list[dict[str, Any]]:
    """Run PhoWhisper ASR with timestamps and return text segments."""
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import pipeline as transformers_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing ASR dependencies. Install them with: "
            "python3 -m pip install torch transformers huggingface_hub"
        ) from exc

    _load_local_env()
    device_name, device, dtype = _select_asr_device(torch)
    model_id = os.environ.get("PHOWHISPER_MODEL_ID", DEFAULT_ASR_MODEL_ID)
    token = os.environ.get("HUGGINGFACE_TOKEN") or None
    chunk_length_s = int(os.environ.get("ASR_CHUNK_LENGTH_SECONDS", DEFAULT_ASR_CHUNK_LENGTH_SECONDS))

    try:
        model_path = snapshot_download(model_id, token=token)
    except Exception as exc:
        raise RuntimeError(f"Could not load ASR model {model_id}. Check Hugging Face access.") from exc

    asr = transformers_pipeline(
        task="automatic-speech-recognition",
        model=model_path,
        torch_dtype=dtype,
        device=device,
        model_kwargs={"low_cpu_mem_usage": True},
        ignore_warning=True,
    )

    try:
        result = asr(
            wav_path,
            return_timestamps=True,
            chunk_length_s=chunk_length_s,
            generate_kwargs={
                "language": "vietnamese",
                "task": "transcribe",
                "num_beams": 1,
                "do_sample": False,
            },
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA ran out of memory. Use a smaller ASR model or CPU.") from exc
    except RuntimeError as exc:
        if device_name == "mps":
            raise RuntimeError("MPS ASR failed. Try MEDSCRIBE_DEVICE=cpu.") from exc
        raise

    segments = []
    for chunk in result.get("chunks") or []:
        text = str(chunk.get("text", "")).strip()
        start, end = _parse_timestamp(chunk.get("timestamp"))
        if not text or start is None or end is None:
            continue
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

    return segments


def merge_asr_with_diarization(
    asr_segments: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign each ASR segment to the diarized speaker with the largest overlap."""
    merged = []
    for asr_segment in asr_segments:
        asr_start = float(asr_segment["start"])
        asr_end = float(asr_segment["end"])
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for diarization_segment in diarization_segments:
            overlap = _timestamp_overlap(
                asr_start,
                asr_end,
                float(diarization_segment["start"]),
                float(diarization_segment["end"]),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(diarization_segment["speaker"])

        merged.append(
            {
                "start": round(asr_start, 3),
                "end": round(asr_end, 3),
                "speaker": best_speaker,
                "text": str(asr_segment.get("text", "")).strip(),
            }
        )

    return merged


def _select_asr_device(torch_module: Any) -> tuple[str, int | str, Any]:
    requested_device = os.environ.get("MEDSCRIBE_DEVICE", "auto").lower()

    if requested_device == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("MEDSCRIBE_DEVICE=cuda was set, but CUDA is not available.")
        return "cuda", 0, torch_module.float16

    if requested_device == "mps":
        if not getattr(torch_module.backends, "mps", None) or not torch_module.backends.mps.is_available():
            raise RuntimeError("MEDSCRIBE_DEVICE=mps was set, but MPS is not available.")
        return "mps", "mps", torch_module.float32

    if requested_device == "cpu":
        return "cpu", -1, torch_module.float32

    if torch_module.cuda.is_available():
        return "cuda", 0, torch_module.float16

    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps", "mps", torch_module.float32

    return "cpu", -1, torch_module.float32


def _get_huggingface_token() -> str | None:
    _load_local_env()
    return os.environ.get("HUGGINGFACE_TOKEN")


def _load_local_env() -> None:
    """Load simple KEY=value entries from .env without adding a dependency."""
    paths = [Path(__file__).resolve().parent / ".env"]
    cwd_env = Path.cwd() / ".env"
    if cwd_env not in paths:
        paths.append(cwd_env)

    for env_path in paths:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue

            os.environ[key] = value.strip().strip("'\"")


def _parse_timestamp(timestamp: Any) -> tuple[float | None, float | None]:
    if not timestamp or len(timestamp) != 2:
        return None, None

    start, end = timestamp
    if start is None and end is None:
        return None, None
    if start is None:
        start = end
    if end is None:
        end = start

    return float(start), float(end)


def _timestamp_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _print_readable_transcript(segments: list[dict[str, Any]]) -> None:
    print("\nTranscript:")
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if text:
            print(f"{segment['speaker']}: {text}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diarize audio and print a speaker-labeled PhoWhisper transcript.",
        epilog=(
            "Setup: install torch, transformers, huggingface_hub, pyannote.audio, and ffmpeg; "
            "then put HUGGINGFACE_TOKEN in .env or export it."
        ),
    )
    parser.add_argument("input_audio", help="Input audio path, such as input.m4a.")
    parser.add_argument("--keep-wav", action="store_true", help="Keep the temporary 16 kHz mono WAV file.")
    args = parser.parse_args(argv)

    wav_path = ""
    try:
        wav_path = convert_to_wav(args.input_audio)
        diarization_segments = diarize_audio(wav_path)
        asr_segments = transcribe_audio_with_timestamps(wav_path)
        merged_segments = merge_asr_with_diarization(asr_segments, diarization_segments)

        print(json.dumps(merged_segments, indent=2, ensure_ascii=False))
        _print_readable_transcript(merged_segments)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if wav_path and not args.keep_wav:
            Path(wav_path).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
