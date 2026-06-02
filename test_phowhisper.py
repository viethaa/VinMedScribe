"""Smoke test for running VinAI PhoWhisper locally on one audio file."""

from __future__ import annotations

import argparse
import time
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import pipeline


MODEL_IDS = {
    "small": "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "large": "vinai/PhoWhisper-large",
}
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def select_device(requested_device: str) -> tuple[str, int | str, torch.dtype]:
    """Pick the fastest available local device unless the user overrides it."""
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda is not available.")
        return "cuda", 0, torch.float16

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps is not available.")
        return "mps", "mps", torch.float32

    if requested_device == "cpu":
        return "cpu", -1, torch.float32

    if torch.cuda.is_available():
        return "cuda", 0, torch.float16

    # Apple Silicon Macs are much faster with MPS than plain CPU.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Whisper can produce unstable nonsense output on MPS float16.
        return "mps", "mps", torch.float32

    return "cpu", -1, torch.float32


def convert_to_wav(audio_path: Path, output_path: Path, limit_seconds: float | None = None) -> None:
    """Convert input audio to 16 kHz mono WAV for Whisper-style ASR."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg before running this test.")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
    ]
    if limit_seconds:
        command.extend(["-t", str(limit_seconds)])
    command.append(str(output_path))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def load_asr_pipeline(model_size: str, requested_device: str):
    """Load PhoWhisper on the fastest available local device."""
    model_id = MODEL_IDS[model_size]
    device_name, device, dtype = select_device(requested_device)

    print(f"Device: {device_name}")
    print(f"Model: {model_id}")

    try:
        model_path = snapshot_download(model_id, local_files_only=True)
        return pipeline(
            task="automatic-speech-recognition",
            model=model_path,
            torch_dtype=dtype,
            device=device,
            model_kwargs={"low_cpu_mem_usage": True},
            ignore_warning=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {model_id}. Check internet/Hugging Face access or try --model small."
        ) from exc


def run_asr(
    audio_path: Path,
    *,
    asr=None,
    model_size: str = "small",
    requested_device: str = "auto",
    timestamps: bool = False,
    chunk_length_s: int = 0,
    preconvert: bool = False,
    limit_seconds: float | None = None,
) -> dict:
    """Transcribe one audio file and return the raw PhoWhisper result dict.

    This is the single source of truth for transcription, used by both the CLI
    (``transcribe`` below) and the web app (``app.py``). Pass a pre-loaded
    ``asr`` pipeline to avoid reloading the model on every call.
    """
    if asr is None:
        asr = load_asr_pipeline(model_size, requested_device)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_for_asr = audio_path
        should_preconvert = preconvert or limit_seconds or audio_path.suffix.lower() != ".wav"
        if should_preconvert:
            audio_for_asr = Path(tmpdir) / "input_16khz_mono.wav"
            convert_to_wav(audio_path, audio_for_asr, limit_seconds)

        asr_kwargs = {
            "return_timestamps": timestamps,
            "generate_kwargs": {
                "language": "vietnamese",
                "task": "transcribe",
                "num_beams": 1,
                "do_sample": False,
                "max_new_tokens": 128,
            },
        }
        if chunk_length_s > 0:
            asr_kwargs["chunk_length_s"] = chunk_length_s

        try:
            return asr(str(audio_for_asr), **asr_kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA ran out of memory. Re-run with --model small or use CPU.") from exc
        except RuntimeError as exc:
            if "MPS" in str(exc) and requested_device == "auto":
                raise RuntimeError(
                    "MPS failed on this audio/model. Re-run with --device cpu or --model small."
                ) from exc
            raise


def transcribe(
    audio_path: Path,
    model_size: str,
    requested_device: str,
    timestamps: bool,
    chunk_length_s: int,
    preconvert: bool,
    limit_seconds: float | None,
) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported format '{audio_path.suffix}'. Use one of: {allowed}")

    start_time = time.perf_counter()
    asr = load_asr_pipeline(model_size, requested_device)
    load_done = time.perf_counter()

    result = run_asr(
        audio_path,
        asr=asr,
        requested_device=requested_device,
        timestamps=timestamps,
        chunk_length_s=chunk_length_s,
        preconvert=preconvert,
        limit_seconds=limit_seconds,
    )

    transcribe_done = time.perf_counter()

    print("\nTranscript:")
    print(result.get("text", "").strip())
    print(
        "\nTiming: "
        f"load={load_done - start_time:.1f}s, "
        f"transcribe={transcribe_done - load_done:.1f}s"
    )

    chunks = result.get("chunks") or []
    if chunks:
        print("\nTimestamped chunks:")
        for chunk in chunks:
            start, end = chunk.get("timestamp") or (None, None)
            text = chunk.get("text", "").strip()
            print(f"[{start} - {end}] {text}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test VinAI PhoWhisper on a local audio file.")
    parser.add_argument("--audio", required=True, help="Path to .wav, .mp3, or .m4a audio.")
    parser.add_argument("--model", choices=MODEL_IDS.keys(), default="small")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--timestamps", action="store_true", help="Print timestamped chunks. Slower.")
    parser.add_argument("--chunk-length", type=int, default=0, help="Chunk size in seconds. 0 disables chunking.")
    parser.add_argument("--preconvert", action="store_true", help="Convert to 16 kHz WAV before ASR.")
    parser.add_argument("--limit-seconds", type=float, help="Only transcribe the first N seconds.")
    args = parser.parse_args()

    try:
        transcribe(
            Path(args.audio).expanduser(),
            args.model,
            args.device,
            args.timestamps,
            args.chunk_length,
            args.preconvert,
            args.limit_seconds,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
