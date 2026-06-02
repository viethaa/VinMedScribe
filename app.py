"""VinMedScribe web server — serves the UI and transcription API."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from test_phowhisper import load_asr_pipeline, run_asr

# ffmpeg handles all these; .webm/.ogg come from browser MediaRecorder
ACCEPTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".mp4"}

_asr = None
_asr_lock = threading.Lock()
DEFAULT_WEB_MODEL_SIZE = os.environ.get("PHOWHISPER_MODEL_SIZE", "medium")
ASR_CHUNK_LENGTH_SECONDS = int(os.environ.get("ASR_CHUNK_LENGTH_SECONDS", "0"))


def _get_asr():
    global _asr
    if _asr is not None:
        return _asr

    with _asr_lock:
        if _asr is None:
            print(f"[web] Loading PhoWhisper {DEFAULT_WEB_MODEL_SIZE}...", flush=True)
            _asr = load_asr_pipeline(DEFAULT_WEB_MODEL_SIZE, "auto")
            print(f"[web] PhoWhisper {DEFAULT_WEB_MODEL_SIZE} ready.", flush=True)
    return _asr


HOST = "0.0.0.0"
PORT = 8000


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    link = f"http://localhost:{PORT}"
    print("\n" + "─" * 48)
    print("  VinMedScribe is running")
    print(f"  ➜  Open in your browser:  {link}")
    print("─" * 48 + "\n")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _get_asr)
    yield


app = FastAPI(title="VinMedScribe", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": _asr is not None, "model": f"PhoWhisper {DEFAULT_WEB_MODEL_SIZE}"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{suffix}'. Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS))}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        src = Path(tmp.name)

    try:
        t0 = time.perf_counter()
        size_bytes = src.stat().st_size
        if size_bytes == 0:
            raise HTTPException(400, "Received an empty audio file. Please record again.")

        print(
            f"[web] Received {file.filename or src.name} ({size_bytes} bytes); preparing ASR.",
            flush=True,
        )
        asr = _get_asr()
        loaded_at = time.perf_counter()
        print(f"[web] Running ASR on {src.name}...", flush=True)

        # run_asr() in test_phowhisper.py is the shared transcription function:
        # it handles ffmpeg conversion and runs PhoWhisper with our standard params.
        raw = run_asr(src, asr=asr, timestamps=True, chunk_length_s=ASR_CHUNK_LENGTH_SECONDS)
        transcribed_at = time.perf_counter()

        transcript = (raw.get("text") or "").strip()
        chunks = [
            {
                "text": c.get("text", "").strip(),
                "start": (c.get("timestamp") or [None])[0],
                "end": (c.get("timestamp") or [None, None])[1],
            }
            for c in (raw.get("chunks") or [])
        ]
        elapsed = round(time.perf_counter() - t0, 1)
        print(
            "[web] Transcription complete: "
            f"load_wait={loaded_at - t0:.1f}s, "
            f"asr={transcribed_at - loaded_at:.1f}s, "
            f"total={elapsed:.1f}s.",
            flush=True,
        )

        soap_note = None
        if transcript:
            from medscribe.pipeline import run_text_pipeline
            soap_note = run_text_pipeline(transcript).note

        return {
            "status": "success",
            "transcript": transcript,
            "chunks": chunks,
            "soap_note": soap_note,
            "elapsed_seconds": elapsed,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        src.unlink(missing_ok=True)


# Static files must be last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    # Run with: python app.py
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
