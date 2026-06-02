"""VinMedScribe web server — serves the UI and transcription API."""
from __future__ import annotations

import os
import shutil
import tempfile
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
DEFAULT_WEB_MODEL_SIZE = os.environ.get("PHOWHISPER_MODEL_SIZE", "medium")
ASR_CHUNK_LENGTH_SECONDS = int(os.environ.get("ASR_CHUNK_LENGTH_SECONDS", "0"))


def _get_asr():
    global _asr
    if _asr is None:
        _asr = load_asr_pipeline(DEFAULT_WEB_MODEL_SIZE, "auto")
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
        asr = _get_asr()

        # run_asr() in test_phowhisper.py is the shared transcription function:
        # it handles ffmpeg conversion and runs PhoWhisper with our standard params.
        raw = run_asr(src, asr=asr, timestamps=True, chunk_length_s=ASR_CHUNK_LENGTH_SECONDS)

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
