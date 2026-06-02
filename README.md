# VinMedScribe 

MedScribe is a proof-of-concept Vietnamese clinical scribe pipeline. The project is being structured to support audio preprocessing, Vietnamese ASR, transcript cleaning, clinical information extraction, deterministic SOAP-note rendering, and evaluation.

It ships with a **dark-themed web UI** for recording or uploading consultation audio and getting an instant transcript plus a SOAP note, alongside the original command-line tools.

## Features

- **Web app**: record from the mic or drag-and-drop an audio file, watch a live waveform while recording, and view the transcript and generated SOAP note — all in a clean, animated dark interface.
- Run VinAI PhoWhisper `small`, `medium`, or `large` models through Hugging Face Transformers.
- Automatically choose CUDA, Apple Silicon MPS, or CPU, with manual device override support.
- Convert audio with `ffmpeg` before transcription for Whisper-compatible input.
- Limit transcription to the first N seconds for quick local tests.
- Print basic timing for model load, audio preparation, and transcription.

## Requirements

- Python 3.10+
- `ffmpeg`
- Python packages:
  - `torch`
  - `transformers`
  - `huggingface_hub`
  - `pyannote.audio` for speaker diarization
  - `fastapi`, `uvicorn`, `python-multipart` for the web app

Install the Python dependencies with:

```bash
python3 -m pip install torch transformers huggingface_hub pyannote.audio fastapi uvicorn python-multipart
```

On macOS, install `ffmpeg` with:

```bash
brew install ffmpeg
```

## Web App

The web app serves a dark-themed UI for recording or uploading audio and runs the
full transcribe → SOAP-note pipeline behind a small FastAPI server.

Start it with:

```bash
python app.py
```

You will see a banner with the link to open:

```text
────────────────────────────────────────────────
  VinMedScribe is running
  ➜  Open in your browser:  http://localhost:8000
────────────────────────────────────────────────
```

Open **http://localhost:8000** in your browser, then either:

- Click the microphone button to record (recording auto-submits when you stop), or
- Drag-and-drop / browse for a `.wav`, `.mp3`, or `.m4a` file and click **Transcribe**.

The header badge shows when the PhoWhisper model has finished loading. The model
loads in the background on startup, so the first request after launch is ready quickly.

You can also run it directly with uvicorn (equivalent, with auto-reload):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API endpoints

- `GET /api/health` — returns `{"ok": true, "model_loaded": <bool>}`.
- `POST /api/transcribe` — multipart form upload (`file`). Accepts `.wav`, `.mp3`,
  `.m4a`, plus the `.webm`/`.ogg`/`.mp4` formats produced by browser recording.
  Returns the transcript, timestamped chunks, generated SOAP note, and elapsed time.

The browser front-end lives in `static/` (`index.html`, `style.css`, `app.js`) and
is served directly by the FastAPI app.

## Model Cache

`app.py` and `test_phowhisper.py` load PhoWhisper from the local Hugging Face cache. Download the model you plan to use before running offline:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('vinai/PhoWhisper-small')"
```

Replace `PhoWhisper-small` with `PhoWhisper-medium` or `PhoWhisper-large` if needed.

## Speaker Diarization + Timestamped ASR

`diarize_transcribe.py` converts input audio to 16 kHz mono WAV, runs
`pyannote/speaker-diarization-3.1`, runs timestamped PhoWhisper ASR, and assigns
each ASR segment to the diarized speaker with the largest timestamp overlap.

Before running it, accept the Hugging Face access conditions for
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then paste
your read-only Hugging Face token into:

```text
huggingface_token.txt
```

The file should contain only the token:

```text
hf_your_token_here
```

`huggingface_token.txt` is ignored by git. You can also use `.env` or
`export HUGGINGFACE_TOKEN=...` if you prefer shell environment variables.

Run diarization and transcription:

```bash
python3 diarize_transcribe.py audio/test1.m4a
```

The ASR model defaults to `vinai/PhoWhisper-small`. Override it with:

```bash
export PHOWHISPER_MODEL_ID=vinai/PhoWhisper-medium
```

## Command-Line Usage

Run the text-first scaffold on a transcript:

```bash
python3 cli.py --transcript data/transcripts/example.txt
```

Write the generated note to disk:

```bash
python3 cli.py --transcript data/transcripts/example.txt --output data/outputs/example_note.md
```

Run the PhoWhisper smoke test on one of the included sample audio files:

```bash
python3 test_phowhisper.py --audio audio/test1.m4a --model small
```

Use a specific device:

```bash
python3 test_phowhisper.py --audio audio/test1.m4a --model small --device cpu
```

Print timestamped chunks:

```bash
python3 test_phowhisper.py --audio audio/test1.m4a --model small --timestamps
```

Only transcribe the first 10 seconds:

```bash
python3 test_phowhisper.py --audio audio/test1.m4a --model small --limit-seconds 10
```

## Project Structure

```text
.
├── app.py                  # FastAPI web server (serves the UI + transcription API)
├── static/                 # Web front-end
│   ├── index.html
│   ├── style.css
│   └── app.js
├── audio/
│   ├── test1.m4a
│   └── test2.m4a
├── data/
│   ├── labels/
│   ├── outputs/
│   └── transcripts/
├── docs/
├── medscribe/
│   ├── asr/
│   ├── audio/
│   ├── cleaning/
│   ├── diarization/
│   ├── evaluation/
│   ├── extraction/
│   └── notes/
├── tests/
├── cli.py
├── diarize_transcribe.py
├── model.py
├── test_phowhisper.py
└── README.md
```

## Pipeline Stages

1. Audio preprocessing: normalize consultation audio for ASR.
2. Speaker diarization: optional speaker segmentation and doctor/patient mapping.
3. Vietnamese ASR: PhoWhisper integration.
4. Transcript cleaning: fillers, repetitions, units, dates, and dosage normalization.
5. Clinical extraction: symptoms, history, medications, diagnosis, follow-up, and plan.
6. Note generation: deterministic SOAP-style rendering.
7. Evaluation: WER, extraction precision/recall/F1, and failure analysis.

## Notes

- `model.py` is currently an empty placeholder.
- The sample files in `audio/` are intended for local smoke testing.
- Larger PhoWhisper models require more memory and may be slower on CPU.
