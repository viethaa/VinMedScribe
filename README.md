# MedScribe

MedScribe is a lightweight local smoke-test utility for Vietnamese speech-to-text transcription with VinAI PhoWhisper.

The current script accepts `.wav`, `.mp3`, and `.m4a` audio, converts input to 16 kHz mono WAV when needed, selects the fastest available local device, and prints the transcript with optional timestamped chunks.

## Features

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

Install the Python dependencies with:

```bash
python3 -m pip install torch transformers huggingface_hub
```

On macOS, install `ffmpeg` with:

```bash
brew install ffmpeg
```

## Model Cache

`test_phowhisper.py` loads PhoWhisper from the local Hugging Face cache. Download the model you plan to use before running offline:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('vinai/PhoWhisper-small')"
```

Replace `PhoWhisper-small` with `PhoWhisper-medium` or `PhoWhisper-large` if needed.

## Usage

Run transcription on one of the included sample audio files:

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
├── audio/
│   ├── test1.m4a
│   └── test2.m4a
├── model.py
├── test_phowhisper.py
└── README.md
```

## Notes

- `model.py` is currently an empty placeholder.
- The sample files in `audio/` are intended for local smoke testing.
- Larger PhoWhisper models require more memory and may be slower on CPU.
