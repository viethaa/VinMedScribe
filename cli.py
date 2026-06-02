"""Command line entry point for the text-first MedScribe pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from medscribe.io import read_text, write_text
from medscribe.pipeline import run_text_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the text-first MedScribe pipeline.")
    parser.add_argument("--transcript", required=True, help="Path to a UTF-8 transcript text file.")
    parser.add_argument("--output", help="Optional path for the generated SOAP note.")
    args = parser.parse_args()

    result = run_text_pipeline(read_text(Path(args.transcript)))

    if args.output:
        write_text(Path(args.output), result.note)
    else:
        print(result.note)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
