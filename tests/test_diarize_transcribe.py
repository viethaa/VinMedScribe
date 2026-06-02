"""Tests for speaker diarization transcript merging."""

from __future__ import annotations

from diarize_transcribe import _get_diarization_annotation, merge_asr_with_diarization


def test_merge_assigns_speaker_with_largest_overlap():
    asr_segments = [
        {"start": 0.5, "end": 2.5, "text": "xin chao"},
        {"start": 2.6, "end": 4.0, "text": "toi bi dau dau"},
    ]
    diarization_segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 4.5, "speaker": "SPEAKER_01"},
    ]

    merged = merge_asr_with_diarization(asr_segments, diarization_segments)

    assert merged == [
        {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01", "text": "xin chao"},
        {"start": 2.6, "end": 4.0, "speaker": "SPEAKER_01", "text": "toi bi dau dau"},
    ]


def test_merge_uses_unknown_when_no_overlap_exists():
    asr_segments = [{"start": 10.0, "end": 11.0, "text": "tam dung"}]
    diarization_segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    merged = merge_asr_with_diarization(asr_segments, diarization_segments)

    assert merged == [{"start": 10.0, "end": 11.0, "speaker": "UNKNOWN", "text": "tam dung"}]


def test_get_diarization_annotation_prefers_exclusive_output():
    class Output:
        speaker_diarization = "regular"
        exclusive_speaker_diarization = "exclusive"

    assert _get_diarization_annotation(Output()) == "exclusive"


def test_get_diarization_annotation_accepts_legacy_annotation():
    annotation = object()

    assert _get_diarization_annotation(annotation) is annotation
