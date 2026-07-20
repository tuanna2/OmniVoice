"""Helpers for generating best-effort SRT subtitles from text."""

from __future__ import annotations

import re
from typing import List

SRT_TARGET_CHUNK_WORDS = 11
SRT_MIN_CHUNK_WORDS = 10
SRT_MAX_CHUNK_WORDS = 12

_SENTENCE_END_RE = re.compile(r"([.!?。！？]+)")


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _split_sentence_fragments(text: str) -> List[str]:
    """Split text into sentence fragments while preserving sentence punctuation."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []

    parts = _SENTENCE_END_RE.split(cleaned)
    fragments: List[str] = []
    i = 0
    while i < len(parts):
        body = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        fragment = f"{body}{punct}".strip()
        if fragment:
            fragments.append(fragment)
        i += 2

    return fragments or [cleaned]


def _tokenize_fragment(fragment: str) -> List[str]:
    """Tokenize a fragment at word boundaries, falling back to characters for CJK."""
    if any(ch.isspace() for ch in fragment):
        return fragment.split()
    return list(fragment)


def _join_tokens(tokens: List[str]) -> str:
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0].strip()
    if all(len(tok) == 1 for tok in tokens):
        return "".join(tokens).strip()
    return " ".join(tokens).strip()


def _split_tokens_balanced(tokens: List[str]) -> List[List[str]]:
    """Split tokens into balanced chunks around the target size."""
    n = len(tokens)
    if n <= SRT_MAX_CHUNK_WORDS:
        return [tokens]

    chunk_count = max(1, int(round(n / SRT_TARGET_CHUNK_WORDS)))
    if chunk_count <= 1:
        return [tokens]

    base, remainder = divmod(n, chunk_count)
    sizes = [base] * chunk_count
    for i in range(remainder):
        sizes[chunk_count - 1 - i] += 1

    chunks: List[List[str]] = []
    start = 0
    for size in sizes:
        end = start + size
        chunk = tokens[start:end]
        if chunk:
            chunks.append(chunk)
        start = end

    if chunks and len(chunks[-1]) < SRT_MIN_CHUNK_WORDS and len(chunks) > 1:
        chunks[-2].extend(chunks[-1])
        chunks.pop()

    return chunks or [tokens]


def split_text_into_srt_cues(text: str) -> List[str]:
    """Split text into readable SRT cue text blocks."""
    cues: List[str] = []
    for fragment in _split_sentence_fragments(text):
        tokens = _tokenize_fragment(fragment)
        if len(tokens) <= SRT_MAX_CHUNK_WORDS:
            cues.append(_join_tokens(tokens))
            continue
        for chunk_tokens in _split_tokens_balanced(tokens):
            cue = _join_tokens(chunk_tokens)
            if cue:
                cues.append(cue)

    return [cue for cue in cues if cue]


def build_srt_content(text: str, duration_s: float) -> str:
    """Build a best-effort SRT file from generated text and audio duration."""
    cues = split_text_into_srt_cues(text)
    if not cues:
        return ""

    total_duration = max(float(duration_s), 0.01)
    total_weight = sum(max(len(cue), 1) for cue in cues)

    srt_parts = []
    start_s = 0.0
    for idx, cue in enumerate(cues, start=1):
        weight = max(len(cue), 1)
        if idx == len(cues):
            end_s = total_duration
        else:
            end_s = start_s + total_duration * weight / total_weight
        end_s = max(end_s, start_s + 0.01)
        srt_parts.append(
            f"{idx}\n"
            f"{format_srt_timestamp(start_s)} --> {format_srt_timestamp(end_s)}\n"
            f"{cue}\n"
        )
        start_s = end_s

    return "\n".join(srt_parts).strip() + "\n"
