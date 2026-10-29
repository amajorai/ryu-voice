"""Script planning: inline tags, pauses, and sentence-aware chunking.

`plan_script` turns one request's text into an ordered list of speech chunks and
silence gaps. Every engine gets the same treatment:

- `[pause]`, `[pause 500ms]`, `[pause 1.5s]` become silence segments.
- Any other `[tag]` survives only when the engine lists it in
  `EngineConfig.paralinguistic_tags` (e.g. Chatterbox Turbo's `[laugh]`);
  otherwise it is stripped so engines never read tag names aloud.
- Speech is split at sentence boundaries into chunks of at most `max_chars`
  characters, without breaking on abbreviations ("Dr.", "e.g."), initials,
  decimals ("3.14"), or inside a kept tag. CJK full-width terminators split
  without needing a following space.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 400
DEFAULT_PAUSE_MS = 500
MAX_PAUSE_MS = 10_000
# Request-wide pause budget: each tag is capped at MAX_PAUSE_MS, but without a
# count/total bound a short script of repeated `[pause 10s]` tags would plan
# hours of silence and allocate gigabytes of samples.
MAX_PAUSES = 200
MAX_TOTAL_PAUSE_MS = 60_000
# Longest script one `/generate` request accepts (Core and the host bridge
# enforce the same bound).
MAX_TEXT_CHARS = 50_000

# `[pause]`, `[pause 300ms]`, `[pause 1.5s]`, `[pause 2 s]`.
_PAUSE_RE = re.compile(
    r"\[\s*pause(?:\s+(\d+(?:\.\d+)?)\s*(ms|s)?)?\s*\]", re.IGNORECASE
)
# Any other bracketed tag (single line, short).
_TAG_RE = re.compile(r"\[([^\[\]\n]{1,40})\]")

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc",
        "inc", "ltd", "co", "corp", "no", "fig", "approx", "dept", "est",
        "gen", "gov", "sgt", "capt", "col", "lt", "rev", "jan", "feb", "mar",
        "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "ph.d",
    }
)
_LATIN_TERMINATORS = ".!?…"
_CJK_TERMINATORS = "。！？；"
_CLOSERS = "\"'”’)]»」』"
_CLAUSE_BREAKS = ",;:，、；："


@dataclass(frozen=True)
class Segment:
    """One planned piece of output: speech text, or a pause in milliseconds."""

    kind: str  # "speech" | "pause"
    text: str = ""
    pause_ms: int = 0


def _pause_ms(amount: str | None, unit: str | None) -> int:
    if amount is None:
        return DEFAULT_PAUSE_MS
    value = float(amount)
    ms = value * 1000 if (unit or "ms").lower() == "s" else value
    # A bare number reads as milliseconds unless it is tiny ("[pause 2]" = 2s).
    if unit is None and value <= 10:
        ms = value * 1000
    return int(min(max(ms, 0), MAX_PAUSE_MS))


class PauseBudgetExceeded(ValueError):
    """The script asks for more pauses, or more total silence, than one request allows."""


def check_pause_budget(text: str) -> None:
    """Raise `PauseBudgetExceeded` when `text` exceeds the request-wide pause budget.

    Counts every pause tag (including zero-length ones, which still cost a
    regex match and a segment) and sums their capped durations.
    """
    count = 0
    total_ms = 0
    for match in _PAUSE_RE.finditer(text):
        count += 1
        if count > MAX_PAUSES:
            raise PauseBudgetExceeded(f"a script may contain at most {MAX_PAUSES} pause tags")
        total_ms += _pause_ms(match.group(1), match.group(2))
        if total_ms > MAX_TOTAL_PAUSE_MS:
            raise PauseBudgetExceeded(
                f"pause tags may add at most {MAX_TOTAL_PAUSE_MS // 1000}s of silence in total"
            )


def filter_tags(text: str, allowed_tags: Iterable[str] = ()) -> str:
    """Strip `[tags]` the engine does not understand; keep allowed ones verbatim."""
    allowed = {t.strip().strip("[]").lower() for t in allowed_tags}

    def _keep_or_drop(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1).strip().lower() in allowed else " "

    stripped = _TAG_RE.sub(_keep_or_drop, text)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def parse_pauses(text: str, allowed_tags: Iterable[str] = ()) -> list[Segment]:
    """Split text on pause tags; filter other tags in the speech between them."""
    segments: list[Segment] = []
    cursor = 0
    for match in _PAUSE_RE.finditer(text):
        speech = filter_tags(text[cursor : match.start()], allowed_tags)
        if speech:
            segments.append(Segment("speech", text=speech))
        ms = _pause_ms(match.group(1), match.group(2))
        if ms > 0:
            segments.append(Segment("pause", pause_ms=ms))
        cursor = match.end()
    tail = filter_tags(text[cursor:], allowed_tags)
    if tail:
        segments.append(Segment("speech", text=tail))
    return segments


def _is_abbreviation(text: str, dot_index: int) -> bool:
    """Whether the `.` at `dot_index` ends an abbreviation or an initial."""
    start = dot_index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    word = text[start:dot_index].lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # Single capital initial ("J. R. R. Tolkien").
    return len(word) == 1 and text[start].isupper()


def _inside_tag(text: str, index: int) -> bool:
    opening = text.rfind("[", 0, index)
    return opening != -1 and text.rfind("]", 0, index) < opening


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Never empty strings; whitespace normalised."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        boundary = False
        end = i + 1
        if ch in _CJK_TERMINATORS:
            boundary = True
        elif ch in _LATIN_TERMINATORS:
            # Swallow runs like "?!" or "..." and trailing quotes/brackets.
            while end < n and text[end] in _LATIN_TERMINATORS:
                end += 1
            while end < n and text[end] in _CLOSERS:
                end += 1
            followed_by_break = end >= n or text[end] == " "
            if ch == "." and end == i + 1 and _is_abbreviation(text, i):
                followed_by_break = False
            boundary = followed_by_break
        if ch in _CJK_TERMINATORS:
            while end < n and text[end] in _CLOSERS:
                end += 1
        if boundary and not _inside_tag(text, i):
            piece = text[start:end].strip()
            if piece:
                sentences.append(piece)
            start = end
            i = end
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long sentence at clause punctuation, then spaces, then hard."""
    pieces: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        window = rest[: max_chars + 1]
        cut = max(window.rfind(c) for c in _CLAUSE_BREAKS)
        if cut < max_chars // 3:
            cut = window.rfind(" ")
        if cut < max_chars // 3:
            cut = max_chars - 1
        # Never cut inside a kept tag; back up to its opening bracket.
        opening = rest.rfind("[", 0, cut + 1)
        if opening > 0 and _inside_tag(rest, cut + 1):
            cut = opening - 1
        pieces.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].strip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Pack whole sentences into chunks of at most `max_chars` characters."""
    if max_chars <= 0:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return [cleaned] if cleaned else []
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        parts = [sentence] if len(sentence) <= max_chars else _split_long(sentence, max_chars)
        for part in parts:
            joiner = "" if not current or _is_cjk(current[-1]) else " "
            if current and len(current) + len(joiner) + len(part) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = f"{current}{joiner}{part}" if current else part
    if current:
        chunks.append(current)
    return chunks


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3000 <= code <= 0x30FF  # CJK punctuation, kana
        or 0x3400 <= code <= 0x9FFF  # CJK ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul
        or 0xFF00 <= code <= 0xFFEF  # full-width forms
    )


def plan_script(
    text: str,
    *,
    allowed_tags: Iterable[str] = (),
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Segment]:
    """Full plan for one request: pauses as-is, speech chunked per sentence.

    Raises `PauseBudgetExceeded` when the pause tags exceed the request budget.
    """
    check_pause_budget(text)
    allowed = tuple(allowed_tags)
    plan: list[Segment] = []
    for segment in parse_pauses(text, allowed):
        if segment.kind == "pause":
            plan.append(segment)
            continue
        plan.extend(Segment("speech", text=c) for c in chunk_text(segment.text, max_chars))
    return plan


def speech_chars(plan: Iterable[Segment]) -> int:
    return sum(len(s.text) for s in plan if s.kind == "speech")
