"""Sentence splitting, chunk packing, pause parsing, and tag filtering."""

from __future__ import annotations

import pytest
from ryu_tts.chunking import (
    DEFAULT_PAUSE_MS,
    MAX_PAUSE_MS,
    MAX_PAUSES,
    MAX_TOTAL_PAUSE_MS,
    PauseBudgetExceeded,
    Segment,
    check_pause_budget,
    chunk_text,
    filter_tags,
    parse_pauses,
    plan_script,
    split_sentences,
)


def test_splits_on_terminal_punctuation():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


def test_abbreviations_initials_and_decimals_do_not_split():
    text = "Dr. Smith met Mrs. Jones at 3.14 p.m. on Main St. today. J. R. R. Tolkien wrote e.g. this. Done."
    assert split_sentences(text) == [
        "Dr. Smith met Mrs. Jones at 3.14 p.m. on Main St. today.",
        "J. R. R. Tolkien wrote e.g. this.",
        "Done.",
    ]


def test_ellipsis_and_closing_quotes_stay_with_their_sentence():
    assert split_sentences('He said "stop." Then... silence?! Yes.') == [
        'He said "stop."',
        "Then...",
        "silence?!",
        "Yes.",
    ]


def test_cjk_terminators_split_without_spaces():
    assert split_sentences("你好。今天天气很好！你呢？") == ["你好。", "今天天气很好！", "你呢？"]


def test_chunks_pack_sentences_under_the_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    assert " ".join(chunks) == text
    assert all(c.endswith(".") for c in chunks)


def test_overlong_sentence_breaks_at_clauses_then_spaces():
    text = ", ".join(["clause with several words"] * 30) + "."
    chunks = chunk_text(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert all(c.endswith((",", ".")) for c in chunks)


def test_cjk_chunks_join_without_spaces():
    chunks = chunk_text("你好。" * 10, max_chars=12)
    assert chunks[0] == "你好。你好。你好。你好。"


def test_zero_max_chars_disables_chunking():
    assert chunk_text("One.  Two.", max_chars=0) == ["One. Two."]


def test_pause_tags_parse_units_and_defaults():
    segments = parse_pauses("A [pause] B [pause 250ms] C [pause 1.5s] D [pause 2] E [PAUSE 99s] F")
    pauses = [s.pause_ms for s in segments if s.kind == "pause"]
    assert pauses == [DEFAULT_PAUSE_MS, 250, 1500, 2000, MAX_PAUSE_MS]
    assert [s.text for s in segments if s.kind == "speech"] == ["A", "B", "C", "D", "E", "F"]


def test_unknown_tags_are_stripped_and_allowed_tags_kept():
    assert filter_tags("Well [laugh] that is [cough] [Sigh] funny", ["laugh", "sigh"]) == (
        "Well [laugh] that is [Sigh] funny"
    )
    assert filter_tags("Plain [laugh] text") == "Plain text"


def test_plan_combines_pauses_and_chunks():
    plan = plan_script("First one. Second one. [pause 1s] Third [laugh].", allowed_tags=["laugh"], max_chars=16)
    assert plan == [
        Segment("speech", text="First one."),
        Segment("speech", text="Second one."),
        Segment("pause", pause_ms=1000),
        Segment("speech", text="Third [laugh]."),
    ]


def test_kept_tag_is_not_split_across_chunks():
    text = "word " * 18 + "[clear throat] end."
    chunks = chunk_text(filter_tags(text, ["clear throat"]), max_chars=100)
    assert any("[clear throat]" in c for c in chunks)


def test_pause_budget_allows_the_limits_exactly():
    at_total = "a " + "[pause 10s] " * (MAX_TOTAL_PAUSE_MS // MAX_PAUSE_MS)
    check_pause_budget(at_total)
    at_count = "a " + "[pause 0ms] " * MAX_PAUSES
    check_pause_budget(at_count)
    assert sum(s.pause_ms for s in plan_script(at_total) if s.kind == "pause") == MAX_TOTAL_PAUSE_MS


@pytest.mark.parametrize(
    "text",
    [
        # Total silence over budget, even though each tag is within its own cap.
        "a" + "[pause 10s]" * (MAX_TOTAL_PAUSE_MS // MAX_PAUSE_MS + 1),
        # Too many tags, even when each adds no silence.
        "a" + "[pause 0ms]" * (MAX_PAUSES + 1),
        # The reported OOM shape: a tiny script that plans hours of silence.
        "a" + "[pause 10s]" * 150_000,
    ],
)
def test_pause_budget_rejects_over_limit_scripts(text):
    with pytest.raises(PauseBudgetExceeded):
        check_pause_budget(text)
    with pytest.raises(PauseBudgetExceeded):
        plan_script(text)
