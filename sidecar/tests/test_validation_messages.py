"""422 bodies relay the validator's own sentence, so the Companion's catalog
can match and translate it (no pydantic "Value error, " prefix)."""

from __future__ import annotations

import pytest


def _messages(res) -> list[str]:
    return [entry["msg"] for entry in res.json()["detail"]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a" + "[pause 10s]" * 7, "pause tags may add at most 60s of silence in total"),
        ("a" + "[pause 0ms]" * 201, "a script may contain at most 200 pause tags"),
    ],
)
def test_pause_budget_refusal_is_the_bare_sentence(client, stub_engine, text, expected):
    stub_engine()
    res = client.post("/generate", json={"engine": "stub", "text": text})
    assert res.status_code == 422
    messages = _messages(res)
    assert messages == [expected]
    assert not any(message.startswith("Value error") for message in messages)


def test_non_value_errors_keep_the_pydantic_message(client, stub_engine):
    stub_engine()
    res = client.post("/generate", json={"engine": "stub", "text": "hi", "speed": "fast"})
    assert res.status_code == 422
    messages = _messages(res)
    assert messages
    assert all(message for message in messages)
