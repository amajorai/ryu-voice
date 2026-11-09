"""Engine errors reach people through "synthesis failed in '{engine}': {error}",
so the inner sentence must be plain prose the satellite catalog can translate:
no request field names, and no local paths (those go to the log)."""

from __future__ import annotations

import logging
import sys
import types

import pytest


@pytest.fixture
def stub_engine_modules(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_qwen_clone_without_a_sample_says_so_in_plain_words(stub_engine_modules):
    from ryu_tts.backends.qwen_tts import QwenCloneBackend

    with pytest.raises(ValueError) as raised:
        QwenCloneBackend().generate("Hi")
    assert str(raised.value) == "Qwen3-TTS needs a voice sample to clone from."


def test_unreadable_sample_keeps_the_path_out_of_the_message(caplog):
    from ryu_tts.backends.chatterbox import _reference_path

    path = "/private/home/someone/voice.wav"
    with caplog.at_level(logging.WARNING, logger="ryu_tts.backends.chatterbox"):
        with pytest.raises(ValueError) as raised:
            _reference_path(path)
    assert str(raised.value) == "The voice sample could not be read."
    assert path not in str(raised.value)
    assert path in caplog.text
