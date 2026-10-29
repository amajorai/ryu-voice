"""Shared fixtures: an authenticated TestClient and a stub engine registered in
the live registry, so the full `/generate` pipeline runs without model weights."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from ryu_tts import registry, server

TOKEN = "test-token"


class StubBackend:
    """Deterministic-length sine with seeded noise. Duration tracks text length
    (~60 ms/char) so chunking and runaway caps are observable."""

    def __init__(self, config: registry.EngineConfig, *, sec_per_char: float = 0.06) -> None:
        self.config = config
        self.sec_per_char = sec_per_char
        self.calls: list[tuple[str, dict]] = []
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def generate(self, text, *, voice=None, speed=1.0, language="en", reference_audio=None,
                 seed=None, instruct=None, reference_text=None):
        self.load()
        self.calls.append((text, {"voice": voice, "speed": speed, "language": language,
                                  "reference_audio": reference_audio, "seed": seed,
                                  "instruct": instruct, "reference_text": reference_text}))
        sr = self.config.sample_rate
        n = int(sr * self.sec_per_char * len(text) / max(speed, 0.25))
        t = np.arange(n) / sr
        tone = 0.4 * np.sin(2 * np.pi * 220 * t)
        noise = 0.01 * np.random.standard_normal(n)
        # 100 ms of leading/trailing silence that the server should trim.
        pad = np.zeros(int(sr * 0.1))
        return np.concatenate([pad, tone + noise, pad]).astype(np.float32), sr

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


def make_stub_config(**overrides) -> registry.EngineConfig:
    fields = {
        "id": "stub",
        "display_name": "Stub",
        "description": "test engine",
        "backend_module": "unused",
        "backend_class": "Unused",
        "import_name": "",
        "default_voice": "a",
        "voices": ["a"],
        "sample_rate": 16000,
        "supports_cloning": False,
        "native_speed": True,
        "paralinguistic_tags": ["laugh"],
    }
    fields.update(overrides)
    return registry.EngineConfig(**fields)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_EXPECTED_TOKEN", TOKEN)
    test_client = TestClient(server.app)
    test_client.headers.update({"Authorization": f"Bearer {TOKEN}"})
    return test_client


@pytest.fixture
def stub_engine(monkeypatch):
    """Register a stub engine; returns a factory `(**config_overrides) -> backend`."""

    def _register(backend_cls=StubBackend, sec_per_char: float = 0.06, **overrides):
        config = make_stub_config(**overrides)
        backend = backend_cls(config, sec_per_char=sec_per_char) if backend_cls is StubBackend else backend_cls(config)
        monkeypatch.setitem(registry._BY_ID, config.id, config)
        monkeypatch.setitem(registry._instances, config.id, backend)
        return backend

    return _register
