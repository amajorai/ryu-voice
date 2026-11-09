"""End-to-end `/generate` pipeline through FastAPI's TestClient with a stub
engine: chunking, pause tags, tag filtering, seed, speed, effects, normalize,
every output format, runaway capping, and the request contract."""

from __future__ import annotations

import io
import subprocess
import sys
import wave

import numpy as np
import pytest
from ryu_tts import effects as fx
from ryu_tts import registry, server
from ryu_tts.encode import ffmpeg_path


def _wav_samples(body: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(body), "rb") as wav:
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0, rate


def _post(client, **body):
    payload = {"engine": "stub", "text": "Hello there."}
    payload.update(body)
    return client.post("/generate", json=payload)


def test_long_text_is_chunked_at_sentences_and_crossfaded(client, stub_engine):
    backend = stub_engine()
    sentence = "This sentence is exactly fifty characters long ok. "
    text = sentence * 20  # ~1000 chars -> several <=400-char chunks
    res = _post(client, text=text)
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "audio/wav"
    assert int(res.headers["x-ryu-tts-chunks"]) == len(backend.calls) >= 3
    assert all(len(call_text) <= 400 for call_text, _ in backend.calls)
    assert all(call_text.endswith(".") for call_text, _ in backend.calls)
    samples, rate = _wav_samples(res.content)
    assert rate == 16000
    # Each chunk's 100 ms silent pads are trimmed (to 40 ms) and seams crossfaded,
    # so the take is shorter than the raw concatenation.
    raw = sum(0.06 * len(t) + 0.2 for t, _ in backend.calls)
    assert samples.size / rate < raw


def test_pause_tags_insert_silence_and_unknown_tags_are_stripped(client, stub_engine):
    backend = stub_engine()
    res = _post(client, text="Hi [laugh] there. [pause 1.5s] Next [cough] line.")
    assert res.status_code == 200, res.text
    spoken = [t for t, _ in backend.calls]
    assert spoken == ["Hi [laugh] there.", "Next line."]  # laugh kept (engine tag), cough dropped
    samples, rate = _wav_samples(res.content)
    # Find the longest run of near-silence: the 1.5 s pause plus trimmed pads.
    quiet = np.abs(samples) < 1e-3
    longest = run = 0
    for flag in quiet:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    assert 1.45 <= longest / rate <= 1.7


def test_seed_makes_takes_reproducible_and_reaches_backend(client, stub_engine):
    backend = stub_engine()
    first = _post(client, seed=7).content
    second = _post(client, seed=7).content
    third = _post(client, seed=8).content
    assert first == second
    assert first != third
    assert backend.calls[0][1]["seed"] == 7


def test_model_is_loaded_before_seeding(client, stub_engine, monkeypatch):
    backend = stub_engine(supports_seed=True)
    order: list[str] = []
    monkeypatch.setattr(backend, "load", lambda: order.append("load"))
    monkeypatch.setattr(server, "seed_everything", lambda seed: order.append(f"seed:{seed}"))
    assert _post(client, seed=7).status_code == 200
    assert order[:2] == ["load", "seed:7"]


def test_seeded_stochastic_engines_share_one_rng_lock(client, stub_engine, monkeypatch):
    """Two different seed-capable engines must not interleave draws from the
    process-global RNGs while either holds a seed."""
    import threading

    held: list[bool] = []

    class Probe:
        def __init__(self, config):
            self.config = config

        def load(self):
            pass

        def generate(self, text, **_kwargs):
            # A non-blocking acquire fails only if the lock is already held.
            held.append(not server._RNG_LOCK.acquire(blocking=False))
            if not held[-1]:
                server._RNG_LOCK.release()
            return np.full(1600, 0.3, dtype=np.float32), 16000

        def unload(self):
            pass

        def is_loaded(self):
            return True

    stub_engine(backend_cls=Probe, id="probe", supports_seed=True)
    assert _post(client, engine="probe").status_code == 200
    stub_engine(backend_cls=Probe, id="plain", supports_seed=False)
    assert _post(client, engine="plain").status_code == 200
    assert _post(client, engine="plain", seed=3).status_code == 200
    assert held == [True, False, True]
    assert isinstance(server._RNG_LOCK, type(threading.Lock()))


def test_instruct_only_reaches_engines_that_support_it(client, stub_engine):
    plain = stub_engine()
    _post(client, instruct="whisper", reference_text="hi")
    assert plain.calls[-1][1]["instruct"] is None
    assert plain.calls[-1][1]["reference_text"] == "hi"
    styled = stub_engine(id="stub", supports_instruct=True)
    _post(client, instruct="whisper")
    assert styled.calls[-1][1]["instruct"] == "whisper"


def test_backend_without_new_kwargs_still_works(client, stub_engine):
    class LegacyBackend:
        def __init__(self, config):
            self.config = config

        def generate(self, text, *, voice=None, speed=1.0, language="en", reference_audio=None, seed=None):
            return np.full(1600, 0.3, dtype=np.float32), 16000

        def load(self):
            pass

        def unload(self):
            pass

        def is_loaded(self):
            return True

    stub_engine(backend_cls=LegacyBackend)
    res = _post(client, instruct="calm", reference_text="x")
    assert res.status_code == 200, res.text


def test_engines_without_native_speed_are_time_stretched(client, stub_engine):
    backend = stub_engine(native_speed=False)
    normal, rate = _wav_samples(_post(client, text="A fairly long sentence for timing purposes.").content)
    fast, _ = _wav_samples(_post(client, text="A fairly long sentence for timing purposes.", speed=2.0).content)
    assert backend.calls[-1][1]["speed"] == 1.0  # engine itself never sees speed
    assert fast.size / normal.size == pytest.approx(0.5, abs=0.06)


def test_native_speed_engines_receive_speed(client, stub_engine):
    backend = stub_engine(native_speed=True)
    _post(client, speed=1.5)
    assert backend.calls[-1][1]["speed"] == 1.5


def test_runaway_output_is_capped_with_header(client, stub_engine):
    stub_engine(sec_per_char=1.0)  # 1 s/char: far beyond the 0.25 s/char ceiling
    res = _post(client, text="Twenty characters!!!")
    assert res.status_code == 200
    assert res.headers["x-ryu-tts-runaway-capped"] == "1"
    samples, rate = _wav_samples(res.content)
    assert samples.size / rate <= 5.1  # max(4 s floor, 20 * 0.25) = 5 s
    assert abs(samples[-1]) < 0.01  # ends in a fade, not a hard cut


def test_effects_and_normalize_apply_end_to_end(client, stub_engine):
    stub_engine()
    dry, _ = _wav_samples(_post(client).content)
    wet, rate = _wav_samples(
        _post(
            client,
            effects=[{"type": "highpass", "params": {"cutoff_hz": 100}},
                     {"type": "reverb", "params": {"room_size": 0.8}}],
            normalize="peak",
        ).content
    )
    assert wet.size > dry.size  # reverb tail
    assert 20 * np.log10(np.max(np.abs(wet))) == pytest.approx(-1.0, abs=0.15)


@pytest.mark.parametrize("mode", ["podcast", "acx"])
def test_loudness_modes_respect_their_peak_ceilings(client, stub_engine, mode):
    stub_engine()
    samples, rate = _wav_samples(_post(client, text="Some speech. " * 10, normalize=mode).content)
    ceiling = -1.0 if mode == "podcast" else -3.0
    assert 20 * np.log10(np.max(np.abs(samples))) <= ceiling + 0.05
    if mode == "acx":
        assert fx.gated_rms_db(samples, rate) == pytest.approx(-20.0, abs=1.0)


@pytest.mark.parametrize(
    ("fmt", "mime", "magic"),
    [("wav", "audio/wav", b"RIFF"), ("flac", "audio/flac", b"fLaC"), ("ogg", "audio/ogg", b"OggS"),
     ("mp3", "audio/mpeg", None)],
)
def test_every_output_format_encodes(client, stub_engine, fmt, mime, magic):
    if fmt == "mp3" and ffmpeg_path() is None:
        pytest.skip("ffmpeg not installed")
    stub_engine()
    res = _post(client, format=fmt)
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == mime
    if magic:
        assert res.content.startswith(magic)
    else:
        assert res.content[:3] == b"ID3" or res.content[0] == 0xFF
    if fmt in ("flac", "ogg"):
        soundfile = pytest.importorskip("soundfile")
        decoded, rate = soundfile.read(io.BytesIO(res.content))
        assert rate == 16000 and decoded.size > 0


def test_missing_encoder_returns_415_with_hint(client, stub_engine, monkeypatch):
    stub_engine()
    monkeypatch.setattr("ryu_tts.encode.ffmpeg_path", lambda: None)
    res = _post(client, format="mp3")
    assert res.status_code == 415
    assert "ffmpeg" in res.json()["hint"]


@pytest.mark.parametrize(
    "body",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"instruct": "x" * 501},
        {"reference_text": "x" * 1001},
        {"effects": [{"type": "speed", "params": {}}]},
        {"effects": [{"type": "gain", "params": {"gain_db": "loud"}}]},
        {"effects": [{"type": "gain", "params": {"db": -6}}]},  # misspelled knob
        {"effects": [{"type": "lowpass", "params": {"hz": 3000}}]},
        {"effects": [{"type": "gain", "params": {}}] * 13},
        {"normalize": "loud"},
        {"format": "aac"},
        {"speed": 10},
    ],
)
def test_contract_validation_rejects_bad_params(client, stub_engine, body):
    stub_engine()
    assert _post(client, **body).status_code == 422


@pytest.mark.parametrize(
    "text",
    [
        "x" * 50_001,
        "a" + "[pause 10s]" * 7,
        "a" + "[pause 0ms]" * 201,
        "a" + "[pause 10s]" * 150_000,
    ],
)
def test_over_limit_text_and_pause_budget_are_rejected_before_synthesis(
    client, stub_engine, text
):
    calls = stub_engine().calls
    res = _post(client, text=text)
    assert res.status_code == 422
    assert calls == []


def test_non_finite_effect_param_is_rejected(client, stub_engine):
    stub_engine()
    raw = '{"engine":"stub","text":"hi","effects":[{"type":"gain","params":{"gain_db":Infinity}}]}'
    res = client.post("/generate", content=raw, headers={"content-type": "application/json"})
    assert res.status_code == 422


def test_null_optional_fields_use_defaults(client, stub_engine):
    stub_engine()
    res = _post(client, effects=None, normalize=None, format=None, instruct=None, seed=None)
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"


def test_only_tags_is_a_502_not_a_crash(client, stub_engine):
    stub_engine()
    res = _post(client, text="[pause 1s] [cough]")
    assert res.status_code == 502
    assert "nothing to speak" in res.json()["error"]


def test_engines_rows_carry_the_capability_contract(client):
    rows = {row["id"]: row for row in client.get("/engines").json()}
    for row in rows.values():
        assert {"supports_instruct", "supports_design", "supports_seed", "native_speed",
                "paralinguistic_tags", "license", "commercial_use", "description"} <= row.keys()
    assert rows["kokoro"]["license"] == "Apache-2.0" and rows["kokoro"]["native_speed"]
    assert rows["pocket"]["native_speed"] is False
    assert rows["chatterbox_turbo"]["paralinguistic_tags"]
    assert rows["qwen_voice_design"]["supports_design"] and rows["qwen_voice_design"]["supports_instruct"]
    assert rows["qwen_custom_voice"]["supports_instruct"] and not rows["qwen"]["supports_instruct"]


def test_effects_route_lists_schema_and_presets(client):
    body = client.get("/effects").json()
    assert {e["type"] for e in body["effects"]} == set(fx.EFFECT_TYPES)
    assert {p["id"] for p in body["presets"]} == {
        "broadcast", "podcast", "warm", "bright", "radio", "telephone", "robotic", "deep", "echo", "cathedral",
    }
    assert body["normalize"] == ["off", "peak", "podcast", "acx"]


def test_routes_other_than_health_require_the_token(client):
    client.headers.pop("Authorization")
    assert client.get("/health").status_code == 200
    assert client.get("/effects").status_code == 401


# ---------------------------------------------------------------------------
# /engines/install regression: server.py used `sys` without importing it, so
# every install attempt raised NameError (HTTP 500).
# ---------------------------------------------------------------------------


@pytest.fixture
def uninstalled_engine(monkeypatch):
    cfg = registry.get_config("kitten")
    state = {"installed": False}
    monkeypatch.setattr(registry, "is_installed", lambda c: state["installed"] if c.id == cfg.id else True)
    return cfg, state


def test_install_outside_a_venv_is_refused_with_409(client, monkeypatch, uninstalled_engine):
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.delenv("RYU_TTS_ALLOW_SYSTEM_INSTALL", raising=False)
    res = client.post("/engines/install", json={"engine": "kitten"})
    assert res.status_code == 409
    assert "virtual environment" in res.json()["error"]


def test_install_inside_a_venv_runs_the_registry_command(client, monkeypatch, uninstalled_engine):
    cfg, state = uninstalled_engine
    monkeypatch.setattr(sys, "prefix", "/fake/venv")
    seen: list[list[str]] = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        state["installed"] = True
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    res = client.post("/engines/install", json={"engine": "kitten"})
    assert res.status_code == 200, res.text
    assert res.json() == {"engine": "kitten", "installed": True, "already_installed": False}
    assert seen[0][-len(cfg.pip_packages):] == cfg.pip_packages


def test_install_refuses_a_conflicting_torch_family(client, monkeypatch):
    monkeypatch.setattr(sys, "prefix", "/fake/venv")
    monkeypatch.setattr(registry, "is_installed", lambda c: c.import_name == "chatterbox")
    ran: list[object] = []
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: ran.append(a))
    res = client.post("/engines/install", json={"engine": "qwen"})
    assert res.status_code == 409
    assert "Chatterbox" in res.json()["error"] and "transformers" in res.json()["hint"]
    assert ran == []
