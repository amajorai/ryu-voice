"""Co-located unit tests for the voice (TTS) sidecar's pure helpers.

Runner: pytest (the sidecar is a Python package — see pyproject.toml). Run with

    pytest test_audio.py

These exercise the two dependency-light pure functions in the package:

  - `ryu_tts.audio.to_wav_bytes` — encodes a numpy sample array to 16-bit PCM
    WAV bytes. Pure: same input -> byte-identical output, no I/O, no model. We
    decode the result back with the stdlib `wave` module and assert on the
    header (RIFF/WAVE, mono, 16-bit, sample rate) and the round-tripped samples.
  - `ryu_tts.registry` lookups (`get_config`, `find_model`, `all_models`) — pure
    reads over the declarative engine registry; no engine deps are imported.

Mirrors the apps-store sidecar test convention (a co-located `*test*` file that
imports the sidecar's own module and asserts on a pure function's output — cf.
browser/sidecar `control.test.ts`), expressed in pytest for a Python sidecar.
"""

from __future__ import annotations

import io
import wave

import numpy as np

from ryu_tts import registry
from ryu_tts.audio import to_wav_bytes


def _decode(wav_bytes: bytes):
    """Parse WAV bytes back into (nchannels, sampwidth, framerate, int16 samples)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2")
    return nchannels, sampwidth, framerate, samples


# ---------------------------------------------------------------------------
# to_wav_bytes — the primary pure function under test
# ---------------------------------------------------------------------------


def test_float_input_produces_valid_mono_16bit_wav_header():
    out = to_wav_bytes(np.array([0.0, 0.5, -0.5, 1.0], dtype="float32"), 24000)

    assert isinstance(out, bytes)
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"

    nchannels, sampwidth, framerate, samples = _decode(out)
    assert nchannels == 1
    assert sampwidth == 2  # 16-bit PCM
    assert framerate == 24000
    assert len(samples) == 4


def test_float_samples_scaled_to_int16_range():
    # 1.0 -> +32767, -1.0 -> -32767, 0.0 -> 0, 0.5 -> ~16383 (round-toward-zero).
    _, _, _, samples = _decode(
        to_wav_bytes(np.array([0.0, 1.0, -1.0, 0.5], dtype="float32"), 16000)
    )
    assert samples[0] == 0
    assert samples[1] == 32767
    assert samples[2] == -32767
    assert samples[3] == 16383


def test_out_of_range_floats_are_clipped_not_wrapped():
    # Values beyond [-1, 1] must saturate at the int16 rails, never overflow/wrap.
    _, _, _, samples = _decode(
        to_wav_bytes(np.array([5.0, -5.0, 1.5, -1.5], dtype="float32"), 24000)
    )
    assert samples[0] == 32767
    assert samples[1] == -32767
    assert samples[2] == 32767
    assert samples[3] == -32767


def test_int16_input_passed_through_unchanged():
    original = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)
    _, sampwidth, _, samples = _decode(to_wav_bytes(original, 22050))
    assert sampwidth == 2
    assert samples.tolist() == original.tolist()


def test_multichannel_input_is_downmixed_to_mono():
    # Shape (2, 4): channel axis is the smaller axis (2) and gets averaged out.
    stereo = np.array(
        [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]], dtype="float32"
    )
    nchannels, _, _, samples = _decode(to_wav_bytes(stereo, 24000))
    assert nchannels == 1
    assert len(samples) == 4  # 4 frames out, not 8
    # Mean of +1.0 and -1.0 channels is 0.0 across the board.
    assert np.all(samples == 0)


def test_sample_rate_is_honoured_and_coerced_to_int():
    # Passing a float rate must not raise and must round-trip as an int framerate.
    _, _, framerate, _ = _decode(
        to_wav_bytes(np.array([0.1, -0.1], dtype="float32"), 48000.0)
    )
    assert framerate == 48000


def test_encoding_is_deterministic():
    arr = np.array([0.25, -0.25, 0.75, -0.75], dtype="float32")
    assert to_wav_bytes(arr, 24000) == to_wav_bytes(arr, 24000)


# ---------------------------------------------------------------------------
# registry — pure lookups over the declarative engine table (no engine deps)
# ---------------------------------------------------------------------------


def test_default_engine_kokoro_is_registered_and_first():
    # Core pins kokoro as the cross-surface default; registry lists it head-of-table.
    assert registry.ENGINES[0].id == "kokoro"
    cfg = registry.get_config("kokoro")
    assert cfg is not None
    assert cfg.default_voice in cfg.voices
    assert cfg.sample_rate == 24000


def test_get_config_unknown_engine_returns_none():
    assert registry.get_config("does-not-exist") is None


def test_engine_ids_are_unique():
    ids = [e.id for e in registry.ENGINES]
    assert len(ids) == len(set(ids))


def test_each_engine_has_exactly_one_default_model_variant():
    for cfg in registry.ENGINES:
        defaults = [m for m in cfg.models if m.default]
        assert len(defaults) == 1, f"{cfg.id} must have exactly one default model"


def test_find_model_resolves_within_engine_only():
    # kokoro's variant is not findable under a different engine id.
    assert registry.find_model("kokoro", "kokoro-82m-v1.0") is not None
    assert registry.find_model("kitten", "kokoro-82m-v1.0") is None
    assert registry.find_model("nope", "kokoro-82m-v1.0") is None


def test_all_models_flattens_every_variant_with_engine_binding():
    rows = registry.all_models()
    expected = sum(len(cfg.models) for cfg in registry.ENGINES)
    assert len(rows) == expected
    assert all(r["engine"] in {e.id for e in registry.ENGINES} for r in rows)
    assert all({"model_name", "hf_repo_id", "installed"} <= r.keys() for r in rows)
