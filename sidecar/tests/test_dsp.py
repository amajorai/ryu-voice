"""Effects (numpy fallback forced, plus pedalboard when installed), loudness
normalization, time-stretch, take assembly helpers, and encoders."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ryu_tts import audio, effects as fx, encode
from ryu_tts.timestretch import pitch_shift, time_stretch, wsola

SR = 16000


def _sine(freq: float, seconds: float = 1.0, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_hz(x: np.ndarray, sr: int = SR) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.fft.rfftfreq(x.size, 1 / sr)[int(np.argmax(spectrum))])


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# ---------------------------------------------------------------------------
# effects — numpy/scipy fallback (pedalboard disabled explicitly)
# ---------------------------------------------------------------------------


def test_params_are_clamped_and_defaulted():
    assert fx.resolve_params("gain", {"gain_db": 99}) == {"gain_db": 24.0}
    assert fx.resolve_params("echo", {"feedback": 5, "bogus": 1})["feedback"] == 0.9
    assert fx.resolve_params("lowpass", {"cutoff_hz": float("nan")})["cutoff_hz"] == 8000.0
    assert fx.resolve_params("compressor", {})["ratio"] == 3.0


def test_gain_scales_by_decibels():
    out = fx.apply_effect("gain", _sine(440), SR, {"gain_db": -6.0206}, use_pedalboard=False)
    assert np.max(np.abs(out)) == pytest.approx(0.25, rel=1e-3)


def test_highpass_and_lowpass_attenuate_out_of_band():
    low, high = _sine(60), _sine(5000)
    assert _rms(fx.apply_effect("highpass", low, SR, {"cutoff_hz": 500}, use_pedalboard=False)) < 0.05 * _rms(low)
    assert _rms(fx.apply_effect("lowpass", high, SR, {"cutoff_hz": 1000}, use_pedalboard=False)) < 0.1 * _rms(high)
    # Cutoffs past Nyquist are clamped instead of erroring.
    passthrough = fx.apply_effect("lowpass", high, SR, {"cutoff_hz": 20000}, use_pedalboard=False)
    assert np.all(np.isfinite(passthrough))


def test_compressor_reduces_loud_material():
    loud = _sine(300, amp=0.9)
    out = fx.apply_effect("compressor", loud, SR, {"threshold_db": -20, "ratio": 8, "attack_ms": 1},
                          use_pedalboard=False)
    assert np.max(np.abs(out[SR // 2 :])) < 0.4
    assert out.size == loud.size


def test_reverb_and_echo_add_a_decaying_tail():
    click = np.zeros(SR // 4, dtype=np.float32)
    click[0] = 1.0
    verb = fx.apply_effect("reverb", click, SR, {"room_size": 0.8, "wet_level": 0.5}, use_pedalboard=False)
    assert verb.size > click.size
    assert np.any(np.abs(verb[SR // 4 :]) > 1e-4)
    assert np.all(np.isfinite(verb)) and np.max(np.abs(verb)) < 4.0
    echo = fx.apply_effect("echo", click, SR, {"delay_ms": 100, "feedback": 0.5, "mix": 1.0}, use_pedalboard=False)
    delay = int(SR * 0.1)
    assert echo[delay] == pytest.approx(1.0) and echo[2 * delay] == pytest.approx(0.5)


def test_numpy_pitch_shift_moves_pitch_and_keeps_duration():
    tone = _sine(220, seconds=1.0)
    up = fx.apply_effect("pitch_shift", tone, SR, {"semitones": 12}, use_pedalboard=False)
    assert up.size == tone.size
    assert _dominant_hz(up[2000:-2000]) == pytest.approx(440, rel=0.05)


def test_every_preset_runs_on_the_numpy_path():
    voice = _sine(180) + 0.2 * _sine(2400)
    for preset in fx.PRESETS:
        out = fx.apply_chain(voice, SR, preset["effects"], use_pedalboard=False)
        assert out.size >= voice.size and np.all(np.isfinite(out)), preset["id"]


@pytest.mark.skipif(not fx.pedalboard_available(), reason="pedalboard not installed")
def test_pedalboard_path_runs_the_same_presets():
    voice = _sine(180)
    for preset in fx.PRESETS:
        out = fx.apply_chain(voice, SR, preset["effects"], use_pedalboard=True)
        assert out.size >= voice.size and np.all(np.isfinite(out)), preset["id"]


def test_unknown_effect_type_raises():
    with pytest.raises(ValueError, match="unknown effect"):
        fx.apply_effect("flanger", _sine(100), SR)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def test_peak_normalize_hits_minus_one_dbfs():
    out = fx.normalize(_sine(440, amp=0.1), SR, "peak")
    assert 20 * math.log10(np.max(np.abs(out))) == pytest.approx(-1.0, abs=0.01)


def test_acx_targets_rms_with_minus_three_ceiling():
    speechy = _sine(200, seconds=3.0, amp=0.05) * (1 + 0.5 * _sine(3, seconds=3.0, amp=1.0))
    out = fx.normalize(speechy, SR, "acx")
    assert fx.gated_rms_db(out, SR) == pytest.approx(-20.0, abs=1.0)
    assert 20 * math.log10(np.max(np.abs(out))) <= -3.0 + 1e-3


def test_podcast_fallback_without_pyloudnorm(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pyloudnorm(name, *args, **kwargs):
        if name == "pyloudnorm":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyloudnorm)
    out = fx.normalize(_sine(300, seconds=2.0, amp=0.02), SR, "podcast")
    assert fx.gated_rms_db(out, SR) == pytest.approx(-16.0, abs=0.5)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-4


def test_normalize_leaves_silence_and_off_alone():
    silent = np.zeros(SR, dtype=np.float32)
    assert np.array_equal(fx.normalize(silent, SR, "podcast"), silent)
    tone = _sine(440)
    assert np.array_equal(fx.normalize(tone, SR, "off"), tone)


def test_limiter_enforces_ceiling():
    out = fx.limit_peak(_sine(100, amp=2.0), SR, -3.0)
    assert np.max(np.abs(out)) <= 10 ** (-3 / 20) + 1e-6


# ---------------------------------------------------------------------------
# time-stretch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.5, 0.8, 1.25, 2.0])
def test_numpy_wsola_scales_duration_and_preserves_pitch(rate):
    tone = _sine(220, seconds=2.0)
    out = time_stretch(tone, SR, rate, backend="numpy")
    assert out.size == pytest.approx(tone.size / rate, rel=0.01)
    assert _dominant_hz(out[2000:-2000]) == pytest.approx(220, rel=0.03)


def test_default_backend_also_preserves_pitch():
    tone = _sine(330, seconds=1.5)
    out = time_stretch(tone, SR, 1.5)
    assert out.size == pytest.approx(tone.size / 1.5, rel=0.05)
    assert _dominant_hz(out[2000:-2000]) == pytest.approx(330, rel=0.03)


def test_stretch_rate_is_clamped_and_identity_is_a_copy():
    tone = _sine(220, seconds=1.0)
    assert time_stretch(tone, SR, 100.0, backend="numpy").size == pytest.approx(tone.size / 4, rel=0.01)
    same = wsola(tone, SR, 1.0)
    assert np.array_equal(same, tone) and same is not tone
    assert pitch_shift(tone, SR, 0).size == tone.size


# ---------------------------------------------------------------------------
# take assembly helpers (audio.py)
# ---------------------------------------------------------------------------


def test_equal_power_crossfade_overlaps_and_keeps_level():
    a = np.full(1000, 0.5, dtype=np.float32)
    b = np.full(1000, 0.5, dtype=np.float32)
    out = audio.crossfade_concat([a, b], 1000, crossfade_ms=100)
    assert out.size == 1900
    # Equal power on correlated material bumps the seam by at most sqrt(2).
    assert np.max(out) <= 0.5 * math.sqrt(2) + 1e-4
    assert audio.crossfade_concat([], 1000).size == 0
    assert audio.crossfade_concat([np.ones(3), np.ones(3)], 1000, crossfade_ms=100).size == 5


def test_silence_trim_and_fades():
    assert audio.silence(250, 16000).size == 4000
    padded = np.concatenate([np.zeros(SR), _sine(440, 0.5), np.zeros(SR)])
    trimmed = audio.trim_silence(padded, SR, pad_ms=40)
    assert trimmed.size == pytest.approx(SR * 0.5 + 2 * SR * 0.04, abs=SR * 0.02)
    assert np.array_equal(audio.trim_silence(np.zeros(100), SR), np.zeros(100))
    faded = audio.fade(np.ones(100, dtype=np.float32), 1000, in_ms=10, out_ms=10)
    assert faded[0] == 0 and faded[-1] == 0 and faded[50] == 1


def test_runaway_cap_uses_duration_per_char_ceiling():
    long_take = np.ones(SR * 30, dtype=np.float32)
    capped, was_capped = audio.cap_runaway(long_take, SR, text_chars=40)
    assert was_capped and capped.size == SR * 10  # 40 chars * 0.25 s
    assert capped[-1] == 0.0
    ok, flagged = audio.cap_runaway(np.ones(SR * 2, dtype=np.float32), SR, text_chars=5)
    assert not flagged and ok.size == SR * 2  # under the 4 s floor


def test_to_float_mono_handles_ints_channels_and_nans():
    assert audio.to_float_mono(np.array([32767, -32767], dtype=np.int16)).tolist() == pytest.approx([1.0, -1.0])
    assert audio.to_float_mono(np.ones((2, 5))).shape == (5,)
    assert audio.to_float_mono(np.array([np.nan, np.inf])).tolist() == [0.0, 0.0]


def test_resample_changes_length():
    assert audio.resample(_sine(100, 1.0, sr=24000), 24000, 16000).size == 16000


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------


def test_wav_encode_matches_to_wav_bytes():
    tone = _sine(440, 0.1)
    data, mime = encode.encode(tone, SR, "wav")
    assert mime == "audio/wav" and data == audio.to_wav_bytes(tone, SR)


def test_flac_and_ogg_fall_back_to_ffmpeg_without_soundfile(monkeypatch):
    if encode.ffmpeg_path() is None:
        pytest.skip("ffmpeg not installed")
    monkeypatch.setattr(encode, "_with_soundfile", lambda *a: None)
    for fmt, magic in (("flac", b"fLaC"), ("ogg", b"OggS")):
        data, _ = encode.encode(_sine(440, 0.2), SR, fmt)
        assert data.startswith(magic)


def test_unknown_format_is_a_value_error():
    with pytest.raises(ValueError):
        encode.encode(_sine(440, 0.1), SR, "aac")


def test_bad_ffmpeg_override_reports_unavailable(monkeypatch):
    monkeypatch.setenv("RYU_FFMPEG", "/nonexistent/ffmpeg")
    with pytest.raises(encode.EncoderUnavailable) as info:
        encode.encode(_sine(440, 0.1), SR, "mp3")
    assert "ffmpeg" in info.value.hint


def test_runaway_budget_weights_cjk_syllables():
    assert audio.speech_length("Hello") == 5
    assert audio.speech_length("你好") == 2 * audio.DENSE_SCRIPT_WEIGHT
    assert audio.speech_length("こんにちは、안녕") == 7 * audio.DENSE_SCRIPT_WEIGHT + 1
    # 40 Chinese characters spoken slowly (~3.5 chars/s, ~11.4 s) must not be cut.
    slow_take = np.ones(int(SR * 11.4), dtype=np.float32) * 0.1
    kept, flagged = audio.cap_runaway(slow_take, SR, audio.speech_length("中" * 40))
    assert not flagged and kept.size == slow_take.size
