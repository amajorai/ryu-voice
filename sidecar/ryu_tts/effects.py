"""Post-synthesis effects chain, built-in presets, and loudness normalization.

Every effect has a numpy/scipy implementation so the chain works on the base
install. When `pedalboard` (Spotify, GPL-3.0; opt-in `pedalboard` extra, never
bundled) is importable, reverb, compressor, and pitch shift use its
higher-quality DSP instead. Parameters are clamped to the ranges published by `GET /effects`, so a
request can never push a filter past Nyquist or a feedback loop past unity.

Normalization targets (`normalize` on `/generate`):
- `peak`: scale to a -1 dBFS sample peak.
- `podcast`: -16 LUFS integrated (pyloudnorm when installed, else a gated-RMS
  approximation of BS.1770 without K-weighting), with a -1 dBFS peak ceiling.
- `acx`: -20 dBFS gated RMS (inside ACX's -23..-18 window) with a -3 dBFS peak
  ceiling.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .timestretch import pedalboard_enabled
from .timestretch import pitch_shift as _numpy_pitch_shift

logger = logging.getLogger(__name__)

EFFECT_TYPES = ("gain", "highpass", "lowpass", "compressor", "reverb", "echo", "pitch_shift")
NORMALIZE_MODES = ("off", "peak", "podcast", "acx")
MAX_EFFECTS = 12

# type -> param -> (min, max, default, unit)
_PARAMS: dict[str, dict[str, tuple[float, float, float, str]]] = {
    "gain": {"gain_db": (-24.0, 24.0, 0.0, "dB")},
    "highpass": {"cutoff_hz": (20.0, 2000.0, 80.0, "Hz")},
    "lowpass": {"cutoff_hz": (500.0, 20000.0, 8000.0, "Hz")},
    "compressor": {
        "threshold_db": (-60.0, 0.0, -18.0, "dB"),
        "ratio": (1.0, 20.0, 3.0, ":1"),
        "attack_ms": (0.1, 200.0, 5.0, "ms"),
        "release_ms": (5.0, 2000.0, 120.0, "ms"),
        "makeup_db": (0.0, 24.0, 0.0, "dB"),
    },
    "reverb": {
        "room_size": (0.0, 1.0, 0.5, ""),
        "damping": (0.0, 1.0, 0.5, ""),
        "wet_level": (0.0, 1.0, 0.25, ""),
        "dry_level": (0.0, 1.0, 0.4, ""),
    },
    "echo": {
        "delay_ms": (20.0, 1500.0, 250.0, "ms"),
        "feedback": (0.0, 0.9, 0.35, ""),
        "mix": (0.0, 1.0, 0.3, ""),
    },
    "pitch_shift": {"semitones": (-12.0, 12.0, 0.0, "st")},
}

_DESCRIPTIONS = {
    "gain": "Volume change in decibels.",
    "highpass": "Removes rumble below the cutoff.",
    "lowpass": "Removes hiss and brightness above the cutoff.",
    "compressor": "Evens out loud and quiet passages.",
    "reverb": "Room ambience.",
    "echo": "Repeating delay.",
    "pitch_shift": "Raises or lowers pitch without changing duration.",
}


def _fx(effect_type: str, **params: float) -> dict[str, Any]:
    return {"type": effect_type, "params": params}


PRESETS: list[dict[str, Any]] = [
    {
        "id": "broadcast",
        "label": "Broadcast",
        "description": "Tight, upfront voice with firm compression.",
        "effects": [
            _fx("highpass", cutoff_hz=80),
            _fx("compressor", threshold_db=-20, ratio=4, attack_ms=5, release_ms=120, makeup_db=4),
        ],
    },
    {
        "id": "podcast",
        "label": "Podcast",
        "description": "Clean spoken word with gentle compression.",
        "effects": [
            _fx("highpass", cutoff_hz=70),
            _fx("compressor", threshold_db=-18, ratio=3, attack_ms=10, release_ms=150, makeup_db=3),
            _fx("lowpass", cutoff_hz=16000),
        ],
    },
    {
        "id": "warm",
        "label": "Warm",
        "description": "Softened highs for a mellow tone.",
        "effects": [_fx("highpass", cutoff_hz=50), _fx("lowpass", cutoff_hz=6500)],
    },
    {
        "id": "bright",
        "label": "Bright",
        "description": "Thinned lows for a crisp, present tone.",
        "effects": [
            _fx("highpass", cutoff_hz=180),
            _fx("compressor", threshold_db=-20, ratio=2, attack_ms=5, release_ms=100, makeup_db=2),
        ],
    },
    {
        "id": "radio",
        "label": "Radio",
        "description": "AM-radio band with heavy compression.",
        "effects": [
            _fx("highpass", cutoff_hz=300),
            _fx("lowpass", cutoff_hz=4500),
            _fx("compressor", threshold_db=-24, ratio=6, attack_ms=2, release_ms=80, makeup_db=6),
        ],
    },
    {
        "id": "telephone",
        "label": "Telephone",
        "description": "Narrow 300-3400 Hz phone line.",
        "effects": [
            _fx("highpass", cutoff_hz=400),
            _fx("lowpass", cutoff_hz=3400),
            _fx("gain", gain_db=3),
        ],
    },
    {
        "id": "robotic",
        "label": "Robotic",
        "description": "Metallic comb resonance.",
        "effects": [
            _fx("pitch_shift", semitones=-2),
            _fx("echo", delay_ms=20, feedback=0.6, mix=0.5),
        ],
    },
    {
        "id": "deep",
        "label": "Deep",
        "description": "Lower pitch with rounded highs.",
        "effects": [_fx("pitch_shift", semitones=-3), _fx("lowpass", cutoff_hz=9000)],
    },
    {
        "id": "echo",
        "label": "Echo",
        "description": "Audible slap-back repeats.",
        "effects": [_fx("echo", delay_ms=300, feedback=0.4, mix=0.3)],
    },
    {
        "id": "cathedral",
        "label": "Cathedral",
        "description": "Large, long reverberant hall.",
        "effects": [_fx("reverb", room_size=0.95, damping=0.3, wet_level=0.4, dry_level=0.35)],
    },
]


def pedalboard_available() -> bool:
    return pedalboard_enabled()


def effects_schema() -> dict[str, Any]:
    """The `GET /effects` payload: effect types with param ranges, and presets."""
    return {
        "effects": [
            {
                "type": name,
                "description": _DESCRIPTIONS[name],
                "params": {
                    param: {"min": lo, "max": hi, "default": default, "unit": unit}
                    for param, (lo, hi, default, unit) in params.items()
                },
            }
            for name, params in _PARAMS.items()
        ],
        "presets": PRESETS,
        "normalize": list(NORMALIZE_MODES),
        "max_effects": MAX_EFFECTS,
        "engine": "pedalboard" if pedalboard_available() else "numpy",
    }


def param_names(effect_type: str) -> list[str]:
    """The param names an effect accepts (the request contract rejects others)."""
    return list(_PARAMS.get(effect_type, {}))


def resolve_params(effect_type: str, params: dict[str, float] | None) -> dict[str, float]:
    """Fill defaults and clamp every known param; unknown names are ignored here
    (presets and direct callers), while `/generate` rejects them up front."""
    spec = _PARAMS[effect_type]
    given = params or {}
    resolved: dict[str, float] = {}
    for name, (lo, hi, default, _unit) in spec.items():
        value = given.get(name, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value):
            value = default
        resolved[name] = min(max(value, lo), hi)
    return resolved


# ---------------------------------------------------------------------------
# numpy/scipy implementations
# ---------------------------------------------------------------------------


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _butter(audio: np.ndarray, sr: int, cutoff: float, kind: str) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    nyquist = sr / 2.0
    cutoff = min(cutoff, nyquist * 0.95)
    if kind == "lowpass" and cutoff >= nyquist * 0.95:
        return audio  # nothing above the cutoff to remove at this rate
    sos = butter(2, cutoff / nyquist, btype=kind, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _compress(audio: np.ndarray, sr: int, p: dict[str, float]) -> np.ndarray:
    """Feed-forward peak compressor with attack/release smoothing in the dB domain,
    evaluated on 32-sample blocks and interpolated back to sample rate."""
    block = 32
    n_blocks = int(np.ceil(audio.size / block))
    if n_blocks == 0:
        return audio
    padded = np.pad(np.abs(audio), (0, n_blocks * block - audio.size))
    level_db = 20.0 * np.log10(np.maximum(padded.reshape(n_blocks, block).max(axis=1), 1e-9))
    over = np.maximum(level_db - p["threshold_db"], 0.0)
    target = -over * (1.0 - 1.0 / p["ratio"])  # gain reduction (<= 0 dB)
    block_sec = block / sr
    att = math.exp(-block_sec / max(p["attack_ms"] / 1000.0, 1e-4))
    rel = math.exp(-block_sec / max(p["release_ms"] / 1000.0, 1e-4))
    smoothed = np.empty_like(target)
    g = 0.0
    for i, t in enumerate(target):
        coeff = att if t < g else rel
        g = coeff * g + (1.0 - coeff) * t
        smoothed[i] = g
    centers = (np.arange(n_blocks) + 0.5) * block
    gain_db = np.interp(np.arange(audio.size), centers, smoothed)
    return (audio * np.power(10.0, (gain_db + p["makeup_db"]) / 20.0)).astype(np.float32)


def _feedback_delay(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """`u[n] = x[n] + g * u[n - d]`, computed one d-sample block at a time (each
    block depends only on the previous one), so long delays stay vectorized."""
    u = np.array(x, dtype=np.float64, copy=True)
    for start in range(d, u.size, d):
        stop = min(start + d, u.size)
        u[start:stop] += g * u[start - d : stop - d]
    return u


def _damped_comb(x: np.ndarray, d: int, g: float, damp: float) -> np.ndarray:
    """Freeverb comb: `v[n] = x[n] + g * lp[n]`, where `lp` is a one-pole low-pass
    of `v[n - d]`; returns the delayed line `v[n - d]`. Block recursion as above,
    with the one-pole filter state carried across blocks."""
    from scipy.signal import lfilter

    v = np.array(x, dtype=np.float64, copy=True)
    state = 0.0
    for start in range(d, v.size, d):
        stop = min(start + d, v.size)
        lp, zf = lfilter([1.0 - damp], [1.0, -damp], v[start - d : stop - d], zi=[state])
        state = float(zf[0])
        v[start:stop] += g * lp
    return np.concatenate([np.zeros(d), v[:-d]]) if v.size > d else np.zeros_like(v)


def _freeverb(audio: np.ndarray, sr: int, p: dict[str, float]) -> np.ndarray:
    """Schroeder/Moorer reverb in the public-domain Freeverb layout: 8 parallel
    damped feedback combs into 4 series allpasses. Level scaling follows the
    common JUCE/pedalboard convention (dry x2, wet x3) so both paths match."""
    scale = sr / 44100.0
    feedback = p["room_size"] * 0.28 + 0.7
    damp = p["damping"] * 0.4
    tail = int(sr * (0.5 + 2.5 * p["room_size"]))
    x = np.concatenate([audio, np.zeros(tail, dtype=np.float32)]).astype(np.float64)
    wet = np.zeros_like(x)
    for base in (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617):
        wet += _damped_comb(x, max(int(base * scale), 1), feedback, damp)
    wet *= 0.015
    for base in (556, 441, 341, 225):
        d = max(int(base * scale), 1)
        # Allpass: y[n] = -x[n] + u[n - d],  u[n] = x[n] + 0.5 u[n - d].
        u = _feedback_delay(wet, d, 0.5)
        delayed = np.concatenate([np.zeros(d), u[:-d]]) if u.size > d else np.zeros_like(u)
        wet = delayed - wet
    out = x * p["dry_level"] * 2.0 + wet * p["wet_level"] * 3.0
    return out.astype(np.float32)


def _echo(audio: np.ndarray, sr: int, p: dict[str, float]) -> np.ndarray:
    d = max(int(sr * p["delay_ms"] / 1000.0), 1)
    fb = p["feedback"]
    # Let the repeats ring out until they fall below -60 dB.
    repeats = 1 if fb <= 0 else min(int(math.ceil(math.log(1e-3) / math.log(fb))), 40)
    x = np.concatenate([audio, np.zeros(d * repeats, dtype=np.float32)]).astype(np.float64)
    u = _feedback_delay(x, d, fb)
    wet = np.concatenate([np.zeros(d), u[:-d]])
    return ((1.0 - p["mix"]) * x + p["mix"] * wet).astype(np.float32)


def _pedalboard_effect(effect_type: str, audio: np.ndarray, sr: int, p: dict[str, float]) -> np.ndarray | None:
    """Apply one effect through pedalboard; None when it has no equivalent here."""
    import pedalboard

    if effect_type == "reverb":
        plugin = pedalboard.Reverb(
            room_size=p["room_size"], damping=p["damping"], wet_level=p["wet_level"], dry_level=p["dry_level"]
        )
        tail = np.zeros(int(sr * (0.5 + 2.5 * p["room_size"])), dtype=np.float32)
        audio = np.concatenate([audio, tail])
    elif effect_type == "compressor":
        plugin = pedalboard.Compressor(
            threshold_db=p["threshold_db"], ratio=p["ratio"], attack_ms=p["attack_ms"], release_ms=p["release_ms"]
        )
        out = plugin(audio.astype(np.float32), sr)
        return (np.asarray(out, dtype=np.float32).reshape(-1) * _db_to_amp(p["makeup_db"])).astype(np.float32)
    elif effect_type == "pitch_shift":
        plugin = pedalboard.PitchShift(semitones=p["semitones"])
    else:
        return None
    return np.asarray(plugin(audio.astype(np.float32), sr), dtype=np.float32).reshape(-1)


def apply_effect(effect_type: str, audio: np.ndarray, sr: int, params: dict[str, float] | None = None,
                 *, use_pedalboard: bool | None = None) -> np.ndarray:
    if effect_type not in _PARAMS:
        raise ValueError(f"unknown effect '{effect_type}'")
    p = resolve_params(effect_type, params)
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    if use_pedalboard is None:
        use_pedalboard = pedalboard_available()
    if use_pedalboard and effect_type in ("reverb", "compressor", "pitch_shift"):
        try:
            out = _pedalboard_effect(effect_type, x, sr, p)
            if out is not None:
                return out
        except Exception as exc:  # noqa: BLE001 — fall back to numpy DSP
            logger.warning("pedalboard %s failed (%s); using numpy", effect_type, exc)
    if effect_type == "gain":
        return (x * _db_to_amp(p["gain_db"])).astype(np.float32)
    if effect_type == "highpass":
        return _butter(x, sr, p["cutoff_hz"], "highpass")
    if effect_type == "lowpass":
        return _butter(x, sr, p["cutoff_hz"], "lowpass")
    if effect_type == "compressor":
        return _compress(x, sr, p)
    if effect_type == "reverb":
        return _freeverb(x, sr, p)
    if effect_type == "echo":
        return _echo(x, sr, p)
    return _numpy_pitch_shift(x, sr, p["semitones"])


def apply_chain(audio: np.ndarray, sr: int, chain: list[dict[str, Any]],
                *, use_pedalboard: bool | None = None) -> np.ndarray:
    """Apply effects in order. Each item is `{"type": ..., "params": {...}}`."""
    out = np.asarray(audio, dtype=np.float32).reshape(-1)
    for item in chain[:MAX_EFFECTS]:
        out = apply_effect(item["type"], out, sr, item.get("params"), use_pedalboard=use_pedalboard)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

PEAK_TARGET_DB = -1.0
PODCAST_LUFS = -16.0
ACX_RMS_DB = -20.0
ACX_PEAK_DB = -3.0


def peak_db(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


def gated_rms_db(audio: np.ndarray, sr: int) -> float:
    """BS.1770-style gated level without K-weighting: 400 ms blocks at 75%
    overlap, absolute gate -70 dB, relative gate -10 dB below the ungated mean."""
    block = max(int(sr * 0.4), 1)
    hop = max(block // 4, 1)
    if audio.size < block:
        power = np.array([float(np.mean(np.square(audio)))]) if audio.size else np.array([0.0])
    else:
        starts = range(0, audio.size - block + 1, hop)
        power = np.array([float(np.mean(np.square(audio[s : s + block]))) for s in starts])
    power = power[power > 10 ** (-70 / 10)]
    if power.size == 0:
        return -math.inf
    relative = 10 * math.log10(float(np.mean(power))) - 10
    gated = power[10 * np.log10(power) > relative]
    if gated.size == 0:
        gated = power
    return 10 * math.log10(float(np.mean(gated)))


def integrated_loudness(audio: np.ndarray, sr: int) -> float:
    """LUFS via pyloudnorm when installed, else the gated-RMS approximation."""
    try:
        import pyloudnorm

        if audio.size >= int(sr * 0.4):
            return float(pyloudnorm.Meter(sr).integrated_loudness(audio.astype(np.float64)))
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("pyloudnorm failed (%s); using gated RMS", exc)
    return gated_rms_db(audio, sr)


def limit_peak(audio: np.ndarray, sr: int, ceiling_db: float) -> np.ndarray:
    """Look-ahead brickwall limiter: the per-sample gain needed to stay under the
    ceiling is spread over a short window and released smoothly, then a final
    clip guarantees the ceiling."""
    ceiling = _db_to_amp(ceiling_db)
    mag = np.abs(audio)
    if mag.size == 0 or float(mag.max()) <= ceiling:
        return audio
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import lfilter

    needed = np.minimum(1.0, ceiling / np.maximum(mag, 1e-9))
    window = max(int(sr * 0.005), 1)
    gain = minimum_filter1d(needed, size=2 * window + 1, mode="nearest")
    release = math.exp(-1.0 / (sr * 0.05))
    # Smooth upward (release) only: take the min of the raw and smoothed curves.
    smoothed = lfilter([1 - release], [1, -release], gain, zi=[gain[0] * release])[0]
    gain = np.minimum(gain, smoothed)
    return np.clip(audio * gain, -ceiling, ceiling).astype(np.float32)


def normalize(audio: np.ndarray, sr: int, mode: str) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mode == "off" or x.size == 0 or not np.any(x):
        return x
    if mode == "peak":
        return (x * _db_to_amp(PEAK_TARGET_DB - peak_db(x))).astype(np.float32)
    if mode == "podcast":
        level, target, ceiling = integrated_loudness(x, sr), PODCAST_LUFS, PEAK_TARGET_DB
    elif mode == "acx":
        level, target, ceiling = gated_rms_db(x, sr), ACX_RMS_DB, ACX_PEAK_DB
    else:
        raise ValueError(f"unknown normalize mode '{mode}'")
    if not math.isfinite(level):
        return x
    # Bound the make-up gain so near-silence is not boosted into noise.
    gain_db = min(max(target - level, -40.0), 30.0)
    return limit_peak((x * _db_to_amp(gain_db)).astype(np.float32), sr, ceiling)
