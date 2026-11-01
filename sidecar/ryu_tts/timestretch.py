"""Pitch-preserving time-stretch for engines without a native speed control.

Preference order: pedalboard's Rubber Band stretcher, then librosa's phase
vocoder, then the built-in numpy WSOLA below. All three keep pitch while
changing duration; the fallback is lower quality on music but clean on speech.

`rate` follows the speaking-rate convention used by `GenerateRequest.speed`:
`rate=2.0` plays twice as fast (half the duration), `rate=0.5` half as fast.
"""

from __future__ import annotations

import importlib.util
import logging

import numpy as np

logger = logging.getLogger(__name__)

MIN_RATE = 0.25
MAX_RATE = 4.0


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def pedalboard_enabled() -> bool:
    """pedalboard (GPL-3.0) is used only when installed; leave the optional
    package out to keep the numpy-only WSOLA fallback."""
    return _has("pedalboard")


def wsola(audio: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    """Waveform-similarity overlap-add time-stretch (numpy only).

    Output frames are laid down at a fixed hop; each input frame is taken near its
    nominal (rate-scaled) position, nudged within a small tolerance to the offset
    whose waveform best continues the previous frame, which avoids phase smearing.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or abs(rate - 1.0) < 1e-3:
        return x.copy()
    frame = max(int(sample_rate * 0.04) // 2 * 2, 64)
    hop_out = frame // 2
    hop_in = hop_out * rate
    tolerance = max(int(sample_rate * 0.012), 1)
    window = np.hanning(frame + 1)[:-1].astype(np.float32)  # periodic: sums to 1 at 50%

    out_len = int(np.ceil(x.size / rate))
    n_frames = int(np.ceil(out_len / hop_out)) + 1
    # Pad so every candidate read stays in bounds.
    padded = np.pad(x, (tolerance, frame + tolerance + int(hop_in) + 1))
    out = np.zeros(n_frames * hop_out + frame, dtype=np.float32)
    norm = np.zeros_like(out)

    prev_pos = 0
    for k in range(n_frames):
        nominal = int(round(k * hop_in))
        if k == 0:
            pos = nominal
        else:
            # The ideal continuation of the previously copied frame.
            natural = padded[prev_pos + hop_out + tolerance : prev_pos + hop_out + tolerance + frame]
            lo = nominal  # == (nominal - tolerance) + tolerance pad offset
            region = padded[lo : lo + frame + 2 * tolerance]
            if natural.size == frame and region.size >= frame:
                corr = np.correlate(region, natural, mode="valid")
                pos = nominal - tolerance + int(np.argmax(corr))
            else:
                pos = nominal
        pos = max(pos, -tolerance)
        start = pos + tolerance
        seg = padded[start : start + frame]
        if seg.size < frame:
            seg = np.pad(seg, (0, frame - seg.size))
        o = k * hop_out
        out[o : o + frame] += seg * window
        norm[o : o + frame] += window
        prev_pos = pos
    out = np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-3)
    return out[:out_len].astype(np.float32)


def time_stretch(audio: np.ndarray, sample_rate: int, rate: float, *, backend: str | None = None) -> np.ndarray:
    """Change duration by `1/rate` while keeping pitch. `backend` forces one of
    "pedalboard" | "librosa" | "numpy" (tests); default picks the best installed."""
    rate = float(min(max(rate, MIN_RATE), MAX_RATE))
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or abs(rate - 1.0) < 1e-3:
        return x
    choice = backend or ("pedalboard" if pedalboard_enabled() else "librosa" if _has("librosa") else "numpy")
    try:
        if choice == "pedalboard":
            from pedalboard import time_stretch as pb_stretch

            return np.asarray(pb_stretch(x, float(sample_rate), stretch_factor=rate), dtype=np.float32).reshape(-1)
        if choice == "librosa":
            import librosa

            return np.asarray(librosa.effects.time_stretch(x, rate=rate), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 — degrade to the numpy path, never fail
        logger.warning("%s time-stretch failed (%s); using numpy WSOLA", choice, exc)
    return wsola(x, sample_rate, rate)


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Duration-preserving pitch shift built from stretch + resample (numpy path)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or abs(semitones) < 1e-3:
        return x
    factor = 2.0 ** (semitones / 12.0)
    # Lengthen by `factor`, then play back `factor` faster: same length, pitch up.
    stretched = wsola(x, sample_rate, 1.0 / factor)
    src = np.arange(stretched.size, dtype=np.float64)
    dst = np.linspace(0, stretched.size - 1, x.size)
    return np.interp(dst, src, stretched).astype(np.float32)
