"""Audio helpers — WAV encoding plus the float-domain take assembly steps.

`to_wav_bytes` stays dependency-free (stdlib `wave` only). The assembly helpers
(crossfade, silence, trim, runaway cap) are numpy-only; `resample` imports scipy
lazily. Other container formats live in `encode.py`.
"""

from __future__ import annotations

import io
import wave

import numpy as np


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a 1-D (mono) sample array as 16-bit PCM WAV.

    Accepts float arrays in [-1, 1] or integer arrays. Multi-channel input is
    down-mixed to mono by averaging channels.
    """
    arr = np.asarray(samples)

    # Down-mix anything 2-D to mono (average across the channel axis).
    if arr.ndim > 1:
        # Heuristic: the smaller axis is channels.
        channel_axis = int(np.argmin(arr.shape))
        arr = arr.mean(axis=channel_axis)
    arr = arr.reshape(-1)

    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2")
    elif arr.dtype == np.int16:
        pcm = arr.astype("<i2")
    else:
        # Integer of another width — scale down conservatively to int16.
        pcm = np.clip(arr, -32768, 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Float-domain helpers used to assemble multi-chunk takes. Everything below
# works on 1-D float32 mono arrays in [-1, 1].
# ---------------------------------------------------------------------------


def to_float_mono(samples: np.ndarray) -> np.ndarray:
    """Normalise a backend's output to a 1-D float32 mono array in [-1, 1]."""
    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.mean(axis=int(np.argmin(arr.shape)))
    arr = arr.reshape(-1)
    if arr.dtype.kind in "iu":
        scale = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / scale
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def silence(ms: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(int(round(sample_rate * ms / 1000.0)), 0), dtype=np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resample (scipy polyphase); identity when rates match."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(src_rate), int(dst_rate))
    out = resample_poly(audio, int(dst_rate) // g, int(src_rate) // g)
    return out.astype(np.float32)


def fade(audio: np.ndarray, sample_rate: int, *, in_ms: float = 0.0, out_ms: float = 0.0) -> np.ndarray:
    """Apply short linear fade-in/out ramps (click-free chunk edges)."""
    out = audio.astype(np.float32, copy=True)
    n_in = min(int(sample_rate * in_ms / 1000.0), out.size)
    n_out = min(int(sample_rate * out_ms / 1000.0), out.size)
    if n_in > 0:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 0:
        out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return out


def crossfade_concat(chunks: list[np.ndarray], sample_rate: int, crossfade_ms: float = 50.0) -> np.ndarray:
    """Join chunks with an equal-power (cos/sin) crossfade between neighbours.

    The overlap is capped at half the shorter neighbour so tiny chunks are never
    swallowed. Equal-power curves keep perceived loudness flat across the seam.
    """
    parts = [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks if np.asarray(c).size]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    out = parts[0]
    wanted = int(sample_rate * max(crossfade_ms, 0.0) / 1000.0)
    for nxt in parts[1:]:
        overlap = min(wanted, out.size // 2, nxt.size // 2)
        if overlap <= 0:
            out = np.concatenate([out, nxt])
            continue
        theta = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
        mixed = out[-overlap:] * np.cos(theta) + nxt[:overlap] * np.sin(theta)
        out = np.concatenate([out[:-overlap], mixed, nxt[overlap:]])
    return out.astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -45.0,
    pad_ms: float = 40.0,
    frame_ms: float = 10.0,
) -> np.ndarray:
    """Trim leading/trailing frames quieter than `threshold_db` (relative to full
    scale), keeping `pad_ms` of breathing room. All-silent input is returned as-is."""
    if audio.size == 0:
        return audio
    frame = max(int(sample_rate * frame_ms / 1000.0), 1)
    n_frames = int(np.ceil(audio.size / frame))
    padded = np.pad(np.abs(audio), (0, n_frames * frame - audio.size))
    peaks = padded.reshape(n_frames, frame).max(axis=1)
    loud = np.flatnonzero(peaks > 10 ** (threshold_db / 20.0))
    if loud.size == 0:
        return audio
    pad = int(sample_rate * pad_ms / 1000.0)
    start = max(int(loud[0]) * frame - pad, 0)
    end = min((int(loud[-1]) + 1) * frame + pad, audio.size)
    return audio[start:end]


# Generous ceiling: natural speech runs ~0.06-0.1 s/char, so 0.25 s/char only
# trips on a decoder that missed its end-of-sequence token and kept going.
RUNAWAY_SEC_PER_CHAR = 0.25
RUNAWAY_FLOOR_SEC = 4.0
# One CJK ideograph, kana, or Hangul syllable is a whole syllable (~0.2-0.3 s
# spoken, slower for a cloned voice), so it counts as this many Latin chars.
DENSE_SCRIPT_WEIGHT = 3
_DENSE_RANGES = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3100, 0x312F),  # Bopomofo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Halfwidth Katakana
    (0x20000, 0x2FA1F),  # CJK Extensions B+ and supplements
)


def _is_dense(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _DENSE_RANGES)


def speech_length(text: str) -> int:
    """Latin-equivalent length of `text` for the runaway budget: syllabic CJK
    characters count `DENSE_SCRIPT_WEIGHT` times so legitimate zh/ja/ko speech
    is never cut."""
    return sum(DENSE_SCRIPT_WEIGHT if _is_dense(c) else 1 for c in text)


def max_duration_sec(text_chars: int, speed: float = 1.0) -> float:
    """Longest plausible take for `text_chars` characters at `speed`."""
    return max(RUNAWAY_FLOOR_SEC, text_chars * RUNAWAY_SEC_PER_CHAR) / max(speed, 0.25)


def cap_runaway(
    audio: np.ndarray,
    sample_rate: int,
    text_chars: int,
    *,
    speed: float = 1.0,
    fade_ms: float = 250.0,
) -> tuple[np.ndarray, bool]:
    """Cut a take that is implausibly long for its text, ending in a fade-out.
    Returns `(audio, was_capped)`."""
    limit = int(max_duration_sec(text_chars, speed) * sample_rate)
    if audio.size <= limit:
        return audio, False
    return fade(audio[:limit], sample_rate, out_ms=fade_ms), True
