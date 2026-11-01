"""Output container encoding for `/generate` (`format` field).

- `wav`: stdlib 16-bit PCM (`audio.to_wav_bytes`), always available.
- `flac`, `ogg` (Vorbis): `soundfile` (libsndfile) when importable, else ffmpeg.
- `mp3`: an `ffmpeg` subprocess (`RYU_FFMPEG` overrides the PATH lookup).

A format whose encoder is missing raises `EncoderUnavailable`; the server maps
it to HTTP 415 with an install hint rather than silently returning WAV.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess

import numpy as np

from .audio import to_wav_bytes

FORMATS = ("wav", "mp3", "ogg", "flac")
MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


class EncoderUnavailable(RuntimeError):
    def __init__(self, fmt: str, hint: str) -> None:
        super().__init__(f"no encoder available for '{fmt}'")
        self.fmt = fmt
        self.hint = hint


def ffmpeg_path() -> str | None:
    override = (os.environ.get("RYU_FFMPEG") or "").strip()
    if override:
        return override if os.path.isfile(override) else None
    return shutil.which("ffmpeg")


def _with_soundfile(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes | None:
    try:
        import soundfile
    except (ImportError, OSError):
        return None
    buffer = io.BytesIO()
    if fmt == "flac":
        soundfile.write(buffer, samples, int(sample_rate), format="FLAC", subtype="PCM_16")
    else:
        soundfile.write(buffer, samples, int(sample_rate), format="OGG", subtype="VORBIS")
    return buffer.getvalue()


_FFMPEG_ARGS = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
    "ogg": ["-f", "ogg", "-codec:a", "libvorbis", "-q:a", "5"],
    "flac": ["-f", "flac", "-codec:a", "flac"],
}


def _with_ffmpeg(wav: bytes, fmt: str) -> bytes:
    binary = ffmpeg_path()
    if binary is None:
        raise EncoderUnavailable(fmt, "Install ffmpeg (or set RYU_FFMPEG) to enable this output format.")
    command = [binary, "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0", *_FFMPEG_ARGS[fmt], "pipe:1"]
    try:
        completed = subprocess.run(command, input=wav, capture_output=True, check=False, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EncoderUnavailable(fmt, f"ffmpeg could not run: {exc}") from exc
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", "replace").strip()[-300:]
        raise EncoderUnavailable(fmt, f"ffmpeg failed to encode {fmt}: {detail}")
    return completed.stdout


def encode(samples: np.ndarray, sample_rate: int, fmt: str = "wav") -> tuple[bytes, str]:
    """Encode float mono samples. Returns `(bytes, mime_type)`."""
    if fmt not in MIME_TYPES:
        raise ValueError(f"unsupported format '{fmt}'")
    wav = to_wav_bytes(samples, sample_rate)
    if fmt == "wav":
        return wav, MIME_TYPES["wav"]
    if fmt in ("flac", "ogg"):
        clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
        try:
            data = _with_soundfile(clipped, sample_rate, fmt)
        except Exception:  # noqa: BLE001 — libsndfile without Vorbis, etc.: try ffmpeg
            data = None
        if data:
            return data, MIME_TYPES[fmt]
    return _with_ffmpeg(wav, fmt), MIME_TYPES[fmt]
