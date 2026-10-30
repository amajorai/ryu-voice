# <img src="https://raw.githubusercontent.com/amajorai/ryu/main/.github/logo.png" width="50" align="middle" alt="" />&nbsp; Ryu TTS Sidecar

> A universal HTTP front over many text-to-speech engines. Part of [Ryu](../../../README.md).

[![License](https://shieldcn.dev/badge/License-Apache--2.0-73DC8C.svg?logo=apache&logoColor=white)](../../../README.md#repository-layout--licensing)
[![Stack](https://shieldcn.dev/badge/Python-FastAPI-3776AB.svg?logo=python&logoColor=white)](../../../README.md)

A thin, universal HTTP runtime that fronts many TTS engines behind one contract, the same pattern `jamiepine/voicebox` (MIT) uses, Ryu-shaped. Ryu Core owns lifecycle, model downloads, the catalog, and the `?engine=` selector on `/api/voice/speak`; this process is a pure inference runtime, the same way Core manages whisper.cpp's `whisper-server` and stable-diffusion.cpp's `sd-server`. Fenced out of the bun/turbo workspace; runs on its own Python toolchain.

**Tier:** OSS, Apache-2.0

## Run

```bash
cd apps-store/voice/sidecar
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[kokoro]"                       # base server + Kokoro 82M (the Ryu default)
python -m ryu_tts                                 # serves 127.0.0.1:8085 (RYU_TTS_PORT to override)
```

Or from the repo root: `bun run dev:tts`.

## What it provides

- **The universal contract:** `GET /health`, `GET /engines`, `GET /effects`, `POST /engines/install`, `POST /generate`, `POST /unload`. Engine installation accepts only an id from the fixed registry; package names and commands are not caller-controlled. Torch engines use the CPU-only PyTorch wheel index on Linux (Pocket always; Chatterbox and Qwen3-TTS unless an NVIDIA GPU is present).
- **Swappable engine registry** (`ryu_tts/registry.py`): adding an engine is one `EngineConfig` row plus one `ryu_tts/backends/<module>.py` implementing the `TtsBackend` protocol (`load`/`generate`/`unload`/`is_loaded`). The server has no per-engine branches; it passes a backend only the `generate` keywords its signature declares, so older backends keep working as the contract grows.
- **Engines:**

  | id | Engine | Notable | Licence |
  | --- | --- | --- | --- |
  | `kokoro` | Kokoro 82M via `kokoro-onnx` (the Ryu **default**; Core downloads and injects `RYU_KOKORO_MODEL`/`RYU_KOKORO_VOICES`) | CPU, multi-voice, native speed | Apache-2.0 |
  | `kitten` | KittenTTS | CPU-only ONNX, native speed | Apache-2.0 |
  | `pocket` | Kyutai Pocket TTS | CPU, cloning via `reference_audio`, time-stretched speed | MIT code, CC-BY-4.0 weights |
  | `chatterbox` | Resemble AI Chatterbox Multilingual | cloning, 23 languages | MIT |
  | `chatterbox_turbo` | Chatterbox Turbo | English cloning, inline `[laugh]`-style tags | MIT |
  | `qwen` | Qwen3-TTS Base (0.6B/1.7B) | cloning that uses `reference_text` | Apache-2.0 |
  | `qwen_custom_voice` | Qwen3-TTS CustomVoice | 9 preset speakers + `instruct` | Apache-2.0 |
  | `qwen_voice_design` | Qwen3-TTS VoiceDesign (1.7B) | `instruct` describes a new voice | Apache-2.0 |

  `/engines` rows carry `supports_instruct`, `supports_design`, `supports_seed`, `native_speed`, `paralinguistic_tags`, `license`, and `commercial_use`. Install extras: `pip install -e ".[chatterbox]"` or `".[qwen]"`. The two torch families pin incompatible `transformers` versions (chatterbox-tts 0.1.7: 5.2.0; qwen-tts 0.1.1: 4.57.3), so install one family per environment; `POST /engines/install` answers 409 instead of installing the second family over the first. Torch engines detect their device: CUDA, then Apple MPS, then CPU (Chatterbox stays on CPU on Apple Silicon). The Qwen checkpoint size is the first cached variant (default first), else the default variant. CustomVoice defaults to 1.7B because qwen-tts drops `instruct` on 0.6B; `/engines` reports `supports_instruct: false` while 0.6B is the active checkpoint, and an instruction sent to it fails instead of being ignored.
- **`POST /generate` parameters** (all optional except `text`/`engine`): `voice`, `language`, `speed` (0.25-4; time-stretched with pitch preserved when the engine has no native speed), `reference_audio`, `reference_text` (≤1000), `seed` (0..2^32-1; the model is loaded first, then Python, NumPy, and torch are seeded, and every seeded request or seed-capable engine holds one process-wide RNG lock until its last chunk, so takes reproduce), `instruct` (≤500; ignored by engines without `supports_instruct`), `effects` (≤12 of `gain`, `highpass`, `lowpass`, `compressor`, `reverb`, `echo`, `pitch_shift`, each `{type, params}`; params are clamped to the `/effects` ranges; an unknown param name is a 422), `normalize` (`off`, `peak` = -1 dBFS, `podcast` = -16 LUFS, `acx` = -20 dB RMS with a -3 dBFS ceiling), and `format` (`wav`, `mp3`, `ogg`, `flac`). The response `Content-Type` matches the format. A missing encoder returns 415 with a hint.
- **One pipeline for every engine:** the script is planned first. `[pause]`, `[pause 500ms]`, and `[pause 1.5s]` become silence. Other `[tags]` are kept only when the engine lists them in `paralinguistic_tags`. Text is split at sentence boundaries into chunks of at most 400 characters (fixed in code), handling abbreviations, decimals, and CJK punctuation. Each chunk's leading and trailing silence is trimmed. A chunk longer than 0.25 s per character (4 s minimum) is cut with a fade and reported in `x-ryu-tts-runaway-capped`. Chunks are joined with equal-power crossfades before the effects chain, normalization, and encoding run.
- **DSP dependencies:** effects, the limiter, and the WSOLA time-stretch fallback are numpy/scipy only. `pyloudnorm` (`effects` extra) adds true LUFS metering; without it a gated-RMS approximation is used. `soundfile` encodes FLAC/OGG in-process; otherwise `ffmpeg` is used, and MP3 always uses `ffmpeg` (`RYU_FFMPEG` overrides the PATH lookup). Spotify's `pedalboard` (GPL-3.0) is opt-in through the `pedalboard` extra and is never bundled. When installed, it handles reverb, compressor, pitch shift, and time-stretch; leave the extra out to keep the numpy-only path. `GET /effects` lists every effect's parameter ranges and the built-in presets (broadcast, podcast, warm, bright, radio, telephone, robotic, deep, echo, cathedral).
- **Lazy heavy deps:** inference libraries are imported inside backend methods, so a missing dependency disables only that engine.

## Shipping

Core provisions this sidecar from a pinned, hash-checked wheel published by the `amajorai/ryu-voice` satellite release, not from this checkout. The request fields, DSP pipeline, encoders, and Chatterbox/Qwen3-TTS engines above ship to provisioned nodes with the next satellite wheel release, when Core's pinned release tag, wheel file name, and sha256 are bumped together. Older wheels ignore the new request fields (pydantic drops unknown keys) and return WAV. An editable install (`pip install -e .` or `bun run dev:tts`) runs this code directly.

## Test

```bash
.venv/bin/python -m pytest -q tests test_audio.py
```

The suite runs the full `/generate` pipeline through FastAPI's `TestClient` with a stub engine. Chatterbox and Qwen backends are tested against stubbed modules, so no weights are downloaded.

## License

Apache-2.0. See [LICENSE](../../../README.md#repository-layout--licensing). © 2026 A Major Pte. Ltd.
