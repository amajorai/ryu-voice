<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./icon-dark.png" />
    <img src="./icon-light.png" alt="Voice" width="144" />
  </picture>
</p>

<div align="center">

# Voice

</div>

The universal TTS sidecar: a self-contained Python HTTP front over several text-to-speech engines (Kokoro, Kitten, Pocket), part of Ryu's speech data path.

> **The public home of `ryu-voice`.** Source, builds, and releases live here —
> binaries for every platform are attached to each release.
>
> This tree is generated from the Ryu monorepo, so commits pushed here
> directly are replaced on the next sync. **Pull requests are welcome** —
> open them here and they are ported into the monorepo, then flow back out.
> Ryu as a whole: https://github.com/amajorai/ryu

## Install

**App:** [Install](ryu://apps/@ryu/voice) (opens the Ryu desktop app and asks you to confirm)

**CLI:**

```bash
ryu apps add @ryu/voice
```

## Source & build

The **source of record** for the universal Ryu TTS sidecar — a self-contained
Python HTTP front over several text-to-speech engines. Install its
dependencies (`pip install -r sidecar/requirements.txt`) and run
`python -m ryu_tts` from `sidecar/`; Core manages it as a sidecar in a
full Ryu install.

## License

Apache-2.0 — see [LICENSE](./LICENSE).

## Parts

- **`sidecar/` — the Ryu TTS sidecar (Python, out-of-process).** A thin FastAPI
  runtime (`ryu-tts-sidecar`) that fronts many TTS engines behind one contract
  (`GET /health`, `GET /engines`, `POST /generate` → `audio/wav`, `POST /unload`).
  Fenced out of the bun/turbo workspace; runs on its own Python toolchain. Core
  owns lifecycle, model downloads, the catalog, and the `?engine=` selector on
  `/api/voice/speak`; this process is a pure inference runtime (the same way Core
  manages whisper.cpp's `whisper-server` and sd.cpp's `sd-server`). See
  `sidecar/README.md`.
- **Manifest** — the Core fixture `apps-store/voice/manifest.json`
  (id `@ryu/voice`). `runnables: []`, no grants: STT/TTS ride Core's existing
  `/api/voice/*` endpoints; this app just declares the capability boundary.

audio.cpp is a separate Core-managed native runtime, not a replacement for this
Python sidecar. Selecting `audiocpp` routes through its loopback OpenAI-compatible
server; the existing `ryutts` process remains available for its own engine registry.

## Engine registry (swap seam)

The HTTP layer never grows a per-engine branch. Adding an engine is one
`EngineConfig` row in `ryu_tts/registry.py` plus one
`ryu_tts/backends/<module>.py` implementing the `TtsBackend` protocol
(`load`/`generate`/`unload`/`is_loaded`). Seeded: **`kokoro`** (Kokoro 82M — the
Ryu default, CPU-only ONNX; Core auto-downloads weights + voice pack during
onboarding and injects them via `RYU_KOKORO_MODEL`/`RYU_KOKORO_VOICES`),
`kitten` (KittenTTS), and `pocket` (Kyutai Pocket TTS, voice cloning via
`reference_audio`). Heavy inference deps are optional extras and imported lazily
inside backend methods, so a missing dep degrades only that one engine.

## Run

`bun run dev:tts`, or from `sidecar/`: `pip install -e ".[kokoro]"` then
`python -m ryu_tts` (serves `127.0.0.1:8085`, `RYU_TTS_PORT` to override).
