"""Ryu TTS sidecar — a thin, universal HTTP front over many TTS engines.

One normalized contract for every engine:

    GET  /health             -> { ok, engines: [...ids that can run...] }
    GET  /engines            -> [ EngineInfo ]   (catalog Core mirrors for the picker)
    GET  /effects            -> effect schema, param ranges, built-in presets
    POST /engines/install    -> install one fixed-registry engine dependency set
    POST /generate           -> audio/wav|mpeg|ogg|flac (body: GenerateRequest)
    POST /unload             -> { unloaded }      (free a model's memory)

`/generate` runs one pipeline for every engine: plan the script (pause tags,
tag filtering, sentence chunks of ~400 chars) -> synthesize each chunk under a
per-engine lock (plus the global RNG lock for stochastic engines) after loading the
model and applying the request seed -> trim + runaway-cap each chunk ->
time-stretch when the engine lacks native speed -> crossfade-join with pause
silences -> effects chain -> loudness normalization -> encode.

Core owns lifecycle, routing, downloads, and the `?engine=` selector on
`/api/voice/speak`; this process is a pure inference runtime. Adding an engine is
one registry row + one backend file — no change here.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import importlib
import inspect
import logging
import os
import subprocess
import sys
import threading
from typing import Any

import anyio
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import effects as fx
from . import registry
from .audio import cap_runaway, speech_length, crossfade_concat, fade, resample, silence, to_float_mono, trim_silence
from .chunking import DEFAULT_MAX_CHARS, plan_script
from .device import requested_device, seed_everything
from .encode import EncoderUnavailable, encode
from .engine_installer import build_engine_install_command
from .request_models import GenerateRequest
from .timestretch import time_stretch

app = FastAPI(title="Ryu TTS Sidecar", version="0.1.0")
logger = logging.getLogger(__name__)

# Shared-secret bearer Core stamps on every proxied hop and injects at spawn
# (`RYU_EXT_TOKEN`). FAIL-CLOSED for every route except /health: no token
# configured => reject all. Closes unauthenticated loopback/CSRF access to the
# synthesis + model-install endpoints.
_EXPECTED_TOKEN = (os.environ.get("RYU_EXT_TOKEN") or "").strip()
_ENGINE_INSTALL_LOCK = asyncio.Lock()


@app.middleware("http")
async def _require_ext_token(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    header = request.headers.get("authorization", "")
    presented = header[len("Bearer ") :].strip() if header.startswith("Bearer ") else ""
    if not (_EXPECTED_TOKEN and hmac.compare_digest(presented, _EXPECTED_TOKEN)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 echoes the offending input, which cannot be
    JSON-encoded when it is NaN/Infinity; report location + message only."""
    detail = [
        {"loc": list(err.get("loc", ())), "msg": str(err.get("msg", "")), "type": str(err.get("type", ""))}
        for err in exc.errors()
    ]
    return JSONResponse({"error": "invalid request", "detail": detail}, status_code=422)


# Sentence-chunk ceiling for one synthesis call, fixed in code.
_MAX_CHUNK_CHARS = DEFAULT_MAX_CHARS
_CROSSFADE_MS = 40.0
_EDGE_FADE_MS = 5.0

# Inference is not re-entrant per model; serialize generation per engine.
_ENGINE_LOCKS: dict[str, threading.Lock] = {}
_ENGINE_LOCKS_GUARD = threading.Lock()
# The torch/numpy/random RNGs are process-global, so a seeded take is only
# reproducible if no other sampler draws from them meanwhile. Every stochastic
# (seed-capable) engine and every seeded request holds this one lock from seeding
# to the last chunk; deterministic engines without a seed still run concurrently.
# Order is always engine lock -> RNG lock, so the two never deadlock.
_RNG_LOCK = threading.Lock()


def _engine_lock(engine_id: str) -> threading.Lock:
    with _ENGINE_LOCKS_GUARD:
        return _ENGINE_LOCKS.setdefault(engine_id, threading.Lock())


def _engine_info(cfg: registry.EngineConfig) -> dict:
    return {
        "id": cfg.id,
        "display_name": cfg.display_name,
        "description": cfg.description,
        "voices": cfg.voices,
        "default_voice": cfg.default_voice,
        "sample_rate": cfg.sample_rate,
        "supports_cloning": cfg.supports_cloning,
        "languages": cfg.languages,
        "size_mb": cfg.size_mb,
        "pip_packages": cfg.pip_packages,
        "installed": registry.is_installed(cfg),
        "loaded": cfg.id in registry.loaded_ids(),
        "supports_instruct": registry.supports_instruct(cfg),
        "supports_design": cfg.supports_design,
        "supports_seed": cfg.supports_seed,
        "native_speed": cfg.native_speed,
        "paralinguistic_tags": cfg.paralinguistic_tags,
        "license": cfg.license,
        "commercial_use": cfg.commercial_use,
    }


@app.get("/health")
def health() -> dict:
    runnable = [c.id for c in registry.ENGINES if registry.is_installed(c)]
    return {
        "ok": True,
        "engines": runnable,
        "total": len(registry.ENGINES),
        "device": requested_device(),
    }


@app.get("/engines")
def engines() -> list[dict]:
    return [_engine_info(c) for c in registry.ENGINES]


@app.get("/effects")
def effects() -> dict:
    """Effect types with their param ranges, plus the built-in presets."""
    return fx.effects_schema()


@app.get("/models")
def models() -> list[dict]:
    """The curated, installable TTS model catalog (voicebox-style) — every
    engine's known-good model variants, each bound to its engine + cache state.
    Core mirrors this for the desktop's curated TTS lane."""
    return registry.all_models()


class InstallRequest(BaseModel):
    engine: str = Field(..., description="Engine id the model belongs to.")
    model_name: str = Field(..., description="Curated model_name from /models.")


class EngineInstallRequest(BaseModel):
    engine: str = Field(..., description="Engine id from /engines.")


@app.post("/engines/install", response_model=None)
async def install_engine(req: EngineInstallRequest) -> dict | JSONResponse:
    cfg = registry.get_config(req.engine)
    if cfg is None:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=404)
    if registry.is_installed(cfg):
        return {"engine": cfg.id, "installed": True, "already_installed": True}
    if not cfg.pip_packages:
        return JSONResponse(
            {"error": f"engine '{cfg.id}' has no managed install command"},
            status_code=400,
        )
    clashing = [
        other.display_name
        for other in registry.ENGINES
        if other.import_name in cfg.conflicts_with and registry.is_installed(other)
    ]
    if clashing:
        return JSONResponse(
            {
                "error": f"'{cfg.display_name}' cannot be installed alongside {', '.join(sorted(set(clashing)))}",
                "hint": "These engines pin incompatible transformers versions. Uninstall the other "
                "engine's package from the Ryu TTS environment first, or run it in a separate sidecar.",
            },
            status_code=409,
        )
    if sys.prefix == sys.base_prefix and os.environ.get("RYU_TTS_ALLOW_SYSTEM_INSTALL") != "1":
        return JSONResponse(
            {
                "error": "managed engine installation requires the Ryu TTS virtual environment",
                "hint": "Set up the sidecar virtual environment or explicitly allow a system install.",
            },
            status_code=409,
        )

    async with _ENGINE_INSTALL_LOCK:
        if registry.is_installed(cfg):
            return {"engine": cfg.id, "installed": True, "already_installed": True}
        try:
            completed = await anyio.to_thread.run_sync(
                lambda: subprocess.run(
                    build_engine_install_command(cfg),
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=15 * 60,
                )
            )
        except subprocess.TimeoutExpired:
            return JSONResponse(
                {"error": f"installing '{cfg.id}' timed out after 15 minutes"},
                status_code=504,
            )
        except OSError as exc:
            return JSONResponse(
                {"error": f"could not start the installer for '{cfg.id}': {exc}"},
                status_code=502,
            )

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-3000:]
            logger.error("TTS engine install failed for %s: %s", cfg.id, detail)
            return JSONResponse(
                {
                    "error": f"installing '{cfg.id}' failed",
                    "hint": "Check the local Ryu TTS sidecar logs for package-manager details.",
                },
                status_code=502,
            )

        importlib.invalidate_caches()
        installed = registry.is_installed(cfg)
        if not installed:
            return JSONResponse(
                {"error": f"installer completed but '{cfg.id}' is still unavailable"},
                status_code=503,
            )
        return {"engine": cfg.id, "installed": True, "already_installed": False}


@app.post("/models/install")
async def models_install(req: InstallRequest) -> dict:
    """Download a curated model into the (Core-managed) HF cache via
    `huggingface_hub.snapshot_download`. Idempotent: a cache hit returns fast.
    Core wraps this in a DownloadCenter entry for progress visibility."""
    variant = registry.find_model(req.engine, req.model_name)
    if variant is None:
        return JSONResponse(
            {"error": f"unknown model '{req.model_name}' for engine '{req.engine}'"},
            status_code=404,
        )
    try:
        from huggingface_hub import snapshot_download

        path = await anyio.to_thread.run_sync(
            lambda: snapshot_download(repo_id=variant.hf_repo_id)
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"error": f"downloading {variant.hf_repo_id} failed: {exc}"},
            status_code=502,
        )
    return {
        "installed": True,
        "engine": req.engine,
        "model_name": req.model_name,
        "hf_repo_id": variant.hf_repo_id,
        "path": str(path),
    }


@app.post("/unload")
def unload(body: dict) -> dict:
    engine_id = str(body.get("engine", ""))
    cfg = registry.get_config(engine_id)
    if cfg is None:
        return JSONResponse({"error": f"unknown engine '{engine_id}'"}, status_code=404)
    backend = registry.get_backend(engine_id)
    was_loaded = backend.is_loaded()
    backend.unload()
    return {"unloaded": was_loaded, "engine": engine_id}


def _accepted_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keywords a backend's `generate` does not declare, so a backend
    written before a newer extra (e.g. `instruct`) keeps working."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class _SynthesisResult:
    def __init__(self, audio: np.ndarray, sample_rate: int, chunks: int, capped: int) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.chunks = chunks
        self.capped = capped


def _assemble(pieces: list[tuple[str, Any]], sample_rate: int) -> np.ndarray:
    """Crossfade consecutive speech chunks; butt-join pauses with short fades."""
    out: list[np.ndarray] = []
    run: list[np.ndarray] = []

    def _flush() -> None:
        if run:
            joined = crossfade_concat(run, sample_rate, _CROSSFADE_MS)
            out.append(fade(joined, sample_rate, in_ms=_EDGE_FADE_MS, out_ms=_EDGE_FADE_MS))
            run.clear()

    for kind, value in pieces:
        if kind == "speech":
            run.append(value)
        else:
            _flush()
            out.append(silence(value, sample_rate))
    _flush()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _synthesize(
    cfg: registry.EngineConfig, backend: registry.TtsBackend, req: GenerateRequest, text: str
) -> _SynthesisResult:
    plan = plan_script(text, allowed_tags=cfg.paralinguistic_tags, max_chars=_MAX_CHUNK_CHARS)
    if not any(seg.kind == "speech" for seg in plan):
        raise ValueError("nothing to speak after removing tags")
    stretch = not cfg.native_speed and abs(req.speed - 1.0) > 1e-3
    kwargs = _accepted_kwargs(
        backend.generate,
        {
            "voice": req.voice,
            "speed": req.speed if cfg.native_speed else 1.0,
            "language": req.language,
            "reference_audio": req.reference_audio,
            "seed": req.seed,
            "instruct": req.instruct if cfg.supports_instruct else None,
            "reference_text": req.reference_text,
        },
    )
    pieces: list[tuple[str, Any]] = []
    sample_rate: int | None = None
    capped = 0
    chunks = 0
    shares_rng = cfg.supports_seed or req.seed is not None
    with _engine_lock(cfg.id):
        # Load first: lazy loads import torch and draw from the RNGs while
        # building the model, which would otherwise happen after seeding and make
        # the first take after start/unload differ from every later one.
        load = getattr(backend, "load", None)
        if callable(load):
            load()
        with _RNG_LOCK if shares_rng else contextlib.nullcontext():
            seed_everything(req.seed)
            for seg in plan:
                if seg.kind == "pause":
                    pieces.append(("pause", seg.pause_ms))
                    continue
                samples, rate = backend.generate(seg.text, **kwargs)
                audio = to_float_mono(samples)
                if sample_rate is None:
                    sample_rate = int(rate)
                elif int(rate) != sample_rate:
                    audio = resample(audio, int(rate), sample_rate)
                native_rate = req.speed if cfg.native_speed else 1.0
                audio, was_capped = cap_runaway(
                    audio, sample_rate, speech_length(seg.text), speed=native_rate
                )
                capped += int(was_capped)
                audio = trim_silence(audio, sample_rate)
                if stretch:
                    audio = time_stretch(audio, sample_rate, req.speed)
                pieces.append(("speech", audio))
                chunks += 1
    assert sample_rate is not None
    audio = _assemble(pieces, sample_rate)
    if req.effects:
        audio = fx.apply_chain(audio, sample_rate, [e.model_dump() for e in req.effects])
    audio = fx.normalize(audio, sample_rate, req.normalize or "off")
    return _SynthesisResult(audio, sample_rate, chunks, capped)


@app.post("/generate")
async def generate(req: GenerateRequest) -> Response:
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "missing `text`"}, status_code=400)

    cfg = registry.get_config(req.engine)
    if cfg is None:
        available = [c.id for c in registry.ENGINES]
        return JSONResponse(
            {"error": f"unknown engine '{req.engine}'", "available": available},
            status_code=404,
        )

    if not registry.is_installed(cfg):
        hint = " ".join(cfg.pip_packages) or cfg.backend_module
        return JSONResponse(
            {
                "error": f"engine '{cfg.id}' is not installed",
                "hint": f"pip install {hint}",
            },
            status_code=503,
        )

    try:
        backend = registry.get_backend(req.engine)
        # Inference is blocking (CPU/GPU); run it off the event loop.
        result = await anyio.to_thread.run_sync(lambda: _synthesize(cfg, backend, req, text))
    except Exception as exc:  # noqa: BLE001 — surface any engine error to the caller
        return JSONResponse(
            {"error": f"synthesis failed in '{req.engine}': {exc}"}, status_code=502
        )

    if result.audio.size == 0:
        return JSONResponse({"error": "engine produced empty audio"}, status_code=502)
    fmt = req.format or "wav"
    try:
        body, media_type = await anyio.to_thread.run_sync(
            lambda: encode(result.audio, result.sample_rate, fmt)
        )
    except EncoderUnavailable as exc:
        return JSONResponse({"error": str(exc), "hint": exc.hint}, status_code=415)
    headers = {
        "x-ryu-tts-engine": cfg.id,
        "x-ryu-tts-chunks": str(result.chunks),
        "x-ryu-tts-sample-rate": str(result.sample_rate),
    }
    if result.capped:
        headers["x-ryu-tts-runaway-capped"] = str(result.capped)
    return Response(content=body, media_type=media_type, headers=headers)
