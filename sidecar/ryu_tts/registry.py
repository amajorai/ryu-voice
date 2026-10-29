"""Universal TTS engine registry — the "voicebox way", Ryu-shaped.

The whole point of this module: adding a new TTS engine is **one new backend
file + one registry row**. The HTTP layer (`server.py`) never grows a per-engine
branch — it looks an engine up here, asks the factory for its backend, and calls
the same `TtsBackend` protocol on every engine.

This mirrors `jamiepine/voicebox`'s `backends/__init__.py` design (MIT licensed):
a declarative `EngineConfig` registry + a lazy factory. The difference is that in
Ryu *Core* owns lifecycle/routing/downloads — this process is a pure inference
runtime, the same way Core treats whisper.cpp's `whisper-server` and
stable-diffusion.cpp's `sd-server` as managed external runtimes.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelVariant:
    """One curated, installable model an engine can serve — voicebox's
    `ModelConfig`. The `hf_repo_id` is the Hugging Face repo; Core downloads it
    into the engine's HF cache (under a Core-managed `HF_HOME`) so the engine
    loads it as a cache hit. This is the *known-good* set, distinct from the raw
    HF `pipeline_tag=text-to-speech` firehose (which Core surfaces as discovery)."""

    model_name: str  # stable id, e.g. "kitten-tts-mini-0.8"
    display_name: str
    hf_repo_id: str
    size_mb: int = 0
    languages: list[str] = field(default_factory=lambda: ["en"])
    default: bool = False  # the engine's default model variant
    # False when this checkpoint ignores `instruct` even though its engine row
    # supports it (qwen-tts silently drops it for the 0.6B CustomVoice model).
    supports_instruct: bool = True


@dataclass(frozen=True)
class EngineConfig:
    """Declarative description of one TTS engine.

    Everything the API and Core's catalog need to know *about* an engine without
    importing its (heavy, optional) inference dependencies.
    """

    id: str  # stable identifier used by `?engine=` end to end (e.g. "kitten")
    display_name: str
    description: str
    backend_module: str  # dotted path under `ryu_tts.backends`, e.g. "kitten"
    backend_class: str  # class name implementing `TtsBackend`
    import_name: str  # top-level module proving the engine's deps exist (e.g. "kittentts")
    default_voice: str
    voices: list[str]
    sample_rate: int  # nominal output rate (an engine may override at generate time)
    supports_cloning: bool  # True if `reference_audio` (a wav path/url) is honoured
    languages: list[str] = field(default_factory=lambda: ["en"])
    size_mb: int = 0
    pip_packages: list[str] = field(default_factory=list)  # for the install hint
    # Curated installable model variants for this engine (the voicebox registry).
    models: list[ModelVariant] = field(default_factory=list)
    # Capability flags surfaced verbatim in `/engines` (all optional for clients).
    supports_instruct: bool = False  # honours `instruct` (delivery/style prompt)
    supports_design: bool = False  # `instruct` describes a brand-new voice
    supports_seed: bool = False  # output varies with `seed` (stochastic sampler)
    native_speed: bool = True  # False => the server time-stretches for `speed`
    paralinguistic_tags: list[str] = field(default_factory=list)  # kept `[tags]`
    license: str = ""  # SPDX-ish licence of the code + weights actually run
    commercial_use: bool = True
    # Import names of engine families whose pinned deps cannot share this venv
    # with this one; `/engines/install` refuses instead of breaking the other.
    conflicts_with: list[str] = field(default_factory=list)


@runtime_checkable
class TtsBackend(Protocol):
    """The single contract every engine implements. Synchronous on purpose —
    inference is CPU/GPU-bound and `server.py` runs it in a threadpool."""

    config: EngineConfig

    def load(self) -> None:
        """Load weights into memory (idempotent; called before first generate)."""
        ...

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str = "en",
        reference_audio: str | None = None,
        seed: int | None = None,
        instruct: str | None = None,
        reference_text: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech. Returns ``(samples, sample_rate)`` where ``samples``
        is a 1-D float32 (or int16) mono array. `reference_audio` is a local wav
        path or URL used only by cloning-capable engines; `reference_text` is its
        transcript. `instruct` is a delivery prompt (or, for design engines, the
        voice description). The server drops any keyword a backend's signature
        does not declare, so older backends without the newer extras keep working.
        `speed` is only passed through when the engine has `native_speed`."""
        ...

    def unload(self) -> None:
        """Release model memory."""
        ...

    def is_loaded(self) -> bool:
        ...


# Language codes Chatterbox Multilingual accepts (`chatterbox.mtl_tts.SUPPORTED_LANGUAGES`).
CHATTERBOX_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
)
# Inline tags Chatterbox Turbo's tokenizer was trained on. List adapted from
# jamiepine/voicebox (MIT) `ParalinguisticInput.tsx`.
CHATTERBOX_TURBO_TAGS = (
    "laugh", "chuckle", "gasp", "cough", "sigh", "groan", "sniff", "shush",
    "clear throat", "angry", "crying", "dramatic", "fear", "happy", "narration",
    "sarcastic", "surprised", "whispering", "advertisement",
)
QWEN_LANGUAGES = ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it")
# Qwen3-TTS CustomVoice preset speaker ids (upstream model card).
QWEN_CUSTOM_VOICES = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee",
)


# ---------------------------------------------------------------------------
# The registry. Add an engine = append one EngineConfig row here + drop a
# `backends/<module>.py` implementing `TtsBackend`. Nothing else changes.
# ---------------------------------------------------------------------------

# chatterbox-tts 0.1.7 pins transformers==5.2.0 and qwen-tts 0.1.1 pins
# transformers==4.57.3 (both read from the wheels' metadata), so installing one
# family upgrades/downgrades the other into a broken state.
_TORCH_FAMILY_CONFLICT = {"chatterbox": ["qwen_tts"], "qwen_tts": ["chatterbox"]}

ENGINES: list[EngineConfig] = [
    # Kokoro 82M — the Ryu default TTS engine. An 82M open-weight model (Apache-2.0)
    # served via the CPU-friendly `kokoro-onnx` runtime (ONNX Runtime, no torch), so
    # it runs on most machines with no GPU — the "runs on most machines" ethos of the
    # bundled Gemma 4 chat default. Multi-voice + multi-lingual. Listed FIRST so it is
    # the natural head of the catalog; Core also pins it as the cross-surface default
    # engine id (`DEFAULT_TTS_ENGINE`). The model + voices files are auto-downloaded
    # by Core during onboarding (like the Gemma/nomic GGUFs) and their local paths are
    # injected via `RYU_KOKORO_MODEL` / `RYU_KOKORO_VOICES` at spawn.
    EngineConfig(
        id="kokoro",
        display_name="Kokoro 82M",
        description="82M open-weight TTS · ONNX · CPU-friendly · multi-voice, multilingual · Apache-2.0",
        backend_module="kokoro",
        backend_class="KokoroBackend",
        import_name="kokoro_onnx",
        default_voice="af_heart",
        voices=[
            # American English
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
            "am_onyx", "am_puck", "am_santa",
            # British English
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        ],
        sample_rate=24000,
        supports_cloning=False,
        languages=["en", "es", "fr", "hi", "it", "ja", "pt", "zh"],
        size_mb=310,
        pip_packages=["kokoro-onnx", "onnxruntime"],
        license="Apache-2.0",
        models=[
            ModelVariant(
                model_name="kokoro-82m-v1.0",
                display_name="Kokoro 82M (v1.0)",
                hf_repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
                size_mb=310,
                languages=["en", "es", "fr", "hi", "it", "ja", "pt", "zh"],
                default=True,
            ),
        ],
    ),
    EngineConfig(
        id="kitten",
        display_name="KittenTTS",
        description="Ultra-light (15-80M) ONNX TTS · CPU-only · fixed expressive voices",
        backend_module="kitten",
        backend_class="KittenBackend",
        import_name="kittentts",
        default_voice="Jasper",
        voices=["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"],
        sample_rate=24000,
        supports_cloning=False,
        languages=["en"],
        size_mb=80,
        pip_packages=["kittentts"],
        license="Apache-2.0",
        models=[
            ModelVariant(
                model_name="kitten-tts-mini-0.8",
                display_name="KittenTTS Mini (80M)",
                hf_repo_id="KittenML/kitten-tts-mini-0.8",
                size_mb=80,
                default=True,
            ),
            ModelVariant(
                model_name="kitten-tts-nano-0.1",
                display_name="KittenTTS Nano (15M)",
                hf_repo_id="KittenML/kitten-tts-nano-0.1",
                size_mb=25,
            ),
        ],
    ),
    EngineConfig(
        id="pocket",
        display_name="Pocket TTS (Kyutai)",
        description="100M CPU TTS · ~6x realtime · ~200ms first audio · voice cloning",
        backend_module="pocket",
        backend_class="PocketBackend",
        import_name="pocket_tts",
        default_voice="alba",
        voices=[
            "alba", "giovanni", "lola", "juergen", "rafael", "estelle", "anna",
            "azelma", "charles", "cosette", "eve", "fantine", "george", "jane",
            "jean", "javert", "marius", "mary", "michael", "paul", "vera",
        ],
        sample_rate=24000,
        supports_cloning=True,
        languages=["en"],
        size_mb=100,
        pip_packages=["pocket-tts"],
        # Verified from the pocket-tts 3.3.0 wheel (MIT) and the Hugging Face
        # model card (cc-by-4.0, attribution required). Preset voices carry their
        # own per-voice licences listed on kyutai/tts-voices.
        license="MIT (code) · CC-BY-4.0 (weights)",
        supports_seed=True,
        native_speed=False,
        models=[
            ModelVariant(
                model_name="pocket-tts",
                display_name="Pocket TTS (100M)",
                hf_repo_id="kyutai/pocket-tts-without-voice-cloning",
                size_mb=100,
                default=True,
            ),
        ],
    ),
    # Chatterbox (Resemble AI, MIT code + weights). Zero-shot cloning from a
    # reference wav in 23 languages; with no reference it speaks the bundled
    # default voice. Torch engine: runs on CUDA/MPS/CPU via `ryu_tts/device.py`.
    EngineConfig(
        id="chatterbox",
        display_name="Chatterbox Multilingual",
        description="500M zero-shot voice cloning · 23 languages · GPU recommended · MIT",
        backend_module="chatterbox",
        backend_class="ChatterboxBackend",
        import_name="chatterbox",
        conflicts_with=_TORCH_FAMILY_CONFLICT["chatterbox"],
        default_voice="default",
        voices=["default"],
        sample_rate=24000,
        supports_cloning=True,
        languages=list(CHATTERBOX_LANGUAGES),
        size_mb=3200,
        pip_packages=["chatterbox-tts"],
        license="MIT",
        supports_seed=True,
        native_speed=False,
        models=[
            ModelVariant(
                model_name="chatterbox-multilingual",
                display_name="Chatterbox Multilingual",
                hf_repo_id="ResembleAI/chatterbox",
                size_mb=3200,
                languages=list(CHATTERBOX_LANGUAGES),
                default=True,
            ),
        ],
    ),
    # Chatterbox Turbo: the faster English-only 350M variant whose tokenizer
    # understands inline paralinguistic tags (kept by the chunker, not stripped).
    EngineConfig(
        id="chatterbox_turbo",
        display_name="Chatterbox Turbo",
        description="350M English cloning · inline [laugh]/[sigh] tags · low latency · MIT",
        backend_module="chatterbox",
        backend_class="ChatterboxTurboBackend",
        import_name="chatterbox",
        conflicts_with=_TORCH_FAMILY_CONFLICT["chatterbox"],
        default_voice="default",
        voices=["default"],
        sample_rate=24000,
        supports_cloning=True,
        languages=["en"],
        size_mb=1500,
        pip_packages=["chatterbox-tts"],
        license="MIT",
        supports_seed=True,
        native_speed=False,
        paralinguistic_tags=list(CHATTERBOX_TURBO_TAGS),
        models=[
            ModelVariant(
                model_name="chatterbox-turbo",
                display_name="Chatterbox Turbo (350M)",
                hf_repo_id="ResembleAI/chatterbox-turbo",
                size_mb=1500,
                default=True,
            ),
        ],
    ),
    # Qwen3-TTS (Alibaba Qwen, Apache-2.0 code + weights), three checkpoints that
    # share one loader in `backends/qwen_tts.py`. Base clones from a reference
    # wav and uses its transcript (`reference_text`) for in-context cloning.
    EngineConfig(
        id="qwen",
        display_name="Qwen3-TTS (voice cloning)",
        description="Zero-shot cloning with reference transcript · 10 languages · GPU recommended · Apache-2.0",
        backend_module="qwen_tts",
        backend_class="QwenCloneBackend",
        import_name="qwen_tts",
        conflicts_with=_TORCH_FAMILY_CONFLICT["qwen_tts"],
        default_voice="reference",
        voices=["reference"],
        sample_rate=24000,
        supports_cloning=True,
        languages=list(QWEN_LANGUAGES),
        size_mb=1200,
        pip_packages=["qwen-tts"],
        license="Apache-2.0",
        supports_seed=True,
        native_speed=False,
        models=[
            ModelVariant(
                model_name="qwen-tts-0.6B",
                display_name="Qwen3-TTS Base 0.6B",
                hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                size_mb=1200,
                languages=list(QWEN_LANGUAGES),
                default=True,
            ),
            ModelVariant(
                model_name="qwen-tts-1.7B",
                display_name="Qwen3-TTS Base 1.7B",
                hf_repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                size_mb=3500,
                languages=list(QWEN_LANGUAGES),
            ),
        ],
    ),
    # CustomVoice: nine preset speakers steered by a natural-language `instruct`.
    EngineConfig(
        id="qwen_custom_voice",
        display_name="Qwen3-TTS CustomVoice",
        description="9 preset speakers with delivery instructions · 10 languages · Apache-2.0",
        backend_module="qwen_tts",
        backend_class="QwenCustomVoiceBackend",
        import_name="qwen_tts",
        conflicts_with=_TORCH_FAMILY_CONFLICT["qwen_tts"],
        default_voice="Ryan",
        voices=list(QWEN_CUSTOM_VOICES),
        sample_rate=24000,
        supports_cloning=False,
        languages=list(QWEN_LANGUAGES),
        size_mb=3500,
        pip_packages=["qwen-tts"],
        license="Apache-2.0",
        supports_instruct=True,
        supports_seed=True,
        native_speed=False,
        models=[
            # 1.7B is the default: qwen-tts 0.1.1 drops `instruct` for the 0.6B
            # CustomVoice checkpoint (`tts_model_size == "0b6"`), so only 1.7B
            # honours delivery instructions. `/engines` reports the active one.
            ModelVariant(
                model_name="qwen-custom-voice-1.7B",
                display_name="Qwen3-TTS CustomVoice 1.7B",
                hf_repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                size_mb=3500,
                languages=list(QWEN_LANGUAGES),
                default=True,
            ),
            ModelVariant(
                model_name="qwen-custom-voice-0.6B",
                display_name="Qwen3-TTS CustomVoice 0.6B (no instructions)",
                hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                size_mb=1200,
                languages=list(QWEN_LANGUAGES),
                supports_instruct=False,
            ),
        ],
    ),
    # VoiceDesign: `instruct` *is* the voice ("a warm, low-pitched narrator").
    # Upstream ships only a 1.7B checkpoint.
    EngineConfig(
        id="qwen_voice_design",
        display_name="Qwen3-TTS VoiceDesign",
        description="Design a new voice from a text description · 10 languages · Apache-2.0",
        backend_module="qwen_tts",
        backend_class="QwenVoiceDesignBackend",
        import_name="qwen_tts",
        conflicts_with=_TORCH_FAMILY_CONFLICT["qwen_tts"],
        default_voice="designed",
        voices=["designed"],
        sample_rate=24000,
        supports_cloning=False,
        languages=list(QWEN_LANGUAGES),
        size_mb=3500,
        pip_packages=["qwen-tts"],
        license="Apache-2.0",
        supports_instruct=True,
        supports_design=True,
        supports_seed=True,
        native_speed=False,
        models=[
            ModelVariant(
                model_name="qwen-voice-design-1.7B",
                display_name="Qwen3-TTS VoiceDesign 1.7B",
                hf_repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                size_mb=3500,
                languages=list(QWEN_LANGUAGES),
                default=True,
            ),
        ],
    ),
]

_BY_ID: dict[str, EngineConfig] = {e.id: e for e in ENGINES}

# Cached backend instances, created lazily and reused (one model load per engine).
_instances: dict[str, TtsBackend] = {}
_lock = threading.Lock()


def get_config(engine_id: str) -> EngineConfig | None:
    return _BY_ID.get(engine_id)


def require_config(engine_id: str) -> EngineConfig:
    """Return a backend-owned registry row or fail immediately on drift."""
    config = get_config(engine_id)
    if config is None:
        raise RuntimeError(
            f"required TTS engine '{engine_id}' is missing from the registry"
        )
    return config


def is_installed(cfg: EngineConfig) -> bool:
    """Whether the engine can actually run — i.e. its heavy inference dependency
    is importable. We check `import_name` (the engine's own package) via
    `find_spec`, not the thin backend wrapper, because backends defer their heavy
    imports to `load()` so the wrapper imports even when the engine is absent."""
    if not cfg.import_name:
        return True
    try:
        return importlib.util.find_spec(cfg.import_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def active_variant(cfg: EngineConfig) -> ModelVariant | None:
    """The checkpoint the engine runs (or will load next): the loaded backend's
    `variant` when it has one, else the backend module's `select_variant`, else
    the default row. Importing a backend module never imports its heavy deps."""
    instance = _instances.get(cfg.id)
    loaded = getattr(instance, "variant", None)
    if isinstance(loaded, ModelVariant):
        return loaded
    try:
        module = importlib.import_module(f"ryu_tts.backends.{cfg.backend_module}")
    except ImportError:
        module = None
    selector = getattr(module, "select_variant", None)
    if callable(selector):
        return selector(cfg)
    return next((m for m in cfg.models if m.default), cfg.models[0] if cfg.models else None)


def supports_instruct(cfg: EngineConfig) -> bool:
    """Whether `instruct` takes effect with the engine's active checkpoint."""
    if not cfg.supports_instruct:
        return False
    if all(m.supports_instruct for m in cfg.models):
        return True
    variant = active_variant(cfg)
    return variant is None or variant.supports_instruct


def get_backend(engine_id: str) -> TtsBackend:
    """Lazily instantiate (and cache) the backend for an engine.

    Imports happen here, not at module load, so a missing optional dependency for
    one engine never breaks the whole server — only that engine fails."""
    if engine_id in _instances:
        return _instances[engine_id]

    cfg = _BY_ID.get(engine_id)
    if cfg is None:
        raise ValueError(
            f"unknown TTS engine '{engine_id}'. Available: {sorted(_BY_ID)}"
        )

    with _lock:
        if engine_id in _instances:
            return _instances[engine_id]
        module = importlib.import_module(f"ryu_tts.backends.{cfg.backend_module}")
        backend_cls = getattr(module, cfg.backend_class)
        instance: TtsBackend = backend_cls()
        _instances[engine_id] = instance
        return instance


def loaded_ids() -> set[str]:
    return {eid for eid, b in _instances.items() if b.is_loaded()}


def is_model_cached(hf_repo_id: str) -> bool:
    """Whether a model repo is present in the engine's HF cache (the same cache
    every huggingface_hub-based engine reads, relocated by Core via `HF_HOME`).
    Mirrors voicebox's `is_model_cached`: a `models--<org>--<repo>` dir with a
    non-empty `snapshots/` means it is downloaded."""
    try:
        from huggingface_hub import constants as hf_constants

        repo_cache = (
            pathlib.Path(hf_constants.HF_HUB_CACHE)
            / f"models--{hf_repo_id.replace('/', '--')}"
        )
        snapshots = repo_cache / "snapshots"
        if not snapshots.exists():
            return False
        # A pending download leaves `.incomplete` blobs.
        blobs = repo_cache / "blobs"
        if blobs.exists() and any(blobs.glob("*.incomplete")):
            return False
        return any(snapshots.iterdir())
    except (ImportError, OSError, ValueError):
        return False


def all_models() -> list[dict]:
    """Flatten every engine's curated model variants into installable rows, each
    bound to its engine and tagged with cache (installed) state."""
    rows: list[dict] = []
    for cfg in ENGINES:
        for m in cfg.models:
            rows.append(
                {
                    "model_name": m.model_name,
                    "display_name": m.display_name,
                    "engine": cfg.id,
                    "engine_display_name": cfg.display_name,
                    "hf_repo_id": m.hf_repo_id,
                    "size_mb": m.size_mb,
                    "languages": m.languages,
                    "default": m.default,
                    "installed": is_model_cached(m.hf_repo_id),
                }
            )
    return rows


def find_model(engine_id: str, model_name: str) -> ModelVariant | None:
    cfg = _BY_ID.get(engine_id)
    if cfg is None:
        return None
    for m in cfg.models:
        if m.model_name == model_name:
            return m
    return None
