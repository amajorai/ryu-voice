"""Chatterbox backends — Resemble AI's zero-shot cloning TTS (MIT).

Upstream: https://github.com/resemble-ai/chatterbox (`pip install chatterbox-tts`).
Two registry rows share this module:

- `chatterbox` -> `ChatterboxMultilingualTTS` (23 languages, `language_id`).
- `chatterbox_turbo` -> `ChatterboxTurboTTS` (English, inline paralinguistic tags
  such as `[laugh]`, which the chunker keeps because the row lists them).

Both clone from `reference_audio` (a local wav path) when given, else speak the
checkpoint's bundled default voice. Upstream `generate(audio_prompt_path=...)`
overwrites the long-lived model's `conds`, so the bundled voice is snapshotted at
load and restored for every request without a reference; a cloned voice never
leaks into a later request. Neither exposes a speaking-rate control, so
their rows set `native_speed=False` and the server time-stretches. Sampling
parameters follow the upstream defaults; the per-language Hebrew tuning and the
eager-attention switch are adapted from jamiepine/voicebox (MIT).
"""

from __future__ import annotations

import copy
import logging
import os

import numpy as np

from ..device import pick_device
from ..registry import EngineConfig, require_config

logger = logging.getLogger(__name__)

# Lower temperature + higher CFG weight reads more clearly for these languages.
_LANG_OVERRIDES = {
    "he": {"exaggeration": 0.4, "cfg_weight": 0.7, "temperature": 0.65, "repetition_penalty": 2.5},
}
_MULTILINGUAL_DEFAULTS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "repetition_penalty": 2.0}


def _to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def _reference_path(reference_audio: str | None) -> str | None:
    """Chatterbox loads prompts with librosa, which needs a local file."""
    if not reference_audio:
        return None
    if not os.path.isfile(reference_audio):
        raise ValueError(f"reference_audio '{reference_audio}' is not a readable local file")
    return reference_audio


def _use_eager_attention(model) -> None:
    """The multilingual alignment analyser reads attention maps, which SDPA
    attention does not return; switch the T3 transformer to eager attention."""
    transformer = getattr(getattr(model, "t3", None), "tfmr", None)
    config = getattr(transformer, "config", None)
    if config is None or not hasattr(config, "_attn_implementation"):
        return
    config._attn_implementation = "eager"
    for layer in getattr(transformer, "layers", []):
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "_attn_implementation"):
            attention._attn_implementation = "eager"


class _ChatterboxBase:
    config: EngineConfig

    def __init__(self) -> None:
        self._model = None
        self._device: str | None = None
        self._default_conds = None

    def _load_model(self, device: str):  # pragma: no cover - overridden
        raise NotImplementedError

    def load(self) -> None:
        if self._model is not None:
            return
        # Upstream Chatterbox has MPS kernel gaps; Apple Silicon runs it on CPU.
        self._device = pick_device(allow_mps=False)
        logger.info("loading %s on %s", self.config.id, self._device)
        self._model = self._load_model(self._device)
        self._default_conds = copy.copy(getattr(self._model, "conds", None))

    def _prompt_for(self, reference_audio: str | None) -> str | None:
        """Resolve the reference path; with none, reinstate the bundled voice
        (a shallow copy, since upstream reassigns attributes on the live conds)."""
        path = _reference_path(reference_audio)
        if path is None:
            assert self._model is not None
            self._model.conds = copy.copy(self._default_conds)
        return path

    def _sample_rate(self) -> int:
        return int(getattr(self._model, "sr", None) or self.config.sample_rate)

    def unload(self) -> None:
        self._model = None
        self._default_conds = None
        if self._device == "cuda":
            import torch

            torch.cuda.empty_cache()
        self._device = None

    def is_loaded(self) -> bool:
        return self._model is not None


class ChatterboxBackend(_ChatterboxBase):
    config: EngineConfig = require_config("chatterbox")

    def _load_model(self, device: str):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # heavy import, deferred

        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        _use_eager_attention(model)
        return model

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "en",
        reference_audio: str | None = None,
        **_unused,
    ) -> tuple[np.ndarray, int]:
        self.load()
        assert self._model is not None
        lang = (language or "en").lower().replace("_", "-").split("-")[0]
        if lang not in self.config.languages:
            raise ValueError(f"language '{language}' is not supported; use one of {self.config.languages}")
        params = _LANG_OVERRIDES.get(lang, _MULTILINGUAL_DEFAULTS)

        import torch

        with torch.inference_mode():
            wav = self._model.generate(
                text,
                language_id=lang,
                audio_prompt_path=self._prompt_for(reference_audio),
                **params,
            )
        return _to_numpy(wav), self._sample_rate()


class ChatterboxTurboBackend(_ChatterboxBase):
    config: EngineConfig = require_config("chatterbox_turbo")

    def _load_model(self, device: str):
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # heavy import, deferred

        return ChatterboxTurboTTS.from_pretrained(device=device)

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        reference_audio: str | None = None,
        **_unused,
    ) -> tuple[np.ndarray, int]:
        self.load()
        assert self._model is not None

        import torch

        with torch.inference_mode():
            wav = self._model.generate(text, audio_prompt_path=self._prompt_for(reference_audio))
        return _to_numpy(wav), self._sample_rate()
