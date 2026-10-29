"""Qwen3-TTS backends — Alibaba Qwen's 12 Hz TTS family (Apache-2.0).

Upstream: https://github.com/QwenLM/Qwen3-TTS (`pip install qwen-tts`). One
`Qwen3TTSModel` class loads three checkpoint kinds, each its own registry row:

- `qwen` (Base): `generate_voice_clone` from `reference_audio`, using
  `reference_text` (the sample's transcript) for in-context cloning; without a
  transcript it falls back to speaker-embedding-only cloning.
- `qwen_custom_voice`: `generate_custom_voice` with a preset speaker and an
  optional natural-language `instruct` ("speak slowly, sounding tired").
- `qwen_voice_design`: `generate_voice_design`, where `instruct` describes the
  voice itself.

The checkpoint size is the row's first cached `ModelVariant` (the default one
first), else the default one. Call shapes were
confirmed against the qwen-tts 0.1.1 wheel; the CPU float32 / accelerator
bfloat16 load split is adapted from jamiepine/voicebox (MIT).
"""

from __future__ import annotations

import logging

import numpy as np

from ..device import pick_device
from ..registry import EngineConfig, ModelVariant, is_model_cached, require_config

logger = logging.getLogger(__name__)

# Qwen3-TTS takes language *names*; it validates them case-insensitively.
_LANGUAGE_NAMES = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}
DEFAULT_DESIGN_PROMPT = "A clear, neutral voice with natural pacing."


def language_name(language: str | None) -> str:
    code = (language or "").lower().replace("_", "-").split("-")[0]
    return _LANGUAGE_NAMES.get(code, "auto")


def select_variant(config: EngineConfig) -> ModelVariant:
    """The first cached variant (default first), else the default variant
    (downloaded on first load)."""
    ordered = sorted(config.models, key=lambda m: not m.default)
    for variant in ordered:
        if is_model_cached(variant.hf_repo_id):
            return variant
    return ordered[0]


def _first_wave(result) -> tuple[np.ndarray, int]:
    wavs, sample_rate = result
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sample_rate)


class _QwenBase:
    config: EngineConfig

    def __init__(self) -> None:
        self._model = None
        self._device: str | None = None
        self.variant: ModelVariant | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_tts import Qwen3TTSModel  # heavy import, deferred

        self.variant = select_variant(self.config)
        self._device = pick_device()
        logger.info("loading %s on %s", self.variant.hf_repo_id, self._device)
        if self._device == "cpu":
            self._model = Qwen3TTSModel.from_pretrained(
                self.variant.hf_repo_id, torch_dtype=torch.float32, low_cpu_mem_usage=False
            )
        else:
            self._model = Qwen3TTSModel.from_pretrained(
                self.variant.hf_repo_id, device_map=self._device, torch_dtype=torch.bfloat16
            )

    def _run(self, method: str, **kwargs) -> tuple[np.ndarray, int]:
        self.load()
        assert self._model is not None
        import torch

        with torch.inference_mode():
            return _first_wave(getattr(self._model, method)(**kwargs))

    def unload(self) -> None:
        self._model = None
        if self._device == "cuda":
            import torch

            torch.cuda.empty_cache()
        self._device = None

    def is_loaded(self) -> bool:
        return self._model is not None


class QwenCloneBackend(_QwenBase):
    config: EngineConfig = require_config("qwen")

    def generate(
        self,
        text: str,
        *,
        language: str = "en",
        reference_audio: str | None = None,
        reference_text: str | None = None,
        **_unused,
    ) -> tuple[np.ndarray, int]:
        if not reference_audio:
            raise ValueError("Qwen3-TTS cloning needs reference_audio (a voice sample)")
        transcript = (reference_text or "").strip() or None
        return self._run(
            "generate_voice_clone",
            text=text,
            language=language_name(language),
            ref_audio=reference_audio,
            ref_text=transcript,
            x_vector_only_mode=transcript is None,
        )


class QwenCustomVoiceBackend(_QwenBase):
    config: EngineConfig = require_config("qwen_custom_voice")

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "en",
        instruct: str | None = None,
        **_unused,
    ) -> tuple[np.ndarray, int]:
        # Unknown speakers are rejected by the model's own (case-insensitive) check.
        speaker = voice or self.config.default_voice
        kwargs = {"text": text, "language": language_name(language), "speaker": speaker}
        if instruct and instruct.strip():
            self.load()
            # Upstream silently drops `instruct` on the 0.6B checkpoint; fail
            # loudly instead of returning an un-steered take.
            if self.variant is not None and not self.variant.supports_instruct:
                raise ValueError(
                    f"{self.variant.display_name} ignores delivery instructions; "
                    "download the 1.7B CustomVoice model or clear the instruction"
                )
            kwargs["instruct"] = instruct.strip()
        return self._run("generate_custom_voice", **kwargs)


class QwenVoiceDesignBackend(_QwenBase):
    config: EngineConfig = require_config("qwen_voice_design")

    def generate(
        self,
        text: str,
        *,
        language: str = "en",
        instruct: str | None = None,
        **_unused,
    ) -> tuple[np.ndarray, int]:
        description = (instruct or "").strip() or DEFAULT_DESIGN_PROMPT
        return self._run(
            "generate_voice_design",
            text=text,
            instruct=description,
            language=language_name(language),
        )
