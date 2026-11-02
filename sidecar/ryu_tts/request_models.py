"""Request bodies for `POST /generate`, kept apart from the route handlers."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import effects as fx
from .chunking import MAX_TEXT_CHARS, check_pause_budget

EffectType = Literal["gain", "highpass", "lowpass", "compressor", "reverb", "echo", "pitch_shift"]


class EffectSpec(BaseModel):
    """One post-synthesis effect. Params are clamped to the `/effects` ranges;
    unknown param names and non-finite values are rejected (a misspelled knob
    must not silently fall back to its default)."""

    model_config = ConfigDict(extra="forbid")

    type: EffectType
    params: dict[str, float] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _finite(cls, value: dict[str, float]) -> dict[str, float]:
        for name, number in value.items():
            if not math.isfinite(number):
                raise ValueError(f"effect param '{name}' must be a finite number")
        return value

    @model_validator(mode="after")
    def _known_params(self) -> EffectSpec:
        known = fx.param_names(self.type)
        unknown = sorted(set(self.params) - set(known))
        if unknown:
            raise ValueError(
                f"unknown {self.type} param(s) {', '.join(unknown)}; use {', '.join(known)}"
            )
        return self


class GenerateRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS, description="The words to speak.")
    engine: str = Field(..., description="Engine id from /engines (e.g. 'kitten').")
    voice: str | None = Field(None, description="Voice id; defaults to engine default.")
    speed: float = Field(
        1.0,
        ge=0.25,
        le=4.0,
        description="Speaking-rate multiplier; time-stretched for engines without native speed.",
    )
    language: str = Field("en")
    reference_audio: str | None = Field(
        None, description="Wav path/URL for cloning engines (ignored otherwise)."
    )
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    instruct: str | None = Field(
        None,
        max_length=500,
        description="Delivery instruction, or the voice description for design engines.",
    )
    reference_text: str | None = Field(
        None, max_length=1000, description="Transcript of `reference_audio`."
    )
    effects: list[EffectSpec] | None = Field(None, max_length=fx.MAX_EFFECTS)
    normalize: Literal["off", "peak", "podcast", "acx"] | None = None
    format: Literal["wav", "mp3", "ogg", "flac"] | None = None

    @field_validator("text")
    @classmethod
    def _pause_budget(cls, value: str) -> str:
        # Each `[pause]` is capped at 10s, but the count and total are bounded
        # too so a short script cannot plan hours of silence (sidecar OOM).
        check_pause_budget(value)
        return value
