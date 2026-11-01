"""Compute-device selection and RNG seeding shared by the torch-based engines.

The device is always detected: CUDA first, then Apple MPS, then CPU. It is not
read from the environment. Torch is only touched when an engine has already
imported it, so the ONNX engines (kokoro, kitten) never pay for it.
"""

from __future__ import annotations

import logging
import random
import sys

import numpy as np

logger = logging.getLogger(__name__)

# The device preference `/health` reports. Selection always detects.
DEVICE_PREFERENCE = "auto"


def requested_device() -> str:
    return DEVICE_PREFERENCE


def _cuda_available(torch_module) -> bool:
    try:
        return bool(torch_module.cuda.is_available())
    except Exception:  # noqa: BLE001 — a broken CUDA install means "no CUDA"
        return False


def _mps_available(torch_module) -> bool:
    try:
        backend = getattr(torch_module.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    except Exception:  # noqa: BLE001
        return False


def pick_device(*, allow_mps: bool = True, torch_module=None) -> str:
    """Return `"cuda"`, `"mps"`, or `"cpu"` for a torch engine.

    `allow_mps=False` is for engines with known MPS kernel gaps; they run on CPU
    on Apple Silicon even when MPS is available."""
    if torch_module is None:
        import torch as torch_module  # heavy import, only for torch engines

    if _cuda_available(torch_module):
        return "cuda"
    if allow_mps and _mps_available(torch_module):
        return "mps"
    return "cpu"


def seed_everything(seed: int | None) -> None:
    """Seed every RNG an engine may draw from. Torch is seeded only if some
    engine already imported it (never imported here just to seed)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return
    try:
        torch_module.manual_seed(seed)
        if _cuda_available(torch_module):
            torch_module.cuda.manual_seed_all(seed)
    except Exception as exc:  # noqa: BLE001 — seeding must never fail synthesis
        logger.warning("could not seed torch: %s", exc)
