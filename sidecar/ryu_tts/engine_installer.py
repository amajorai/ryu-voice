"""Shell-free install command construction for registered optional engines."""

from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import EngineConfig

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# Torch engines that only ever run on CPU.
_CPU_ONLY_TORCH = frozenset({"pocket"})
# Torch engines that benefit from a GPU; CPU wheels only when no GPU is in play.
_GPU_CAPABLE_TORCH = frozenset(
    {"chatterbox", "chatterbox_turbo", "qwen", "qwen_custom_voice", "qwen_voice_design"}
)


def _nvidia_gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def build_engine_install_command(
    cfg: EngineConfig,
    *,
    executable: str = sys.executable,
    platform: str = sys.platform,
    device: str | None = None,
    cuda_present: bool | None = None,
) -> list[str]:
    """Build an install command from one trusted engine registry row.

    PyPI's default Linux Torch wheel also installs the CUDA runtime (GBs). CPU-only
    engines (Pocket) always use the upstream CPU index on Linux. GPU-capable
    engines (Chatterbox, Qwen3-TTS) use it too unless `device="cuda"`, or `auto`
    (the default) on a machine where `nvidia-smi` is present; then the default
    CUDA wheels are kept. macOS and Windows default wheels are already CPU/MPS sized.
    """
    command = [executable, "-m", "pip", "install", *cfg.pip_packages]
    if not platform.startswith("linux"):
        return command
    if cfg.id in _CPU_ONLY_TORCH:
        command.extend(["--extra-index-url", CPU_TORCH_INDEX])
    elif cfg.id in _GPU_CAPABLE_TORCH:
        wanted = (device or "auto").strip().lower()
        has_gpu = _nvidia_gpu_present() if cuda_present is None else cuda_present
        wants_cuda = wanted == "cuda" or (wanted == "auto" and has_gpu)
        if not wants_cuda:
            command.extend(["--extra-index-url", CPU_TORCH_INDEX])
    return command
