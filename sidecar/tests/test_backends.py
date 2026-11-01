"""Chatterbox and Qwen3-TTS backends against stubbed `torch`/`chatterbox`/
`qwen_tts` modules (no weights, no downloads), plus device selection, seeding,
the registry rows, and the install-command special cases."""

from __future__ import annotations

import contextlib
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from ryu_tts import device, registry
from ryu_tts.engine_installer import CPU_TORCH_INDEX, build_engine_install_command


def _fake_torch(*, cuda: bool = False, mps: bool = False) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"
    torch.seeds = []
    torch.manual_seed = torch.seeds.append
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda, manual_seed_all=lambda s: None, empty_cache=lambda: None
    )
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    torch.inference_mode = contextlib.nullcontext
    return torch


@pytest.fixture
def fake_torch(monkeypatch):
    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# ---------------------------------------------------------------------------
# device.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cuda", "mps", "allow_mps", "expected"),
    [
        (True, True, True, "cuda"),
        (False, True, True, "mps"),
        (False, True, False, "cpu"),
        (False, False, True, "cpu"),
    ],
)
def test_pick_device_detects_the_best_available(cuda, mps, allow_mps, expected):
    torch = _fake_torch(cuda=cuda, mps=mps)
    assert device.pick_device(allow_mps=allow_mps, torch_module=torch) == expected


def test_pick_device_ignores_the_removed_env_override(monkeypatch):
    monkeypatch.setenv("RYU_TTS_DEVICE", "cpu")
    torch = _fake_torch(cuda=True, mps=False)
    assert device.pick_device(torch_module=torch) == "cuda"
    assert device.requested_device() == "auto"


def test_seed_everything_seeds_python_numpy_and_imported_torch(fake_torch):
    device.seed_everything(123)
    first = (random.random(), float(np.random.rand()))
    device.seed_everything(123)
    assert (random.random(), float(np.random.rand())) == first
    assert fake_torch.seeds == [123, 123]
    device.seed_everything(None)
    assert fake_torch.seeds == [123, 123]


# ---------------------------------------------------------------------------
# Chatterbox
# ---------------------------------------------------------------------------


class _FakeChatterbox:
    instances: list[_FakeChatterbox] = []

    def __init__(self, device: str) -> None:
        self.device = device
        self.sr = 24000
        self.calls: list[tuple[str, dict]] = []
        self.conds = SimpleNamespace(voice="bundled")
        self.t3 = SimpleNamespace(
            tfmr=SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="sdpa"),
                layers=[SimpleNamespace(self_attn=SimpleNamespace(_attn_implementation="sdpa"))],
            )
        )
        _FakeChatterbox.instances.append(self)

    @classmethod
    def from_pretrained(cls, device):
        return cls(device)

    def generate(self, text, **kwargs):
        # Upstream `prepare_conditionals` replaces the model's long-lived conds.
        if kwargs.get("audio_prompt_path"):
            self.conds = SimpleNamespace(voice=kwargs["audio_prompt_path"])
        self.calls.append((text, kwargs, self.conds.voice))
        return np.zeros((1, 480), dtype=np.float32)


@pytest.fixture
def fake_chatterbox(monkeypatch, fake_torch):
    package = types.ModuleType("chatterbox")
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = type("ChatterboxMultilingualTTS", (_FakeChatterbox,), {})
    turbo = types.ModuleType("chatterbox.tts_turbo")
    turbo.ChatterboxTurboTTS = type("ChatterboxTurboTTS", (_FakeChatterbox,), {})
    monkeypatch.setitem(sys.modules, "chatterbox", package)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", turbo)
    _FakeChatterbox.instances.clear()
    return _FakeChatterbox


def test_chatterbox_multilingual_passes_language_and_reference(fake_chatterbox, tmp_path):
    from ryu_tts.backends.chatterbox import ChatterboxBackend

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    backend = ChatterboxBackend()
    samples, rate = backend.generate("Hola", language="es-MX", reference_audio=str(ref), seed=3, instruct="x")
    model = fake_chatterbox.instances[-1]
    text, kwargs, _voice = model.calls[-1]
    assert (text, kwargs["language_id"], kwargs["audio_prompt_path"]) == ("Hola", "es", str(ref))
    assert samples.shape == (480,) and rate == 24000
    assert model.device == "cpu"
    assert model.t3.tfmr.config._attn_implementation == "eager"
    assert backend.is_loaded()
    backend.unload()
    assert not backend.is_loaded()


def test_chatterbox_hebrew_uses_tuned_sampling(fake_chatterbox):
    from ryu_tts.backends.chatterbox import ChatterboxBackend

    ChatterboxBackend().generate("שלום", language="he")
    assert fake_chatterbox.instances[-1].calls[-1][1]["cfg_weight"] == 0.7


def test_chatterbox_rejects_unsupported_language_and_missing_reference(fake_chatterbox):
    from ryu_tts.backends.chatterbox import ChatterboxBackend

    backend = ChatterboxBackend()
    with pytest.raises(ValueError, match="not supported"):
        backend.generate("Hi", language="xx")
    with pytest.raises(ValueError, match="readable local file"):
        backend.generate("Hi", reference_audio="/no/such/file.wav")


def test_chatterbox_never_uses_mps(fake_chatterbox, monkeypatch, fake_torch):
    from ryu_tts.backends.chatterbox import ChatterboxTurboBackend

    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    ChatterboxTurboBackend().generate("Hi [laugh]")
    model = fake_chatterbox.instances[-1]
    assert model.device == "cpu"
    assert model.calls[-1] == ("Hi [laugh]", {"audio_prompt_path": None}, "bundled")


@pytest.mark.parametrize("backend_name", ["ChatterboxBackend", "ChatterboxTurboBackend"])
def test_chatterbox_cloned_voice_never_leaks_into_a_later_request(fake_chatterbox, tmp_path, backend_name):
    from ryu_tts.backends import chatterbox

    ref = tmp_path / "consented.wav"
    ref.write_bytes(b"RIFF")
    backend = getattr(chatterbox, backend_name)()
    backend.generate("Cloned", reference_audio=str(ref))
    backend.generate("Default")
    model = fake_chatterbox.instances[-1]
    assert [call[2] for call in model.calls] == [str(ref), "bundled"]


# ---------------------------------------------------------------------------
# Qwen3-TTS
# ---------------------------------------------------------------------------


class _FakeQwenModel:
    loads: list[tuple[str, dict]] = []

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        cls.loads.append((repo, kwargs))
        return cls()

    def _record(self, name, kwargs):
        self.calls.append((name, kwargs))
        return [np.zeros(240, dtype=np.float32)], 24000

    def generate_voice_clone(self, **kwargs):
        return self._record("clone", kwargs)

    def generate_custom_voice(self, **kwargs):
        return self._record("custom", kwargs)

    def generate_voice_design(self, **kwargs):
        return self._record("design", kwargs)


@pytest.fixture
def fake_qwen(monkeypatch, fake_torch):
    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = _FakeQwenModel
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    monkeypatch.setattr("ryu_tts.backends.qwen_tts.is_model_cached", lambda repo: False)
    _FakeQwenModel.loads.clear()
    return _FakeQwenModel


def test_qwen_clone_uses_reference_text_for_icl(fake_qwen):
    from ryu_tts.backends.qwen_tts import QwenCloneBackend

    backend = QwenCloneBackend()
    samples, rate = backend.generate("Hi", language="en", reference_audio="/tmp/ref.wav", reference_text=" hello ")
    name, kwargs = backend._model.calls[-1]
    assert name == "clone"
    assert kwargs == {"text": "Hi", "language": "english", "ref_audio": "/tmp/ref.wav",
                      "ref_text": "hello", "x_vector_only_mode": False}
    assert samples.shape == (240,) and rate == 24000
    repo, load_kwargs = fake_qwen.loads[-1]
    assert repo == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # default variant, CPU float32
    assert load_kwargs["torch_dtype"] == "float32"


def test_qwen_clone_without_transcript_uses_xvector_only(fake_qwen):
    from ryu_tts.backends.qwen_tts import QwenCloneBackend

    backend = QwenCloneBackend()
    backend.generate("Hi", language="xx", reference_audio="/tmp/ref.wav")
    kwargs = backend._model.calls[-1][1]
    assert kwargs["x_vector_only_mode"] is True and kwargs["ref_text"] is None
    assert kwargs["language"] == "auto"
    with pytest.raises(ValueError, match="reference_audio"):
        backend.generate("Hi")


def test_qwen_custom_voice_speaker_and_instruct(fake_qwen):
    from ryu_tts.backends.qwen_tts import QwenCustomVoiceBackend

    backend = QwenCustomVoiceBackend()
    backend.generate("Hi", voice="Vivian", language="zh", instruct="  happy  ")
    assert backend._model.calls[-1] == ("custom", {"text": "Hi", "language": "chinese",
                                                   "speaker": "Vivian", "instruct": "happy"})
    backend.generate("Hi")
    assert backend._model.calls[-1][1] == {"text": "Hi", "language": "english", "speaker": "Ryan"}
    assert fake_qwen.loads[-1][0] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"  # instruct-capable default


def test_qwen_custom_voice_0_6b_refuses_instruct_and_reports_it(fake_qwen, monkeypatch):
    from ryu_tts.backends.qwen_tts import QwenCustomVoiceBackend

    cfg = registry.require_config("qwen_custom_voice")
    assert registry.supports_instruct(cfg)  # nothing cached -> 1.7B default
    monkeypatch.setattr(
        "ryu_tts.backends.qwen_tts.is_model_cached", lambda repo: repo.endswith("0.6B-CustomVoice")
    )
    assert not registry.supports_instruct(cfg)  # only 0.6B cached -> 0.6B
    backend = QwenCustomVoiceBackend()
    with pytest.raises(ValueError, match="ignores delivery instructions"):
        backend.generate("Hi", instruct="speak slowly")
    backend.generate("Hi")  # plain speech still works on 0.6B
    assert "instruct" not in backend._model.calls[-1][1]


def test_qwen_voice_design_uses_instruct_as_the_voice(fake_qwen, monkeypatch):
    from ryu_tts.backends.qwen_tts import DEFAULT_DESIGN_PROMPT, QwenVoiceDesignBackend

    monkeypatch.setitem(sys.modules["torch"].__dict__, "cuda",
                        SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda s: None,
                                        empty_cache=lambda: None))
    backend = QwenVoiceDesignBackend()
    backend.generate("Hi", instruct="A gravelly old pirate")
    assert backend._model.calls[-1][1]["instruct"] == "A gravelly old pirate"
    repo, load_kwargs = fake_qwen.loads[-1]
    assert repo == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert load_kwargs == {"device_map": "cuda", "torch_dtype": "bfloat16"}
    backend.generate("Hi")
    assert backend._model.calls[-1][1]["instruct"] == DEFAULT_DESIGN_PROMPT


def test_qwen_variant_selection(monkeypatch):
    from ryu_tts.backends import qwen_tts

    cfg = registry.require_config("qwen")
    monkeypatch.setattr(qwen_tts, "is_model_cached", lambda repo: repo.endswith("1.7B-Base"))
    assert qwen_tts.select_variant(cfg).model_name == "qwen-tts-1.7B"  # the cached one
    monkeypatch.setattr(qwen_tts, "is_model_cached", lambda repo: False)
    assert qwen_tts.select_variant(cfg).model_name == "qwen-tts-0.6B"  # the default


# ---------------------------------------------------------------------------
# registry rows + backend wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_id", [e.id for e in registry.ENGINES])
def test_every_row_resolves_to_a_backend_class(engine_id):
    import importlib

    cfg = registry.require_config(engine_id)
    module = importlib.import_module(f"ryu_tts.backends.{cfg.backend_module}")
    backend_cls = getattr(module, cfg.backend_class)
    assert backend_cls.config is cfg
    assert cfg.default_voice in cfg.voices
    assert cfg.license


def test_licences_are_recorded():
    by_id = {e.id: e for e in registry.ENGINES}
    assert by_id["kokoro"].license == by_id["kitten"].license == "Apache-2.0"
    assert "CC-BY-4.0" in by_id["pocket"].license
    assert by_id["chatterbox"].license == by_id["chatterbox_turbo"].license == "MIT"
    assert all(by_id[i].license == "Apache-2.0" for i in ("qwen", "qwen_custom_voice", "qwen_voice_design"))
    assert all(e.commercial_use for e in registry.ENGINES)


# ---------------------------------------------------------------------------
# install commands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_id", ["chatterbox", "qwen_voice_design"])
def test_gpu_capable_engines_use_cpu_wheels_without_a_gpu(engine_id):
    cfg = registry.require_config(engine_id)
    command = build_engine_install_command(cfg, executable="py", platform="linux", device="auto", cuda_present=False)
    assert command[-2:] == ["--extra-index-url", CPU_TORCH_INDEX]
    assert command[4:-2] == cfg.pip_packages


@pytest.mark.parametrize(("device_env", "gpu"), [("auto", True), ("cuda", False)])
def test_gpu_capable_engines_keep_cuda_wheels_when_wanted(device_env, gpu):
    cfg = registry.require_config("chatterbox")
    command = build_engine_install_command(cfg, executable="py", platform="linux", device=device_env, cuda_present=gpu)
    assert "--extra-index-url" not in command


def test_gpu_capable_engines_forced_to_cpu_still_get_cpu_wheels():
    cfg = registry.require_config("qwen")
    command = build_engine_install_command(cfg, executable="py", platform="linux", device="cpu", cuda_present=True)
    assert CPU_TORCH_INDEX in command
    assert "--extra-index-url" not in build_engine_install_command(cfg, executable="py", platform="darwin")
