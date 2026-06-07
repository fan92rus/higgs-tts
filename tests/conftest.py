"""pytest fixtures for Higgs TTS tests.

Strategy:
  - Tests that need torch (server API tests via TestClient) use
    pytest.importorskip("torch") and run with real torch.
  - Tests that don't need torch (config, text chunking, hardware
    detection with mocking) run without it.
  - The CI workflow installs torch as part of the test step.
  - For hardware detection tests, torch is mocked at function level
    (inside detect()) via unittest.mock.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add src/ to sys.path so tests can import project modules
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Ensure torchaudio is importable (needed by audio_codec)
# ---------------------------------------------------------------------------
try:
    import torchaudio  # noqa: F401
except ImportError:
    import types
    _MOCK_TORCHAUDIO = types.ModuleType("torchaudio")
    _MOCK_TORCHAUDIO.__version__ = "0.0.0"
    _MOCK_TORCHAUDIO.__spec__ = importlib.util.spec_from_loader("torchaudio", loader=None)
    import sys
    sys.modules["torchaudio"] = _MOCK_TORCHAUDIO

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_autostart() -> None:
    """Prevent the server from starting the model download on TestClient init.

    The real model download (~9.3 GB) must never run during tests.
    """
    os.environ.setdefault("HIGGS_NO_AUTOSTART", "1")
    yield


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient with model autostart disabled.

    Requires real torch to be installed (skip with pytest.importorskip).
    """
    from src.server import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_no_cuda() -> None:
    """Mock environment: no CUDA available, no NVIDIA GPU.

    torch is imported INSIDE detect(), so we mock at the function level.
    """
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("src.hardware._nvidia_gpu_present", return_value=False),
    ):
        yield


@pytest.fixture
def mock_cuda_4090() -> None:
    """Mock environment: RTX 4090 with 24 GB VRAM, CUDA available."""
    _mock_torch_cuda("NVIDIA GeForce RTX 4090", 8, 24 * 1024**3, 20 * 1024**3)
    with patch("src.hardware._nvidia_gpu_present", return_value=True):
        yield


def _mock_torch_cuda(
    name: str,
    major: int,
    total_mem: int,
    free_mem: int,
) -> None:
    """Configure torch.cuda mock state."""
    props = MagicMock()
    props.name = name
    props.major = major
    props.minor = 0
    props.total_memory = total_mem
    props.multi_processor_count = 128

    patchers = [
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.mem_get_info", return_value=(free_mem, total_mem)),
        patch("torch.cuda.get_device_properties", return_value=props),
    ]
    for p in patchers:
        p.start()
