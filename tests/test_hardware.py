"""Tests for hardware detection module."""

from __future__ import annotations

from src.hardware import HardwarePlan, detect


class TestHardwarePlan:
    """HardwarePlan dataclass tests."""

    def test_to_dict_contains_precision(self) -> None:
        plan = HardwarePlan(
            device="cuda",
            gpu_name="RTX 4090",
            vram_total_gb=24.0,
            vram_free_gb=20.0,
            mode="bf16",
            codec_device="cpu",
            cuda_available=True,
            bnb_available=True,
            reason="test",
        )
        d = plan.to_dict()
        assert d["device"] == "cuda"
        assert d["precision"] == "bf16"
        assert d["gpu_name"] == "RTX 4090"
        assert d["vram_total_gb"] == 24.0

    def test_to_dict_cpu_mode(self) -> None:
        plan = HardwarePlan(
            device="cpu",
            gpu_name=None,
            vram_total_gb=None,
            vram_free_gb=None,
            mode="cpu",
            codec_device="cpu",
            reason="CPU mode",
        )
        d = plan.to_dict()
        assert d["device"] == "cpu"
        assert d["precision"] == "cpu"
        assert d["gpu_name"] is None


class TestDetect:
    """detect() function with mocked hardware."""

    def test_detect_cpu_fallback(self, mock_no_cuda) -> None:
        plan = detect("auto")
        assert plan.device == "cpu"
        assert plan.mode == "cpu"

    def test_detect_forced_cpu(self, mock_cuda_4090) -> None:
        plan = detect("cpu")
        assert plan.device == "cpu"
        assert plan.mode == "cpu"

    def test_detect_forced_bf16(self, mock_cuda_4090) -> None:
        plan = detect("bf16")
        assert plan.device == "cuda"
        assert plan.mode == "bf16"

    def test_detect_forced_fp16(self, mock_cuda_4090) -> None:
        plan = detect("fp16")
        assert plan.device == "cuda"
        assert plan.mode == "fp16"

    def test_detect_auto_high_vram(self, mock_cuda_4090) -> None:
        """RTX 4090 with 20 GB free → bf16 on GPU."""
        plan = detect("auto")
        assert plan.device == "cuda"
        assert plan.mode == "bf16"

    def test_detect_returns_hardware_plan(self, mock_no_cuda) -> None:
        plan = detect("auto")
        assert isinstance(plan, HardwarePlan)
        assert isinstance(plan.to_dict(), dict)
