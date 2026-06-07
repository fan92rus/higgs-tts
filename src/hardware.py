"""Hardware detection + precision-mode auto-selection (Portable by Neurogen).

Emits a human-readable ``reason`` (shown in the console and the UI) so it's clear
WHY a given mode was chosen — especially why something fell back to CPU.

Mode tiers (the codec always runs on CPU, so only the ~8.8 GB LM must fit on GPU):
  >=10 GB free  -> bf16/fp16 (full precision, no bitsandbytes)   [most reliable]
  >=6  GB free  -> 8-bit  (needs bitsandbytes)
  >=4.5 GB free -> 4-bit  (needs bitsandbytes)
  otherwise / no CUDA / no bitsandbytes -> CPU
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class HardwarePlan:
    device: str            # "cuda" | "cpu"
    gpu_name: str | None
    vram_total_gb: float | None
    vram_free_gb: float | None
    mode: str              # bf16 | fp16 | 8bit | 4bit | cpu
    codec_device: str      # "cuda" | "cpu"
    cuda_available: bool = False
    bnb_available: bool = False
    torch_version: str | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["precision"] = self.mode
        return d


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def _nvidia_gpu_present() -> bool:
    """Is there a physical NVIDIA GPU (per nvidia-smi), regardless of torch?"""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def detect(force_mode: str = "auto") -> HardwarePlan:
    """Return a :class:`HardwarePlan`. ``force_mode`` overrides auto-selection."""
    cuda = False
    gpu_name = None
    total_gb = None
    free_gb = None
    cc_major = 0
    tver = None
    try:
        import torch

        tver = torch.__version__
        if torch.cuda.is_available():
            cuda = True
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            cc_major = props.major
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                free_gb = round(free_b / (1024**3), 2)
                total_gb = round(total_b / (1024**3), 2)
            except Exception:
                total_gb = round(props.total_memory / (1024**3), 2)
                free_gb = total_gb
    except Exception:
        cuda = False

    has_bnb = _bnb_available()
    fm = (force_mode or "auto").lower()

    def plan(device: str, mode: str, codec: str, reason: str) -> HardwarePlan:
        p = HardwarePlan(
            device, gpu_name, total_gb, free_gb, mode, codec,
            cuda_available=cuda, bnb_available=has_bnb, torch_version=tver, reason=reason,
        )
        print(
            f"[higgs] железо: torch={tver} cuda={cuda} gpu={gpu_name} "
            f"vram_free={free_gb} bnb={has_bnb} => режим={mode} ({reason})",
            flush=True,
        )
        return p

    if not cuda:
        if _nvidia_gpu_present():
            return plan(
                "cpu", "cpu", "cpu",
                "NVIDIA-видеокарта есть, но PyTorch установлен без CUDA — "
                "переустановите через 'Установка' и выберите вариант [1] GPU",
            )
        return plan("cpu", "cpu", "cpu", "Совместимая NVIDIA-видеокарта не найдена — работаю на CPU")

    # Manual override (HIGGS_MODE).
    if fm in ("bf16", "fp16", "8bit", "4bit", "cpu"):
        mode = fm
        if mode in ("8bit", "4bit") and not has_bnb:
            mode = "bf16" if cc_major >= 8 else "fp16"
        if mode == "cpu":
            return plan("cpu", "cpu", "cpu", "Принудительный режим CPU (HIGGS_MODE)")
        return plan("cuda", mode, "cpu", f"Принудительный режим {mode} (HIGGS_MODE)")

    budget = free_gb if free_gb is not None else (total_gb or 0.0)
    full = "bf16" if cc_major >= 8 else "fp16"

    if budget >= 10.0:
        return plan("cuda", full, "cpu", f"{free_gb} ГБ свободно → {full} на GPU")
    if has_bnb and budget >= 6.0:
        return plan("cuda", "8bit", "cpu", f"{free_gb} ГБ → 8-bit на GPU (bitsandbytes)")
    if has_bnb and budget >= 4.5:
        return plan("cuda", "4bit", "cpu", f"{free_gb} ГБ → 4-bit на GPU (bitsandbytes)")
    if budget >= 4.5 and not has_bnb:
        return plan(
            "cpu", "cpu", "cpu",
            f"{free_gb} ГБ GPU, но bitsandbytes недоступен → CPU "
            "(переустановите с вариантом [1] GPU, чтобы поставить bitsandbytes)",
        )
    return plan("cpu", "cpu", "cpu", f"VRAM мало ({free_gb} ГБ) для GPU-режима → CPU")


__all__ = ["HardwarePlan", "detect"]
