"""Higgs Audio v3 TTS — portable inference engine (Portable by Neurogen).

A self-contained, pure-PyTorch + HuggingFace transformers reimplementation of the
bosonai/higgs-audio-v3-tts-4b inference pipeline that runs natively on Windows
(GPU or CPU) without SGLang, Docker, or any CUDA-only kernels.

Pipeline:  text (+ optional reference voice) -> Qwen3 backbone -> multi-codebook
audio tokens (delay pattern) -> bundled Higgs v2 codec -> 24 kHz waveform.
"""

from .higgs_engine import HiggsTTSEngine, GenerationResult

__all__ = ["HiggsTTSEngine", "GenerationResult"]
