"""Paths and runtime defaults (Portable by Neurogen)."""

from __future__ import annotations

import os
from pathlib import Path

from branding import MODEL_ID

# src/ -> project root
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

WEB_DIR = SRC_DIR / "web"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
VOICES_DIR = BASE_DIR / "voices"
REFS_DIR = OUTPUTS_DIR / "_refs"

# Raw checkpoint download location (deleted/reclaimed after conversion).
DOWNLOAD_DIR = MODELS_DIR / "original"

REPO_ID = MODEL_ID

# HTTP server
HOST = os.environ.get("HIGGS_HOST", "127.0.0.1")
PORT = int(os.environ.get("HIGGS_PORT", "7861"))

# Sampling defaults (match the model card cookbook).
DEFAULTS = {
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 1.0,        # 1.0 == disabled
    "max_new_tokens": 1024,   # cap; the model emits EOC when speech ends
}

# Force a specific precision mode via env (auto | bf16 | fp16 | 8bit | 4bit | cpu).
FORCE_MODE = os.environ.get("HIGGS_MODE", "auto").strip().lower() or "auto"

# Idle timeout: unload model from GPU after N seconds of inactivity.
# Set to 0 to disable auto-unload entirely.
IDLE_UNLOAD_SEC = int(os.environ.get("HIGGS_IDLE_UNLOAD_SEC", "1800"))


def ensure_dirs() -> None:
    for d in (MODELS_DIR, OUTPUTS_DIR, VOICES_DIR, REFS_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "SRC_DIR",
    "BASE_DIR",
    "WEB_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "VOICES_DIR",
    "REFS_DIR",
    "DOWNLOAD_DIR",
    "REPO_ID",
    "HOST",
    "PORT",
    "DEFAULTS",
    "FORCE_MODE",
    "IDLE_UNLOAD_SEC",
    "ensure_dirs",
]
