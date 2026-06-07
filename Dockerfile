# ============================================================================
# Higgs Audio v3 TTS — Portable by Neurogen
# Dockerfile (multi-stage, CUDA 12.1)
# ============================================================================
# Base image: official PyTorch with CUDA 12.1 + cuDNN 9 runtime.
# This image includes torch, torchvision, torchaudio, and the CUDA runtime.
# ============================================================================

# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS build

WORKDIR /app

# System libraries for audio I/O (soundfile, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leveraging Docker layer cache)
COPY requirements.txt .

# Install pip dependencies.
# torch/torchaudio are already in the base image — skip them.
# bitsandbytes is platform-conditional in requirements.txt (Windows-only marker),
# so we install it explicitly for Linux.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir bitsandbytes

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS runtime

WORKDIR /app

# Runtime system libraries (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from build stage
COPY --from=build /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

# Copy application source
COPY src/ src/
COPY voices/ voices/
COPY requirements.txt .

# Create persistent data directories (overridden by volumes in production)
RUN mkdir -p models outputs .cache/huggingface .cache/torch && \
    chmod 777 models outputs .cache

# ── Runtime configuration ────────────────────────────────────────────────────
ENV HIGGS_HOST=0.0.0.0
ENV HIGGS_PORT=7861
ENV HIGGS_MODE=auto
ENV HIGGS_IDLE_UNLOAD_SEC=1800
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV HIGGS_NO_AUTOSTART=0
ENV PYTHONPATH=/app/src

EXPOSE 7861

# Health check: /api/state returns 200 even during download/convert phases.
# start-period is generous because the first run downloads ~9.3 GB of weights.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -sf http://localhost:7861/api/state || exit 1

# Run FastAPI with uvicorn directly (no pywebview — Docker/browser only)
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7861"]
