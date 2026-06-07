"""HiggsTTSEngine — pure-PyTorch Higgs Audio v3 TTS inference (Portable by Neurogen).

Loads a *converted* checkpoint (produced by ``convert.py``):

    models_dir/
      engine_config.json      # {num_codebooks, codebook_size, hidden_size, audio_token_id}
      backbone/               # standard Qwen3ForCausalLM dir (+ tokenizer)
      heads.safetensors       # fused multi-codebook embedding/head weight [N*V, D]
      codec/model.safetensors # v2 codec weights (keys keep the ckpt prefix)

and runs:  text (+ optional reference voice) -> Qwen3 hidden states ->
multi-codebook audio tokens (delay pattern) -> codec -> 24 kHz waveform.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from .audio_codec import HiggsAudioCodec
from .delay import EOC_ID, apply_delay_pattern, reverse_delay_pattern
from .modeling import HiggsFusedMultiTextEmbedding, HiggsFusedMultiTextHead
from .prompt import AUDIO_PLACEHOLDER_ID, HiggsTokenizerAdapter
from .sampler import HiggsSamplerState, step as sampler_step

SAMPLE_RATE = HiggsAudioCodec.SAMPLE_RATE  # 24_000

MAX_REF_SECONDS = 15   # cap reference length — prefill cost grows ~25 tokens/sec of ref
_PREFILL_CHUNK = 8     # prefill tokens per forward (lets us show progress + honor Stop)

# progress_cb(stage, current, total); stage in {"prefill", "decode"}
ProgressCb = Callable[[str, int, int], None]
WaveformLike = Union[torch.Tensor, np.ndarray]


def _total_ram_gb() -> float:
    """Best-effort total physical RAM in GB (used to pick CPU dtype)."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / (1024**3)
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        except Exception:
            return 16.0


@dataclass
class GenerationResult:
    waveform: np.ndarray          # float32 mono [-1, 1]
    sample_rate: int
    num_frames: int               # decoded codec frames (post un-delay)
    engine_seconds: float
    real_time_factor: float       # engine_seconds / audio_seconds (<1 = faster than realtime)


class HiggsTTSEngine:
    """Loads the converted checkpoint and synthesises speech."""

    def __init__(
        self,
        models_dir: str | Path,
        *,
        mode: str = "auto",
        device: str | None = None,
        codec_device: str | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.mode = mode
        self._requested_device = device
        self._requested_codec_device = codec_device

        cfg_path = self.models_dir / "engine_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"engine_config.json not found in {self.models_dir} — run the "
                f"converter first."
            )
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.num_codebooks: int = int(cfg["num_codebooks"])
        self.codebook_size: int = int(cfg["codebook_size"])
        self.hidden_size: int = int(cfg["hidden_size"])
        self.audio_token_id: int = int(cfg.get("audio_token_id", AUDIO_PLACEHOLDER_ID))
        # Valid acoustic code range is [0, codebook_size - 2); BOC/EOC are the top two.
        self.codec_vocab: int = self.codebook_size - 2

        self.backbone = None       # transformers Qwen3Model (hidden states)
        self.fused_embed: HiggsFusedMultiTextEmbedding | None = None
        self.fused_head: HiggsFusedMultiTextHead | None = None
        self.codec: HiggsAudioCodec | None = None
        self.prompt_adapter: HiggsTokenizerAdapter | None = None
        self.device = torch.device("cpu")
        self.compute_dtype = torch.float32
        self._loaded = False
        self.load_note = ""

    # ------------------------------------------------------------------ load / unload
    def unload(self) -> None:
        """Выгружает модель из GPU, освобождая VRAM. Кодек остаётся на CPU
        для быстрой перезагрузки (кодек ~0.4 ГБ в VRAM, его релоад идёт
        отдельно при load).
        Повторный вызов безопасен (не роняет ничего, если уже выгружено).
        """
        self.backbone = None
        self.fused_embed = None
        self.fused_head = None
        self.prompt_adapter = None
        self._loaded = False
        self._free_gpu()

    def load(self) -> None:
        if self._loaded:
            return
        # Try the requested mode, then degrade on failure (a VRAM spike / OOM /
        # weight-conversion error / "no kernel image" on GPU shouldn't crash the
        # app — fall back to a lighter mode, finally CPU). A successful mode wins.
        requested = self.mode
        self.load_note = ""
        last_err: Exception | None = None
        for mode in self._candidate_modes(requested):
            try:
                self._free_gpu()
                self._load_backbone(mode)
                if mode != self.mode:
                    print(
                        f"[higgs] режим '{self.mode}' не подошёл — работаю в '{mode}'.",
                        flush=True,
                    )
                self.mode = mode
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(
                    f"[higgs] не удалось загрузить в режиме '{mode}': "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                self.backbone = None
                self._free_gpu()
        if last_err is not None or self.backbone is None:
            raise RuntimeError(
                "Не удалось загрузить модель ни в одном режиме (GPU/CPU). "
                f"Последняя ошибка: {last_err}"
            )

        if requested != "cpu" and self.mode == "cpu":
            self.load_note = (
                "GPU-режим не заработал (видеокарта несовместима с этой сборкой "
                "PyTorch — например, новее, RTX 50xx) → работаю на CPU. "
                "Переустановите через 'Установка' [1] GPU, чтобы поставить подходящий PyTorch."
            )

        # Keep the codec on CPU: it frees ~0.4 GB of VRAM for the model (helps it
        # fit and avoids load-time spikes), and codec decode is fast on CPU anyway.
        self._requested_codec_device = "cpu"
        self._load_heads()
        self._load_codec()
        self._load_tokenizer()
        self._loaded = True

    @staticmethod
    def _candidate_modes(mode: str) -> list[str]:
        """Ordered modes to try: requested first, then progressively lighter, CPU last."""
        if mode == "cpu":
            return ["cpu"]
        order = ["bf16", "fp16", "8bit", "4bit", "cpu"]
        if mode in order:
            cands = order[order.index(mode):]
        else:
            cands = list(order)
        if "cpu" not in cands:
            cands.append("cpu")
        return cands

    @staticmethod
    def _free_gpu() -> None:
        import gc

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _load_backbone(self, mode: str) -> None:
        from transformers import AutoModelForCausalLM

        backbone_dir = str(self.models_dir / "backbone")
        kwargs: dict = {"low_cpu_mem_usage": True}
        quantization_config = None

        if mode in ("8bit", "4bit"):
            from transformers import BitsAndBytesConfig

            if mode == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            kwargs["quantization_config"] = quantization_config
            kwargs["device_map"] = {"": 0}
            kwargs["dtype"] = torch.float16
        elif mode == "cpu":
            # fp32 gives best CPU throughput but needs ~16 GB just for weights;
            # use bf16 (~8 GB) unless the machine has plenty of RAM.
            kwargs["dtype"] = (
                torch.float32 if _total_ram_gb() >= 24.0 else torch.bfloat16
            )
        elif mode == "fp16":
            kwargs["dtype"] = torch.float16
        else:  # bf16 (default GPU)
            kwargs["dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(backbone_dir, **kwargs)

        if quantization_config is None:
            # Quantized models are already placed by device_map; others move here.
            model = model.to("cpu" if mode == "cpu" else "cuda")
        model.eval()

        if mode != "cpu":
            # "no kernel image is available" (GPU newer than this PyTorch build,
            # e.g. RTX 50xx on a cu121 wheel) only fails at COMPUTE time, not at
            # load. Probe a real kernel now so the fallback chain can degrade to
            # CPU instead of crashing mid-generation.
            t = torch.ones(16, 16, device="cuda")
            _ = (t @ t).sum().item()
            del t

        # We only need the inner Qwen3Model (hidden states); lm_head is unused.
        self.backbone = model.model
        self.device = self.backbone.embed_tokens.weight.device
        self.compute_dtype = self.backbone.embed_tokens.weight.dtype

    def _load_heads(self) -> None:
        heads_path = self.models_dir / "heads.safetensors"
        state = load_safetensors(str(heads_path))
        weight = state["weight"]
        n, v, d = self.num_codebooks, self.codebook_size, self.hidden_size
        if tuple(weight.shape) != (n * v, d):
            raise ValueError(
                f"fused head weight shape {tuple(weight.shape)} != expected "
                f"{(n * v, d)}"
            )
        self.fused_embed = HiggsFusedMultiTextEmbedding(n, v, d)
        self.fused_head = HiggsFusedMultiTextHead(n, v, d)
        with torch.no_grad():
            self.fused_embed.weight.copy_(weight)
        # tie_word_embeddings (default for this checkpoint): share the storage.
        self.fused_head.weight = self.fused_embed.weight
        self.fused_embed.to(device=self.device, dtype=self.compute_dtype)
        self.fused_head.to(device=self.device, dtype=self.compute_dtype)
        self.fused_embed.eval()
        self.fused_head.eval()

    def _load_codec(self) -> None:
        codec_dir = str(self.models_dir / "codec")
        cdev = self._requested_codec_device or "cpu"
        # Codec stays fp32 (decode ConvTranspose is bf16/fp16-unstable).
        self.codec = HiggsAudioCodec.from_pretrained(
            codec_dir, device=cdev, dtype=torch.float32
        )

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(self.models_dir / "backbone"))
        self.prompt_adapter = HiggsTokenizerAdapter(tok)

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        reference_waveform: WaveformLike | None = None,
        reference_sample_rate: int | None = None,
        reference_text: str | None = None,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
        max_new_tokens: int = 1024,
        seed: int | None = None,
        progress_cb: ProgressCb | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        if not self._loaded:
            raise RuntimeError("Engine not loaded; call load() first.")
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        N = self.num_codebooks
        if seed is not None:
            torch.manual_seed(int(seed))

        t0 = time.perf_counter()

        # 1) reference voice -> delayed codes (for cloning)
        ref_codes_delayed: torch.Tensor | None = None
        num_ref_tokens = 0
        if reference_waveform is not None:
            # Cap reference length: each second of reference adds ~25 prefill tokens,
            # and on CPU every prefill token is a full model step (minutes for long clips).
            rw = reference_waveform
            sr = reference_sample_rate or SAMPLE_RATE
            max_samples = int(MAX_REF_SECONDS * sr)
            if hasattr(rw, "shape") and rw.shape[-1] > max_samples:
                rw = rw[..., :max_samples]
            ref_codes = self.codec.encode_reference(
                rw, sample_rate=reference_sample_rate
            )  # [T_ref, N] int64 (CPU)
            ref_codes_delayed = apply_delay_pattern(ref_codes)  # [T_ref + N - 1, N]
            num_ref_tokens = int(ref_codes_delayed.shape[0])

        # 2) prompt token ids
        ids = self.prompt_adapter.build_prompt(
            text,
            num_ref_tokens=num_ref_tokens,
            reference_text=reference_text,
        )
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)[None]  # [1, L]

        # 3) prefill input embeddings (paste ref-audio fused embeds at -100)
        placeholder_mask = input_ids == self.audio_token_id  # [1, L]
        safe_ids = torch.where(
            placeholder_mask, torch.zeros_like(input_ids), input_ids
        )
        inputs_embeds = self.backbone.embed_tokens(safe_ids)  # [1, L, D]
        if num_ref_tokens > 0:
            fused = self.fused_embed(
                ref_codes_delayed.to(self.device)
            )  # [num_ref_tokens, D]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[placeholder_mask] = fused.to(inputs_embeds.dtype)

        # 4) prefill — chunked so we can show progress and honor Stop. A single
        # forward over a long voice-clone prompt takes minutes on CPU and would be
        # uninterruptible with the counter stuck at 0. Chunking is equivalent for a
        # causal model (the KV cache accumulates across chunks).
        L = inputs_embeds.shape[1]
        past = None
        hidden_last = None
        for c0 in range(0, L, _PREFILL_CHUNK):
            if should_stop is not None and should_stop():
                raise RuntimeError("Генерация остановлена.")
            chunk = inputs_embeds[:, c0 : c0 + _PREFILL_CHUNK, :]
            # Explicit positions so chunked prefill is provably identical to one
            # forward pass (don't rely on the cache inferring them).
            cache_position = torch.arange(
                c0, c0 + chunk.shape[1], device=chunk.device
            )
            out = self.backbone(
                inputs_embeds=chunk,
                past_key_values=past,
                use_cache=True,
                cache_position=cache_position,
            )
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]  # [1, D]
            if progress_cb is not None:
                progress_cb("prefill", min(c0 + _PREFILL_CHUNK, L), L)

        # 5) autoregressive multi-codebook decode (delay pattern + EOC stop)
        state = HiggsSamplerState(num_codebooks=N)
        output_codes: list[torch.Tensor] = []
        max_steps = int(max_new_tokens)
        stopped = False
        for step_i in range(max_steps):
            if should_stop is not None and should_stop():
                stopped = True
                break
            logits_NV = self.fused_head.generate(hidden_last).to(torch.float32)[0]  # [N, V]
            codes_N = sampler_step(
                logits_NV,
                state,
                temperature=float(temperature),
                top_p=top_p,
                top_k=top_k,
            )
            output_codes.append(codes_N.to("cpu"))
            if progress_cb is not None and (step_i % 4 == 0):
                progress_cb("decode", step_i + 1, max_steps)
            if state.generation_done:
                break
            # next input embedding = fused embed of just-sampled codes
            next_embed = self.fused_embed(codes_N[None].to(self.device))  # [1, D]
            out = self.backbone(
                inputs_embeds=next_embed[:, None, :].to(self.compute_dtype),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]

        if len(output_codes) < N:
            if stopped:
                raise RuntimeError("Генерация остановлена (слишком рано — нет аудио).")
            raise RuntimeError(
                f"Generation produced only {len(output_codes)} frames (< num_codebooks="
                f"{N}); nothing decodable. Try again or adjust sampling settings."
            )

        # 6) un-delay -> clamp specials -> codec decode
        delayed_LN = torch.stack(output_codes, dim=0).to(torch.long)  # [L, N]
        codes_TN = reverse_delay_pattern(delayed_LN)  # [T, N]
        codes_TN = torch.where(
            codes_TN >= self.codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )
        waveform = self.codec.decode(codes_TN)  # [L_samples] float32 (CPU)
        wav = waveform.detach().to(torch.float32).cpu().numpy()
        # Guard against rare clipping from fp accumulation.
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak

        engine_seconds = time.perf_counter() - t0
        audio_seconds = wav.shape[-1] / SAMPLE_RATE if wav.size else 0.0
        rtf = (engine_seconds / audio_seconds) if audio_seconds > 0 else 0.0

        return GenerationResult(
            waveform=wav.astype(np.float32),
            sample_rate=SAMPLE_RATE,
            num_frames=int(codes_TN.shape[0]),
            engine_seconds=engine_seconds,
            real_time_factor=rtf,
        )


__all__ = ["HiggsTTSEngine", "GenerationResult", "SAMPLE_RATE"]
