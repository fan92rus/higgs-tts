"""One-time checkpoint conversion: bosonai/higgs-audio-v3-tts-4b -> portable layout.

The original checkpoint stores everything in one ``model.safetensors`` with custom
names (``body.layers.*``, ``tied.embedding.*``). This converter splits it into:

    backbone/   - a standard ``Qwen3ForCausalLM`` dir (so transformers can load it
                  with native bf16 / 8-bit / 4-bit / device_map support) + tokenizer
    heads.safetensors  - the fused multi-codebook embedding/head weight [N*V, D]
    codec/model.safetensors - the bundled v2 codec weights (keys keep the prefix)
    engine_config.json - {num_codebooks, codebook_size, hidden_size, audio_token_id}

Backbone weights are written in memory-frugal ~1.8 GB shards so peak RAM stays low.
After a successful conversion the giant original ``model.safetensors`` can be
deleted to reclaim ~9 GB.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Higgs ckpt prefixes.
_TEXT_EMB = "tied.embedding.text_embedding."
_BODY_LAYERS = "body.layers."
_BODY_NORM = "body.norm."
_FUSED_EMB = "tied.embedding.modality_embeddings.0.embedding."
_TEXT_HEAD = "tied.head.text_head."
_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."

_SHARD_BYTES = 1_800_000_000  # ~1.8 GB per backbone shard

# (fraction 0..1, message) -> None
ProgressCb = Callable[[float, str], None]


def _dtype_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _map_backbone_name(name: str) -> str | None:
    """Higgs ckpt name -> standard Qwen3ForCausalLM name, or None if not backbone."""
    if name.startswith(_TEXT_EMB):
        return "model.embed_tokens." + name[len(_TEXT_EMB) :]
    if name.startswith(_BODY_LAYERS):
        return "model.layers." + name[len(_BODY_LAYERS) :]
    if name.startswith(_BODY_NORM):
        return "model.norm." + name[len(_BODY_NORM) :]
    if name.startswith(_TEXT_HEAD):
        return "lm_head." + name[len(_TEXT_HEAD) :]
    return None


def _build_backbone_config(text_config: dict) -> "object":
    """Construct a standard Qwen3Config from the checkpoint's text_config."""
    from transformers import Qwen3Config

    cfg = dict(text_config)
    cfg.pop("architectures", None)
    cfg.pop("_name_or_path", None)
    cfg.pop("dtype", None)
    # transformers Qwen3 wants a flat rope_theta; the ckpt nests it under
    # rope_parameters and may leave the top-level null (-> wrong 10000 default).
    rope_theta = cfg.get("rope_theta")
    rope_params = cfg.pop("rope_parameters", None)
    if rope_theta is None and isinstance(rope_params, dict):
        rope_theta = rope_params.get("rope_theta")
    cfg["rope_theta"] = int(rope_theta) if rope_theta else 1_000_000
    cfg.setdefault("tie_word_embeddings", True)

    config = Qwen3Config(**cfg)
    config.architectures = ["Qwen3ForCausalLM"]
    config.torch_dtype = "bfloat16"
    return config


def convert_checkpoint(
    src_dir: str | Path,
    models_dir: str | Path,
    *,
    progress_cb: ProgressCb | None = None,
    reclaim_disk: bool = True,
) -> None:
    """Convert ``src_dir`` (downloaded repo) into the portable ``models_dir`` layout."""
    src_dir = Path(src_dir)
    models_dir = Path(models_dir)
    backbone_dir = models_dir / "backbone"
    codec_dir = models_dir / "codec"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    codec_dir.mkdir(parents=True, exist_ok=True)

    def report(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0.0, min(1.0, frac)), msg)

    src_config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    text_config = src_config["text_config"]
    audio_cfg = src_config.get("audio_encoder_config", {})

    num_codebooks = int(audio_cfg.get("num_codebooks", 8))
    codebook_size = int(audio_cfg.get("vocab_size", 1026))
    hidden_size = int(audio_cfg.get("out_dim", text_config.get("hidden_size", 2560)))
    audio_token_id = int(src_config.get("audio_token_id", -100))

    weights_path = src_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"{weights_path} missing; download incomplete.")

    report(0.0, "Анализ весов…")

    # Sharded backbone writer state.
    shard_buf: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_files: list[dict[str, torch.Tensor]] = []  # finalized shards (as dicts)
    weight_map: dict[str, str] = {}
    total_backbone_bytes = 0
    shard_index = 0

    codec_state: dict[str, torch.Tensor] = {}
    fused_weight: torch.Tensor | None = None

    def flush_shard() -> None:
        nonlocal shard_buf, shard_bytes, shard_index
        if not shard_buf:
            return
        shard_index += 1
        fname = f"model-{shard_index:05d}.safetensors"
        for k in shard_buf:
            weight_map[k] = fname
        shard_files.append({"__fname__": fname, **shard_buf})  # type: ignore
        shard_buf = {}
        shard_bytes = 0

    with safe_open(str(weights_path), framework="pt") as f:
        keys = list(f.keys())
        n_keys = len(keys)
        for i, name in enumerate(keys):
            if i % 25 == 0:
                report(i / max(n_keys, 1) * 0.85, f"Конвертация {i}/{n_keys}…")
            if name.startswith(_CODEC_PREFIX):
                # Keep the prefix so HiggsAudioCodec.from_pretrained finds them.
                codec_state[name] = f.get_tensor(name)
                continue
            if name.startswith(_FUSED_EMB):
                fused_weight = f.get_tensor(name)
                continue
            mapped = _map_backbone_name(name)
            if mapped is None:
                continue  # anything unexpected is dropped
            t = f.get_tensor(name)
            shard_buf[mapped] = t
            nb = _dtype_bytes(t)
            shard_bytes += nb
            total_backbone_bytes += nb
            if shard_bytes >= _SHARD_BYTES:
                flush_shard()
        flush_shard()

    # Re-number shards as XofY and write them out.
    total_shards = len(shard_files)
    report(0.86, "Запись бэкбона…")
    final_weight_map: dict[str, str] = {}
    for idx, shard in enumerate(shard_files, start=1):
        old_fname = shard.pop("__fname__")  # type: ignore
        new_fname = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, str(backbone_dir / new_fname), metadata={"format": "pt"})
        for k in list(weight_map.keys()):
            if weight_map[k] == old_fname:
                final_weight_map[k] = new_fname

    index = {
        "metadata": {"total_size": total_backbone_bytes},
        "weight_map": final_weight_map,
    }
    (backbone_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    # Backbone config + tokenizer files.
    config = _build_backbone_config(text_config)
    config.save_pretrained(str(backbone_dir))
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = src_dir / fn
        if src.is_file():
            shutil.copy2(src, backbone_dir / fn)

    # Fused head/embedding weight.
    report(0.92, "Запись аудио-голов…")
    if fused_weight is None:
        raise RuntimeError("Fused embedding weight not found in checkpoint.")
    save_file({"weight": fused_weight}, str(models_dir / "heads.safetensors"))

    # Codec weights.
    report(0.95, "Запись кодека…")
    if not codec_state:
        raise RuntimeError("Codec weights not found under expected prefix.")
    save_file(codec_state, str(codec_dir / "model.safetensors"))

    # engine_config.json
    (models_dir / "engine_config.json").write_text(
        json.dumps(
            {
                "num_codebooks": num_codebooks,
                "codebook_size": codebook_size,
                "hidden_size": hidden_size,
                "audio_token_id": audio_token_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Mark conversion complete (sentinel the orchestrator checks).
    (models_dir / "CONVERTED.ok").write_text("ok", encoding="utf-8")

    if reclaim_disk:
        report(0.98, "Освобождение места…")
        try:
            weights_path.unlink()
            idx = src_dir / "model.safetensors.index.json"
            if idx.is_file():
                idx.unlink()
        except OSError:
            pass

    report(1.0, "Конвертация завершена")


def is_converted(models_dir: str | Path) -> bool:
    models_dir = Path(models_dir)
    return (
        (models_dir / "CONVERTED.ok").is_file()
        and (models_dir / "engine_config.json").is_file()
        and (models_dir / "heads.safetensors").is_file()
        and (models_dir / "codec" / "model.safetensors").is_file()
        and (models_dir / "backbone" / "model.safetensors.index.json").is_file()
    )


__all__ = ["convert_checkpoint", "is_converted"]
