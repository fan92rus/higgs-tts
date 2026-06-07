"""FastAPI backend for Higgs Audio v3 TTS — Portable by Neurogen.

Serves the web UI and the JSON API, and orchestrates the one-time
download -> convert -> load pipeline in a background thread, exposing live
progress over Server-Sent Events.
"""

from __future__ import annotations

import base64
import io
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import branding
import config
import hardware
from convert import convert_checkpoint, is_converted
from downloader import download_model

# ----------------------------------------------------------------------------- state


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.version = 0
        self.phase = "idle"           # idle|downloading|converting|loading|ready|error
        self.progress = 0.0           # 0..1
        self.message = "Готов к запуску"
        self.error: str | None = None
        self.plan: hardware.HardwarePlan | None = None
        self.engine = None
        self._prepare_thread: threading.Thread | None = None
        # transient generation progress
        self.gen_active = False
        self.gen_stage = "decode"   # "prefill" | "decode"
        self.gen_step = 0
        self.gen_total = 0
        self.gen_chunk = 0          # current text chunk (1-based)
        self.gen_chunks = 0         # total chunks for this request
        self.gen_started = 0.0
        self.cancel_event = threading.Event()
        self.last_used_at = time.time()  # для idle-unload

    def update(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)
            self.version += 1

    def hardware_dict(self) -> dict:
        if self.plan is None:
            return {
                "device": None,
                "gpu_name": None,
                "vram_total_gb": None,
                "vram_free_gb": None,
                "precision": None,
            }
        return self.plan.to_dict()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase,
                "progress": round(self.progress, 4),
                "message": self.message,
                "error": self.error,
                "hardware": self.hardware_dict(),
                "generation": {
                    "active": self.gen_active,
                    "stage": self.gen_stage,
                    "step": self.gen_step,
                    "total": self.gen_total,
                    "chunk": self.gen_chunk,
                    "chunks": self.gen_chunks,
                    "elapsed_s": (
                        round(time.time() - self.gen_started, 1)
                        if self.gen_active and self.gen_started
                        else 0.0
                    ),
                },
                "version": self.version,
                "idle_timeout_s": config.IDLE_UNLOAD_SEC,
                "idle_remaining_s": (
                    max(0, config.IDLE_UNLOAD_SEC - round(time.time() - self.last_used_at, 1))
                    if config.IDLE_UNLOAD_SEC > 0 and self.phase == "ready"
                    else None
                ),
            }


STATE = AppState()
app = FastAPI(title=branding.APP_TITLE)


# ---------------------------------------------------------------- idle unload watcher


def _unload_watcher() -> None:
    """Фоновый поток: проверяет каждые 5 сек, не простаивает ли модель дольше
    лимита. Если простаивает — выгружает из GPU."""
    while True:
        time.sleep(5)
        timeout = config.IDLE_UNLOAD_SEC
        if timeout <= 0:
            continue  # отключено
        try:
            with STATE.lock:
                if STATE.phase != "ready":
                    continue
                if STATE.engine is None or not STATE.engine.is_loaded:
                    continue
                idle = time.time() - STATE.last_used_at
                if idle < timeout:
                    continue
            # Вне блокировки — unload() может быть долгим (gc).
            engine = STATE.engine
            if engine is not None:
                engine.unload()
            with STATE.lock:
                STATE.engine = None
                STATE.phase = "idle"
                STATE.progress = 0.0
                STATE.message = f"Модель выгружена (простой {int(idle // 60)} мин)"
        except Exception:
            traceback.print_exc()


# ----------------------------------------------------------------------------- prepare flow


# ----------------------------------------------------------------------- prepare flow


def _prepare() -> None:
    """Background: detect hardware -> download -> convert -> load engine."""
    try:
        config.ensure_dirs()
        STATE.update(
            plan=hardware.detect(config.FORCE_MODE),
            phase="downloading",
            progress=0.0,
            message="Подготовка…",
            error=None,
        )

        # 1) download (idempotent / resumable) — skip if already converted.
        if not is_converted(config.MODELS_DIR):
            def dl_cb(frac: float, msg: str) -> None:
                STATE.update(phase="downloading", progress=frac, message=msg)

            download_model(config.REPO_ID, config.DOWNLOAD_DIR, progress_cb=dl_cb)

            # 2) convert
            def cv_cb(frac: float, msg: str) -> None:
                STATE.update(phase="converting", progress=frac, message=msg)

            STATE.update(phase="converting", progress=0.0, message="Конвертация…")
            convert_checkpoint(
                config.DOWNLOAD_DIR, config.MODELS_DIR, progress_cb=cv_cb
            )

        # 3) load engine
        STATE.update(phase="loading", progress=0.0, message="Загрузка модели в память…")
        from engine import HiggsTTSEngine

        plan = STATE.plan
        engine = HiggsTTSEngine(
            config.MODELS_DIR,
            mode=plan.mode,
            device=plan.device,
            codec_device=plan.codec_device,
        )
        engine.load()
        # Reflect the mode actually used (auto-fallback may have downgraded it).
        if STATE.plan is not None:
            STATE.plan.mode = engine.mode
            STATE.plan.device = "cpu" if engine.mode == "cpu" else "cuda"
            STATE.plan.codec_device = engine._requested_codec_device
            if getattr(engine, "load_note", ""):
                STATE.plan.reason = engine.load_note
        STATE.update(
            engine=engine,
            phase="ready",
            progress=1.0,
            message="Готово",
            error=None,
        )
    except Exception as e:  # noqa: BLE001
        STATE.update(
            phase="error",
            message="Ошибка подготовки модели",
            error=f"{type(e).__name__}: {e}",
        )
        traceback.print_exc()


def start_prepare() -> None:
    with STATE.lock:
        running = (
            STATE._prepare_thread is not None and STATE._prepare_thread.is_alive()
        )
        if running or STATE.phase == "ready":
            return
        t = threading.Thread(target=_prepare, name="higgs-prepare", daemon=True)
        STATE._prepare_thread = t
    t.start()


# ----------------------------------------------------------------------------- audio


def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any common audio file -> (mono float32 [-1,1], sample_rate)."""
    path = str(path)
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        return mono.astype(np.float32), int(sr)
    except Exception:
        pass
    # Fallback: torchaudio (handles mp3 if an ffmpeg/sox backend is present).
    import torchaudio

    wav, sr = torchaudio.load(path)
    mono = wav.mean(dim=0).numpy().astype(np.float32)
    return mono, int(sr)


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    """float32 mono [-1,1] -> 16-bit PCM WAV bytes."""
    try:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, wav, sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()
    except Exception:
        # Stdlib fallback.
        import wave

        pcm = np.clip(wav, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm16)
        return buf.getvalue()


# --------------------------------------------------------------------------- schemas


class Reference(BaseModel):
    ref_id: str | None = None
    preset: str | None = None


class GenerateRequest(BaseModel):
    text: str
    reference: Reference | None = None
    reference_text: str | None = None
    temperature: float = config.DEFAULTS["temperature"]
    top_k: int | None = config.DEFAULTS["top_k"]
    top_p: float | None = config.DEFAULTS["top_p"]
    max_new_tokens: int = config.DEFAULTS["max_new_tokens"]
    seed: int | None = None


# --------------------------------------------------------------------------- routes


@app.get("/api/info")
def api_info() -> JSONResponse:
    from engine.control_tokens import catalog

    return JSONResponse(
        {
            "branding": branding.PORTABLE_BY,
            "app_name": branding.APP_NAME,
            "version": branding.VERSION,
            "model_id": branding.MODEL_ID,
            "license_ru": branding.LICENSE_NOTE_RU,
            "license_en": branding.LICENSE_NOTE_EN,
            "defaults": config.DEFAULTS,
            "control_tokens": catalog(),
            "voices": _voices(),
        }
    )


@app.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(STATE.snapshot())


@app.get("/api/events")
def api_events() -> StreamingResponse:
    def gen():
        last = -1
        ticks = 0
        while True:
            snap = STATE.snapshot()
            if snap["version"] != last or ticks % 20 == 0:
                last = snap["version"]
                import json

                yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
            ticks += 1
            time.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/load")
def api_load() -> JSONResponse:
    start_prepare()
    return JSONResponse({"ok": True})


@app.post("/api/stop")
def api_stop() -> JSONResponse:
    """Request cancellation of the in-flight generation (cooperative)."""
    STATE.cancel_event.set()
    return JSONResponse({"ok": True})


def _voices() -> list[dict]:
    out: list[dict] = []
    if config.VOICES_DIR.is_dir():
        for wav in sorted(config.VOICES_DIR.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            out.append(
                {
                    "id": wav.stem,
                    "name": wav.stem.replace("_", " ").title(),
                    "description": (
                        txt.read_text(encoding="utf-8").strip()[:120]
                        if txt.is_file()
                        else "Пресет-голос"
                    ),
                }
            )
    return out


@app.get("/api/voices")
def api_voices() -> JSONResponse:
    return JSONResponse({"voices": _voices()})


@app.post("/api/upload_preset")
async def api_upload_preset(file: UploadFile = File(...), name: str | None = None) -> JSONResponse:
    """Загружает WAV-файл в voices/ — он становится доступен как пресет-голос."""
    config.ensure_dirs()
    suffix = Path(file.filename or "voice.wav").suffix or ".wav"
    if suffix.lower() not in (".wav", ".mp3"):
        return JSONResponse({"ok": False, "error": "Поддерживаются только .wav и .mp3"})

    stem = name.strip() if name else Path(file.filename or "voice").stem
    # Транслитерация/очистка для имени файла
    safe_stem = "".join(c for c in stem if c.isalnum() or c in " _-").strip() or "voice"
    dest = config.VOICES_DIR / f"{safe_stem}{suffix}"

    # Если файл уже существует — добавляем суффикс
    counter = 1
    while dest.exists():
        dest = config.VOICES_DIR / f"{safe_stem}_{counter}{suffix}"
        counter += 1

    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "Пустой файл"})
    dest.write_bytes(data)

    try:
        wav, sr = _load_audio(dest)
        duration = round(len(wav) / sr, 2) if sr else 0.0
    except Exception as e:
        dest.unlink(missing_ok=True)
        return JSONResponse({"ok": False, "error": f"Не удалось прочитать аудио: {e}"})

    return JSONResponse(
        {
            "ok": True,
            "preset_id": dest.stem,
            "name": dest.stem.replace("_", " ").title(),
            "duration_s": duration,
            "voices": _voices(),
        }
    )


@app.delete("/api/voices/{preset_id}")
def api_delete_voice(preset_id: str) -> JSONResponse:
    """Удаляет пресет-голос из voices/."""
    path = config.VOICES_DIR / f"{Path(preset_id).stem}.wav"
    txt = path.with_suffix(".txt")
    deleted = False
    if path.is_file():
        path.unlink()
        deleted = True
    if txt.is_file():
        txt.unlink()
    return JSONResponse({"ok": deleted, "voices": _voices()})


@app.post("/api/upload_reference")
async def api_upload_reference(file: UploadFile = File(...)) -> JSONResponse:
    config.ensure_dirs()
    ref_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "ref.wav").suffix or ".wav"
    dest = config.REFS_DIR / f"{ref_id}{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    try:
        wav, sr = _load_audio(dest)
        duration = round(len(wav) / sr, 2) if sr else 0.0
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"Не удалось прочитать аудио: {e}"})
    return JSONResponse(
        {"ok": True, "ref_id": dest.name, "name": file.filename, "duration_s": duration}
    )


def _resolve_reference(ref: Reference | None) -> tuple[np.ndarray | None, int | None]:
    if ref is None:
        return None, None
    if ref.ref_id:
        path = config.REFS_DIR / ref.ref_id
        if not path.is_file():
            raise FileNotFoundError("Референс не найден (ref_id).")
        return _load_audio(path)
    if ref.preset:
        path = config.VOICES_DIR / f"{ref.preset}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Пресет-голос '{ref.preset}' не найден.")
        return _load_audio(path)
    return None, None


# --------------------------------------------------------------- text chunking

# Higgs v3 is conversational — one turn naturally stops (~EOC) after ~30 s of
# audio, so long text must be split into short turns and concatenated.
CHUNK_MAX_CHARS = 220

_LEAD_RE = re.compile(r"^\s*((?:<\|(?:emotion|style|prosody):[^|>]+\|>\s*)+)")
_CTRL_RE = re.compile(r"<\|[^|>]+:[^|>]+\|>")
_SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def _strip_control(text: str) -> str:
    return _CTRL_RE.sub("", text).strip()


def _split_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS):
    """(lead_tokens, [chunks]) — split on sentence boundaries. Leading delivery
    tokens (emotion/style/prosody) are returned separately so they can be
    re-prepended to every chunk (keeps emotion through the whole text)."""
    m = _LEAD_RE.match(text)
    lead = m.group(1).strip() if m else ""
    body = (text[m.end():] if m else text).strip()
    if len(body) <= max_chars:
        return lead, [body] if body else [""]
    chunks: list[str] = []
    cur = ""
    for p in (s.strip() for s in _SENT_RE.split(body) if s.strip()):
        while len(p) > max_chars:  # hard-split an overlong sentence
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:max_chars].strip())
            p = p[max_chars:].strip()
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur += " " + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return lead, chunks or [body]


@app.post("/api/generate")
async def api_generate(req: GenerateRequest) -> JSONResponse:
    # Если модель выгружена по таймауту — перезагружаем автоматически.
    if STATE.phase == "idle" or (STATE.engine is None and STATE.phase != "error"):
        STATE.update(
            phase="loading",
            progress=0.0,
            message="Загрузка модели по запросу…",
        )
        # Загружаем модель синхронно в threadpool (так как load() блокирующий).
        from starlette.concurrency import run_in_threadpool

        def _reload() -> None:
            from engine import HiggsTTSEngine
            plan = STATE.plan or hardware.detect(config.FORCE_MODE)
            STATE.plan = plan
            engine = HiggsTTSEngine(
                config.MODELS_DIR,
                mode=plan.mode,
                device=plan.device,
                codec_device=plan.codec_device,
            )
            engine.load()
            if STATE.plan is not None:
                STATE.plan.mode = engine.mode
                STATE.plan.device = "cpu" if engine.mode == "cpu" else "cuda"
                STATE.plan.codec_device = engine._requested_codec_device
                if getattr(engine, "load_note", ""):
                    STATE.plan.reason = engine.load_note
            STATE.engine = engine
            STATE.phase = "ready"
            STATE.progress = 1.0
            STATE.message = "Готово"
            STATE.last_used_at = time.time()

        try:
            await run_in_threadpool(_reload)
        except Exception as e:
            STATE.update(phase="error", message="Ошибка загрузки модели", error=str(e))
            return JSONResponse(
                {"ok": False, "error": f"Не удалось загрузить модель: {e}"}, status_code=503
            )

    if STATE.engine is None or STATE.phase != "ready":
        return JSONResponse(
            {"ok": False, "error": "Модель ещё не готова."}, status_code=409
        )

    STATE.last_used_at = time.time()

    from starlette.concurrency import run_in_threadpool

    try:
        ref_wav, ref_sr = _resolve_reference(req.reference)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)})

    # preset transcript auto-fill
    ref_text = req.reference_text
    if ref_text is None and req.reference and req.reference.preset:
        txt = config.VOICES_DIR / f"{req.reference.preset}.txt"
        if txt.is_file():
            ref_text = txt.read_text(encoding="utf-8").strip()

    top_p = req.top_p if (req.top_p is not None and req.top_p < 1.0) else None

    def _progress(stage: str, current: int, total: int) -> None:
        STATE.update(gen_active=True, gen_stage=stage, gen_step=current, gen_total=total)

    SR = 24000

    def _run():
        lead, chunks = _split_into_chunks(req.text)
        is_zero_shot = ref_wav is None
        cur_ref_wav, cur_ref_sr, cur_ref_text = ref_wav, ref_sr, ref_text
        wavs: list[np.ndarray] = []
        total_engine = 0.0
        n = len(chunks)
        for i, ch in enumerate(chunks):
            if STATE.cancel_event.is_set():
                break
            prompt = (f"{lead} {ch}".strip() if lead else ch).strip()
            if not prompt:
                continue
            STATE.update(
                gen_active=True, gen_chunk=i + 1, gen_chunks=n,
                gen_stage="prefill", gen_step=0, gen_total=req.max_new_tokens,
                gen_started=time.time(),
            )
            seed = (int(req.seed) + i) if req.seed is not None else None
            try:
                r = STATE.engine.generate(
                    prompt,
                    reference_waveform=cur_ref_wav,
                    reference_sample_rate=cur_ref_sr,
                    reference_text=cur_ref_text,
                    temperature=req.temperature, top_k=req.top_k, top_p=top_p,
                    max_new_tokens=req.max_new_tokens, seed=seed,
                    progress_cb=_progress, should_stop=STATE.cancel_event.is_set,
                )
            except RuntimeError as e:
                if STATE.cancel_event.is_set() or "останов" in str(e).lower():
                    break  # stopped — keep whatever already finished
                if not wavs:
                    raise
                print(f"[higgs] часть {i + 1} пропущена: {e}", flush=True)
                continue
            wavs.append(r.waveform.astype(np.float32))
            total_engine += r.engine_seconds
            # Zero-shot: lock the voice by cloning the first chunk for the rest,
            # so the voice doesn't drift across chunks.
            if is_zero_shot and i == 0 and r.waveform.size > SR // 2:
                cur_ref_wav = r.waveform[: 12 * SR]
                cur_ref_sr = SR
                cur_ref_text = _strip_control(ch)[:300] or None
        if not wavs:
            raise RuntimeError("Ничего не сгенерировано (возможно, остановлено).")
        gap = np.zeros(int(0.12 * SR), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for w in wavs:
            pieces.append(w)
            pieces.append(gap)
        full = np.concatenate(pieces[:-1])
        return full, SR, total_engine, n

    STATE.cancel_event.clear()
    STATE.update(
        gen_active=True, gen_stage="prefill", gen_step=0,
        gen_total=req.max_new_tokens, gen_chunk=0, gen_chunks=0,
        gen_started=time.time(),
    )
    try:
        full, sr, total_engine, n_chunks = await run_in_threadpool(_run)
    except Exception as e:  # noqa: BLE001
        STATE.update(gen_active=False)
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"})
    STATE.update(gen_active=False)

    wav_bytes = _wav_bytes(full, sr)
    fname = f"higgs_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
    (config.OUTPUTS_DIR / fname).write_bytes(wav_bytes)
    dur = len(full) / sr if sr else 0.0
    return JSONResponse(
        {
            "ok": True,
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": sr,
            "duration_s": round(dur, 3),
            "engine_s": round(total_engine, 3),
            "rtf": round(total_engine / dur, 3) if dur > 0 else 0.0,
            "output_file": fname,
            "chunks": n_chunks,
        }
    )


@app.get("/api/audio/{filename}", response_model=None)
def api_audio(filename: str) -> FileResponse | PlainTextResponse:
    safe = Path(filename).name
    path = config.OUTPUTS_DIR / safe
    if not path.is_file():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(path, media_type="audio/wav")


# Static web UI (mounted LAST so /api/* routes take precedence).
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")


@app.on_event("startup")
def _on_startup() -> None:
    import os

    # Idle-unload watcher (daemon — не блокирует завершение).
    t = threading.Thread(target=_unload_watcher, name="higgs-unload", daemon=True)
    t.start()

    # Set HIGGS_NO_AUTOSTART=1 to bring up the server without kicking off the
    # download/convert/load pipeline (used for UI/static smoke tests).
    if os.environ.get("HIGGS_NO_AUTOSTART", "").strip() not in ("1", "true", "True"):
        start_prepare()


__all__ = ["app", "STATE", "start_prepare"]
