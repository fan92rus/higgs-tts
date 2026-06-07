"""Resumable HuggingFace downloader for the Higgs Audio v3 TTS checkpoint.

Streams each repo file with a Range-resumable HTTP GET and reports aggregate
byte progress through a callback, so the GUI can show a real progress bar for the
~9.3 GB weights download. No HF token required (the repo is public, not gated).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import requests

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# Files we need from the repo (the big one is model.safetensors).
REPO_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
]

# (fraction 0..1, human message) -> None
ProgressCb = Callable[[float, str], None]

_CHUNK = 1024 * 1024  # 1 MiB


def _resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{HF_ENDPOINT}/{repo_id}/resolve/{revision}/{filename}"


def _remote_size(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None


def _download_one(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    total_size: int | None,
    bytes_done_before: int,
    grand_total: int | None,
    cb: ProgressCb | None,
    label: str,
) -> int:
    """Download a single file with resume. Returns total bytes of this file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    # If a completed file already exists with the expected size, skip.
    if dest.exists() and total_size is not None and dest.stat().st_size == total_size:
        return total_size

    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with session.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # range not satisfiable -> already complete
            existing = tmp.stat().st_size if tmp.exists() else 0
        else:
            r.raise_for_status()
            mode = "ab" if existing > 0 and r.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0
            written = existing
            with open(tmp, mode) as f:
                for chunk in r.iter_content(chunk_size=_CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if cb is not None and grand_total:
                        frac = (bytes_done_before + written) / grand_total
                        mb = written / (1024 * 1024)
                        cb(min(frac, 0.999), f"{label}: {mb:,.0f} МБ")
            existing = written

    os.replace(tmp, dest)
    return dest.stat().st_size


def download_model(
    repo_id: str,
    dest_dir: str | Path,
    *,
    progress_cb: ProgressCb | None = None,
    revision: str = "main",
) -> Path:
    """Download all REPO_FILES into ``dest_dir``. Idempotent / resumable."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "HiggsTTS-Portable-by-Neurogen/1.0"})

    # Pre-compute total size for an accurate aggregate bar.
    sizes: dict[str, int | None] = {}
    for fn in REPO_FILES:
        sizes[fn] = _remote_size(session, _resolve_url(repo_id, fn, revision))
    known = [s for s in sizes.values() if s]
    grand_total = sum(known) if known else None

    bytes_done = 0
    for fn in REPO_FILES:
        url = _resolve_url(repo_id, fn, revision)
        dest = dest_dir / fn
        size = sizes.get(fn)
        # Some optional files (LICENSE/chat_template) may 404 on certain repos.
        try:
            this = _download_one(
                session,
                url,
                dest,
                total_size=size,
                bytes_done_before=bytes_done,
                grand_total=grand_total,
                cb=progress_cb,
                label=f"Скачивание {fn}",
            )
        except requests.HTTPError as e:
            if fn in ("LICENSE", "chat_template.jinja") and e.response is not None and e.response.status_code == 404:
                continue
            raise
        bytes_done += this or 0

    if progress_cb is not None:
        progress_cb(1.0, "Скачивание завершено")
    return dest_dir


__all__ = ["download_model", "REPO_FILES"]
