"""Entry point — Higgs Audio v3 TTS, Portable by Neurogen.

Launches the FastAPI backend on a background thread, then opens the UI in a native
desktop window (pywebview). Falls back to the default web browser if pywebview is
unavailable.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

# Make ``src/`` importable whether launched as ``python src/main.py`` or from src/.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Keep all HF/torch caches inside the portable folder (don't pollute the profile).
_BASE = SRC_DIR.parent
os.environ.setdefault("HF_HOME", str(_BASE / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_BASE / ".cache" / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import branding  # noqa: E402
import config  # noqa: E402


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _run_server() -> None:
    import uvicorn

    from server import app

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


def main() -> None:
    config.ensure_dirs()
    print(f"  {branding.APP_TITLE}")
    print(f"  v{branding.VERSION}  ·  {branding.MODEL_ID}")
    print(f"  Сервер: http://{config.HOST}:{config.PORT}")

    server_thread = threading.Thread(target=_run_server, name="higgs-server", daemon=True)
    server_thread.start()

    url = f"http://{config.HOST}:{config.PORT}/"
    if not _wait_for_port(config.HOST, config.PORT, timeout=40.0):
        print("!! Не удалось запустить сервер.")
        sys.exit(1)

    # Preferred: native desktop window.
    try:
        import webview  # type: ignore

        webview.create_window(
            branding.APP_TITLE,
            url,
            width=1200,
            height=820,
            min_size=(940, 680),
            background_color="#0b0d12",
        )
        webview.start()  # blocks until the window is closed
        return
    except Exception as e:  # noqa: BLE001
        print(f"(pywebview недоступен: {e}) — открываю в браузере.")

    # Fallback: default browser, keep the server alive.
    import webbrowser

    webbrowser.open(url)
    try:
        while server_thread.is_alive():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
