"""Tests for FastAPI server endpoints.

Uses the TestClient fixture (autostart disabled) so no model download
is triggered during tests.

NOTE: requires torch to be installed (the server module imports
engine/audio_codec which imports transformers which imports torch).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="server tests require torch")

from fastapi.testclient import TestClient


class TestApiInfo:
    """GET /api/info — branding, defaults, control tokens, voices."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/info")
        assert resp.status_code == 200

    def test_contains_branding(self, client: TestClient) -> None:
        data = client.get("/api/info").json()
        assert "app_name" in data
        assert "version" in data
        assert "model_id" in data
        assert "branding" in data

    def test_contains_defaults(self, client: TestClient) -> None:
        data = client.get("/api/info").json()
        assert "defaults" in data
        for key in ("temperature", "top_k", "top_p", "max_new_tokens"):
            assert key in data["defaults"]

    def test_contains_control_tokens(self, client: TestClient) -> None:
        data = client.get("/api/info").json()
        assert "control_tokens" in data
        assert "emotions" in data["control_tokens"]

    def test_contains_voices(self, client: TestClient) -> None:
        data = client.get("/api/info").json()
        assert "voices" in data
        assert isinstance(data["voices"], list)

    def test_contains_licenses(self, client: TestClient) -> None:
        data = client.get("/api/info").json()
        assert "license_ru" in data
        assert "license_en" in data


class TestApiState:
    """GET /api/state — current server snapshot."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/state")
        assert resp.status_code == 200

    def test_contains_phase(self, client: TestClient) -> None:
        data = client.get("/api/state").json()
        assert "phase" in data
        assert data["phase"] in ("idle", "downloading", "converting", "loading", "ready", "error")

    def test_contains_hardware(self, client: TestClient) -> None:
        data = client.get("/api/state").json()
        assert "hardware" in data
        hw = data["hardware"]
        assert "device" in hw
        assert "precision" in hw

    def test_contains_generation(self, client: TestClient) -> None:
        data = client.get("/api/state").json()
        assert "generation" in data
        gen = data["generation"]
        assert "active" in gen
        assert "stage" in gen

    def test_version_increments(self, client: TestClient) -> None:
        v1 = client.get("/api/state").json()["version"]
        v2 = client.get("/api/state").json()["version"]
        assert v2 >= v1


class TestApiVoices:
    """GET /api/voices — list of preset voices."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/voices")
        assert resp.status_code == 200

    def test_returns_list(self, client: TestClient) -> None:
        data = client.get("/api/voices").json()
        assert "voices" in data
        assert isinstance(data["voices"], list)


class TestApiLoad:
    """POST /api/load — trigger model preparation."""

    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.post("/api/load")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestApiStop:
    """POST /api/stop — cancel generation."""

    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.post("/api/stop")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestApiAudio:
    """GET /api/audio/{filename} — serve generated files."""

    def test_nonexistent_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/audio/nonexistent.wav")
        assert resp.status_code == 404


class TestStaticFiles:
    """Static web UI files served at /."""

    def test_index_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_app_js(self, client: TestClient) -> None:
        resp = client.get("/app.js")
        assert resp.status_code == 200
        assert "application/javascript" in resp.headers["content-type"] or "text/javascript" in resp.headers["content-type"]

    def test_styles_css(self, client: TestClient) -> None:
        resp = client.get("/styles.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
