"""Tests for config module — paths, defaults, ensure_dirs."""

from __future__ import annotations

from pathlib import Path

import config


class TestConfigPaths:
    """Verify that all path constants point to valid locations."""

    def test_src_dir_exists(self) -> None:
        assert config.SRC_DIR.is_dir()
        assert (config.SRC_DIR / "server.py").is_file()

    def test_base_dir_is_parent_of_src(self) -> None:
        assert config.BASE_DIR == config.SRC_DIR.parent

    def test_web_dir_exists(self) -> None:
        assert config.WEB_DIR.is_dir()
        assert (config.WEB_DIR / "index.html").is_file()


class TestConfigDefaults:
    """Verify default sampling parameters."""

    def test_defaults_contains_required_keys(self) -> None:
        for key in ("temperature", "top_k", "top_p", "max_new_tokens"):
            assert key in config.DEFAULTS

    def test_temperature_in_range(self) -> None:
        assert 0 < config.DEFAULTS["temperature"] <= 2.0

    def test_max_new_tokens_positive(self) -> None:
        assert config.DEFAULTS["max_new_tokens"] > 0

    def test_force_mode_default(self) -> None:
        assert config.FORCE_MODE == "auto"


class TestEnsureDirs:
    """Verify ensure_dirs creates required directories."""

    def test_ensure_dirs_creates_refs(self, tmp_path: Path) -> None:
        # Temporarily patch REFS_DIR
        test_refs = tmp_path / "outputs" / "_refs"
        original = config.REFS_DIR
        try:
            config.REFS_DIR = test_refs  # type: ignore[assignment]
            config.ensure_dirs()
            assert test_refs.is_dir()
        finally:
            config.REFS_DIR = original
