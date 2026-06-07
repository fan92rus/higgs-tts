"""Tests for text chunking logic used in server.py.

Tests the two main functions:
  - _strip_control(text) — removes all <|token:value|> tags
  - _split_into_chunks(text) — splits long text on sentence boundaries,
    preserving leading delivery tokens (emotion/style/prosody).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="text chunking tests import from server.py which needs torch")

from src.server import _split_into_chunks, _strip_control


class TestStripControl:
    """_strip_control — remove inline control tokens."""

    def test_no_tokens(self) -> None:
        assert _strip_control("Hello world") == "Hello world"

    def test_single_emotion(self) -> None:
        assert _strip_control("<|emotion:amusement|> Hello") == "Hello"

    def test_multiple_tokens(self) -> None:
        text = "<|emotion:amusement|> <|style:whispering|> Hello world"
        assert _strip_control(text) == "Hello world"

    def test_mixed_positions(self) -> None:
        text = "Hello <|sfx:laughter|>Haha world"
        assert _strip_control(text) == "Hello Haha world"

    def test_empty_string(self) -> None:
        assert _strip_control("") == ""


class TestSplitIntoChunks:
    """_split_into_chunks — split long text while preserving lead tokens."""

    def test_short_text_no_lead(self) -> None:
        lead, chunks = _split_into_chunks("Hello world")
        assert lead == ""
        assert chunks == ["Hello world"]

    def test_short_text_with_lead(self) -> None:
        lead, chunks = _split_into_chunks("<|emotion:amusement|> Hello world")
        assert lead == "<|emotion:amusement|>"
        assert chunks == ["Hello world"]

    def test_text_within_limit(self) -> None:
        text = "Short sentence."
        lead, chunks = _split_into_chunks(text, max_chars=100)
        assert lead == ""
        assert chunks == ["Short sentence."]

    def test_splits_on_sentence_boundary(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        lead, chunks = _split_into_chunks(text, max_chars=20)
        assert lead == ""
        # Each chunk should be at most max_chars
        for ch in chunks:
            assert len(ch) <= 20, f"Chunk '{ch}' exceeds {20} chars"

    def test_lead_prepended_to_first_chunk_only(self) -> None:
        text = "<|emotion:amusement|> " + "Long sentence " * 20
        lead, chunks = _split_into_chunks(text, max_chars=50)
        assert lead == "<|emotion:amusement|>"
        assert len(chunks) > 1
        # Lead is returned separately — caller prepends it
        assert not chunks[0].startswith("<|emotion")

    def test_empty_body(self) -> None:
        lead, chunks = _split_into_chunks("<|emotion:amusement|> ")
        assert lead == "<|emotion:amusement|>"
        assert chunks == [""]

    def test_lead_only_no_body(self) -> None:
        lead, chunks = _split_into_chunks("<|emotion:amusement|>")
        assert lead == "<|emotion:amusement|>"
        # body is empty, but we still get an empty chunk
        assert chunks == [""]

    def test_newline_splits(self) -> None:
        text = "Line one.\n\nLine two.\nLine three."
        lead, chunks = _split_into_chunks(text, max_chars=30)
        assert lead == ""
        for ch in chunks:
            assert len(ch) <= 30

    def test_overlong_sentence_hard_split(self) -> None:
        """A single sentence longer than max_chars is hard-split."""
        long_word = "A" * 200
        text = f"Short. {long_word}. Short again."
        lead, chunks = _split_into_chunks(text, max_chars=50)
        assert lead == ""
        # The 200-char word should be split into 50-char pieces
        for ch in chunks:
            assert len(ch) <= 50

    def test_preserves_leading_tokens_order(self) -> None:
        text = "<|emotion:amusement|> <|prosody:speed_fast|> Hello world"
        lead, chunks = _split_into_chunks(text)
        assert "<|emotion:amusement|>" in lead
        assert "<|prosody:speed_fast|>" in lead
