from __future__ import annotations

from convsearch.passages.builder import build_passages_for_message


def test_short_message() -> None:
    passages = build_passages_for_message(1, 1, "short useful text", 20, 3)
    assert len(passages) == 1
    assert passages[0].text == "short useful text"


def test_empty_content() -> None:
    assert build_passages_for_message(1, 1, "  ", 20, 3) == []


def test_paragraph_preservation() -> None:
    text = "First paragraph.\n\nSecond paragraph has more words."
    passages = build_passages_for_message(1, 1, text, 50, 5)
    assert "\n\n" in passages[0].text


def test_long_message_splits() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    passages = build_passages_for_message(1, 1, text, 20, 5)
    assert len(passages) > 1
    assert all(p.word_count > 0 for p in passages)


def test_overlap_behavior() -> None:
    text = " ".join(f"word{i}" for i in range(60))
    passages = build_passages_for_message(1, 1, text, 20, 5)
    assert "word15 word16 word17 word18 word19" in passages[1].text
