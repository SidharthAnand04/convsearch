from __future__ import annotations

from convsearch.passages.builder import build_passages_for_message


def test_repeated_paragraph_offsets_are_stable() -> None:
    text = "repeat paragraph one.\n\nrepeat paragraph one.\n\nfinal paragraph."
    passages = build_passages_for_message(1, 1, text, target_words=3, overlap_words=0)
    assert len(passages) >= 2
    for passage in passages:
        source = text[passage.start_offset : passage.end_offset]
        assert passage.text.replace("\n\n", " ") in source.replace("\n\n", " ")
