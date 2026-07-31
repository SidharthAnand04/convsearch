from __future__ import annotations

import re
from dataclasses import dataclass

from convsearch.domain.models import Passage
from convsearch.utils import stable_hash

WORD_RE = re.compile(r"\S+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextSpan:
    text: str
    start_offset: int
    end_offset: int


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def build_passages_for_message(
    conversation_id: int,
    message_id: int,
    text: str,
    target_words: int,
    overlap_words: int,
) -> list[Passage]:
    if not text.strip():
        return []
    units = split_units(text, target_words)
    chunks = combine_units(units, target_words, overlap_words)
    passages: list[Passage] = []
    for index, chunk in enumerate(chunks):
        passage_text = chunk.text.strip()
        if not passage_text:
            continue
        passages.append(
            Passage(
                conversation_id=conversation_id,
                message_id=message_id,
                passage_order=index,
                text=passage_text,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                word_count=word_count(passage_text),
                content_hash=stable_hash(conversation_id, message_id, index, passage_text),
            )
        )
    return passages


def split_units(text: str, target_words: int) -> list[TextSpan]:
    paragraphs = _paragraph_spans(text)
    spans: list[TextSpan] = []
    for paragraph in paragraphs or [TextSpan(text.strip(), 0, len(text.strip()))]:
        if word_count(paragraph.text) <= target_words * 2:
            spans.append(paragraph)
            continue
        sentences = _sentence_spans(paragraph)
        if len(sentences) > 1:
            spans.extend(sentences)
            continue
        spans.extend(_word_spans(paragraph, target_words))
    return spans


def combine_units(units: list[TextSpan], target_words: int, overlap_words: int) -> list[TextSpan]:
    chunks: list[TextSpan] = []
    current: list[TextSpan] = []
    current_words = 0
    for unit in units:
        unit_words = word_count(unit.text)
        if current and current_words + unit_words > target_words:
            chunk = _join_spans(current)
            chunks.append(chunk)
            current = overlap_tail(chunk, overlap_words)
            current_words = sum(word_count(span.text) for span in current)
        current.append(unit)
        current_words += unit_words
    if current:
        chunks.append(_join_spans(current))
    return chunks


def overlap_tail(chunk: TextSpan, overlap_words: int) -> list[TextSpan]:
    if overlap_words <= 0:
        return []
    words = list(WORD_RE.finditer(chunk.text))
    if len(words) <= overlap_words:
        return [chunk]
    start = words[-overlap_words].start()
    return [
        TextSpan(
            chunk.text[start:],
            chunk.start_offset + start,
            chunk.end_offset,
        )
    ]


def _paragraph_spans(text: str) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        spans.append(TextSpan(match.group(0), match.start(), match.end()))
    return spans


def _sentence_spans(paragraph: TextSpan) -> list[TextSpan]:
    spans: list[TextSpan] = []
    start = 0
    for part in SENTENCE_RE.split(paragraph.text):
        stripped = part.strip()
        if not stripped:
            start += len(part)
            continue
        local_start = paragraph.text.find(stripped, start)
        local_end = local_start + len(stripped)
        spans.append(
            TextSpan(
                stripped,
                paragraph.start_offset + local_start,
                paragraph.start_offset + local_end,
            )
        )
        start = local_end
    return spans


def _word_spans(paragraph: TextSpan, target_words: int) -> list[TextSpan]:
    words = list(WORD_RE.finditer(paragraph.text))
    spans: list[TextSpan] = []
    for start_index in range(0, len(words), target_words):
        selected = words[start_index : start_index + target_words]
        if not selected:
            continue
        start = selected[0].start()
        end = selected[-1].end()
        spans.append(
            TextSpan(
                paragraph.text[start:end],
                paragraph.start_offset + start,
                paragraph.start_offset + end,
            )
        )
    return spans


def _join_spans(spans: list[TextSpan]) -> TextSpan:
    if not spans:
        return TextSpan("", 0, 0)
    return TextSpan(
        "\n\n".join(span.text for span in spans),
        spans[0].start_offset,
        spans[-1].end_offset,
    )
