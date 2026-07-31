from __future__ import annotations

import re
from collections.abc import Sequence

from convsearch.config.settings import SegmentationSettings
from convsearch.segmentation.models import ProposedSegment, SegmentableMessage
from convsearch.utils import stable_hash

BOUNDARY_RE = re.compile(r"\b(new topic|separately|another question|different topic)\b", re.I)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S+", re.M)


class RuleBasedSegmentationProvider:
    version = "rules-v1"

    def __init__(self, settings: SegmentationSettings) -> None:
        self.settings = settings

    def segment(self, messages: Sequence[SegmentableMessage]) -> list[ProposedSegment]:
        ordered = sorted(messages, key=lambda message: message.source_order)
        if not ordered:
            return []
        groups: list[list[SegmentableMessage]] = []
        current: list[SegmentableMessage] = []
        for message in ordered:
            if current and self._should_split(current, message):
                groups.append(current)
                current = []
            current.append(message)
        if current:
            groups.append(current)
        groups = self._merge_tiny(groups)
        return [self._make_segment(group, index) for index, group in enumerate(groups) if group]

    def _should_split(self, current: list[SegmentableMessage], message: SegmentableMessage) -> bool:
        if current[-1].is_primary_path != message.is_primary_path:
            return True
        if len(current) >= self.settings.maximum_segment_messages:
            return True
        if len(current) < self.settings.minimum_segment_messages:
            return False
        return bool(BOUNDARY_RE.search(message.text) or HEADING_RE.search(message.text))

    def _merge_tiny(self, groups: list[list[SegmentableMessage]]) -> list[list[SegmentableMessage]]:
        merged: list[list[SegmentableMessage]] = []
        for group in groups:
            if (
                merged
                and len(group) < self.settings.minimum_segment_messages
                and merged[-1][-1].is_primary_path == group[0].is_primary_path
            ):
                merged[-1].extend(group)
            else:
                merged.append(group)
        return merged

    def _make_segment(self, group: list[SegmentableMessage], segment_order: int) -> ProposedSegment:
        text = "\n".join(f"{message.role}: {message.text}" for message in group)
        title = _title_from(group)
        reasons = ("branch_boundary",) if not group[0].is_primary_path else ("rule_boundary",)
        return ProposedSegment(
            conversation_id=group[0].conversation_id,
            segment_order=segment_order,
            start_message_id=group[0].message_id,
            end_message_id=group[-1].message_id,
            title=title,
            summary=None,
            boundary_confidence=0.70,
            reasons=reasons,
            message_ids=tuple(message.message_id for message in group),
            content_hash=stable_hash(self.version, group[0].conversation_id, segment_order, text),
        )


def _title_from(messages: Sequence[SegmentableMessage]) -> str:
    for message in messages:
        if message.role == "user" and message.text.strip():
            return " ".join(message.text.split())[:80]
    return " ".join(messages[0].text.split())[:80]
