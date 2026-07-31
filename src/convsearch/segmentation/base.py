from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from convsearch.segmentation.models import ProposedSegment, SegmentableMessage


class SegmentationProvider(Protocol):
    version: str

    def segment(self, messages: Sequence[SegmentableMessage]) -> list[ProposedSegment]: ...
