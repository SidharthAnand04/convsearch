from __future__ import annotations

from dataclasses import dataclass, field

from convsearch.domain.models import ImportedConversation


@dataclass
class ImportParseResult:
    conversations: list[ImportedConversation] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
