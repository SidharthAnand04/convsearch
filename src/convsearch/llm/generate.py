from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from convsearch.config.settings import Settings
from convsearch.llm.client import MessagesClient, make_messages_client

_OLLAMA_READ_TIMEOUT = 120.0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    backend: str  # "ollama" or "anthropic"
    model: str


class LLMUnavailableError(RuntimeError):
    """Raised when no configured generation backend could produce an answer."""


def generate_text(
    system: str,
    prompt: str,
    *,
    settings: Settings,
    max_tokens: int | None = None,
) -> GenerationResult:
    llm = settings.llm
    tokens = max_tokens if max_tokens is not None else llm.answer_max_tokens

    if llm.backend == "anthropic":
        return _generate_anthropic(system, prompt, settings, tokens)

    if llm.backend == "ollama":
        try:
            return _generate_ollama(system, prompt, settings, tokens)
        except Exception as exc:
            raise LLMUnavailableError(
                f"Ollama not reachable at {llm.ollama_host}: {exc}. "
                "Install/start Ollama (`ollama serve` + "
                f"`ollama pull {llm.ollama_model}`) or set llm.backend to "
                "'anthropic' with ANTHROPIC_API_KEY."
            ) from exc

    # backend == "auto": try local Ollama first, fall back to the cloud.
    try:
        return _generate_ollama(system, prompt, settings, tokens)
    except Exception:
        try:
            return _generate_anthropic(system, prompt, settings, tokens)
        except Exception as cloud_exc:
            raise LLMUnavailableError(
                f"Ollama not reachable at {llm.ollama_host} and cloud fallback "
                f"failed: {cloud_exc}. Install/start Ollama (`ollama serve` + "
                f"`ollama pull {llm.ollama_model}`) or set llm.backend to "
                "'anthropic' with ANTHROPIC_API_KEY."
            ) from cloud_exc


def _generate_ollama(
    system: str,
    prompt: str,
    settings: Settings,
    max_tokens: int,
) -> GenerationResult:
    llm = settings.llm
    url = f"{llm.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": llm.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_OLLAMA_READ_TIMEOUT) as response:
        raw = response.read().decode("utf-8")
    data: dict[str, Any] = json.loads(raw)
    text = data["message"]["content"]
    return GenerationResult(text=text, backend="ollama", model=llm.ollama_model)


def _generate_anthropic(
    system: str,
    prompt: str,
    settings: Settings,
    max_tokens: int,
    *,
    client: MessagesClient | None = None,
) -> GenerationResult:
    messages = client if client is not None else make_messages_client()
    response = messages.create(
        model=settings.llm.model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(block.text for block in response.content if block.type == "text")
    return GenerationResult(text=text, backend="anthropic", model=settings.llm.model)
