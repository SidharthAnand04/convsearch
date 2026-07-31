"""Inference tests for the local-LLM generation layer.

These verify the Ollama-first-with-cloud-fallback behavior in
``convsearch.llm.generate`` without any real network I/O or model download.

Run the default (offline) suite with::

    uv run pytest tests/test_llm_generate.py -q -m "not real_model"

The opt-in live smoke test is marked ``real_model`` and is deselected by the
default suite; even when selected it skips gracefully if Ollama is unreachable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import TracebackType
from typing import Any

import pytest

from convsearch.config.settings import LLMSettings, Settings
from convsearch.llm import generate as generate_mod
from convsearch.llm.generate import (
    GenerationResult,
    LLMUnavailableError,
    generate_text,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _settings(**llm_updates: Any) -> Settings:
    """A Settings instance with LLM overrides applied via model_copy."""
    base = Settings()
    return base.model_copy(update={"llm": base.llm.model_copy(update=llm_updates)})


class _FakeResponse:
    """Context-manager stand-in for the object returned by urlopen."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _install_ollama_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    captured: dict[str, Any],
) -> None:
    """Patch urlopen (where generate.py looks it up) to return a fake response.

    The Request object passed to urlopen is recorded in ``captured``.
    """

    def _fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> _FakeResponse:
        captured["request"] = request
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[union-attr]
        return _FakeResponse(payload)

    monkeypatch.setattr(generate_mod.urllib.request, "urlopen", _fake_urlopen)


def _install_ollama_failure(
    monkeypatch: pytest.MonkeyPatch,
    exc: BaseException,
) -> None:
    def _fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> _FakeResponse:
        raise exc

    monkeypatch.setattr(generate_mod.urllib.request, "urlopen", _fake_urlopen)


def _forbid_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> _FakeResponse:
        raise AssertionError("urlopen must not be called on this backend")

    monkeypatch.setattr(generate_mod.urllib.request, "urlopen", _fake_urlopen)


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]


class _FakeMessagesClient:
    """Fake for MessagesClient: records create() kwargs, returns fixed text."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.calls.append(kwargs)
        return _FakeAnthropicResponse(self._text)


def _install_cloud_client(monkeypatch: pytest.MonkeyPatch, text: str) -> _FakeMessagesClient:
    client = _FakeMessagesClient(text)
    monkeypatch.setattr(generate_mod, "make_messages_client", lambda: client)
    return client


def _install_cloud_failure(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    def _boom() -> Any:
        raise RuntimeError(message)

    monkeypatch.setattr(generate_mod, "make_messages_client", _boom)


# ---------------------------------------------------------------------------
# 1. Ollama success (mocked urllib)
# ---------------------------------------------------------------------------


def test_ollama_success(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(backend="ollama", ollama_model="llama3.2:1b")
    captured: dict[str, Any] = {}
    _install_ollama_response(monkeypatch, {"message": {"content": "hi from ollama"}}, captured)

    result = generate_text("sys prompt", "user prompt", settings=settings, max_tokens=42)

    assert result == GenerationResult(
        text="hi from ollama", backend="ollama", model=settings.llm.ollama_model
    )

    # The request targeted the Ollama chat endpoint.
    assert captured["url"] == f"{settings.llm.ollama_host}/api/chat"
    assert captured["method"] == "POST"

    body = captured["body"]
    assert body["model"] == "llama3.2:1b"
    assert body["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert body["options"]["num_predict"] == 42


# ---------------------------------------------------------------------------
# 2. auto falls back to cloud when Ollama is refused
# ---------------------------------------------------------------------------


def test_auto_falls_back_to_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(backend="auto", model="claude-haiku-4-5")
    _install_ollama_failure(monkeypatch, ConnectionRefusedError("refused"))
    client = _install_cloud_client(monkeypatch, "cloud answer")

    result = generate_text("sys", "prompt", settings=settings)

    assert result.backend == "anthropic"
    assert result.text == "cloud answer"
    assert result.model == settings.llm.model == "claude-haiku-4-5"

    # The cloud client actually received the request.
    assert client.calls
    assert client.calls[0]["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 3. backend='ollama' with Ollama down -> LLMUnavailableError, no fallback
# ---------------------------------------------------------------------------


def test_ollama_only_down_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(backend="ollama")
    _install_ollama_failure(monkeypatch, urllib.error.URLError("connection refused"))
    # No cloud fallback should be attempted; make it explode if it is.
    _install_cloud_failure(monkeypatch, "cloud must not be used")

    with pytest.raises(LLMUnavailableError) as excinfo:
        generate_text("sys", "prompt", settings=settings)

    message = str(excinfo.value)
    assert settings.llm.ollama_host in message
    assert "ollama" in message.lower()


# ---------------------------------------------------------------------------
# 4. backend='anthropic' ignores Ollama entirely
# ---------------------------------------------------------------------------


def test_anthropic_backend_never_touches_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(backend="anthropic", model="claude-haiku-4-5")
    _forbid_urlopen(monkeypatch)
    client = _install_cloud_client(monkeypatch, "cloud only")

    result = generate_text("sys", "prompt", settings=settings)

    assert result.backend == "anthropic"
    assert result.text == "cloud only"
    assert result.model == "claude-haiku-4-5"
    assert client.calls  # cloud was used, urlopen was not (would have raised)


# ---------------------------------------------------------------------------
# 5. backend='auto' with BOTH down -> LLMUnavailableError mentioning both
# ---------------------------------------------------------------------------


def test_auto_both_down_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(backend="auto")
    _install_ollama_failure(monkeypatch, ConnectionRefusedError("no ollama"))
    _install_cloud_failure(monkeypatch, "cloud creds missing")

    with pytest.raises(LLMUnavailableError) as excinfo:
        generate_text("sys", "prompt", settings=settings)

    message = str(excinfo.value)
    assert settings.llm.ollama_host in message
    assert "cloud creds missing" in message


# ---------------------------------------------------------------------------
# 6. max_tokens default falls back to settings.llm.answer_max_tokens
# ---------------------------------------------------------------------------


def test_max_tokens_defaults_to_answer_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(backend="ollama", answer_max_tokens=256)
    captured: dict[str, Any] = {}
    _install_ollama_response(monkeypatch, {"message": {"content": "ok"}}, captured)

    generate_text("sys", "prompt", settings=settings, max_tokens=None)

    assert captured["body"]["options"]["num_predict"] == 256
    assert settings.llm.answer_max_tokens == 256


# ---------------------------------------------------------------------------
# 7. LIVE smoke test (opt-in) — never runs in the default suite
# ---------------------------------------------------------------------------


@pytest.mark.real_model
def test_ollama_live_smoke() -> None:
    settings = LLMSettings()  # defaults: real localhost host + model
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Reply with a single short word."},
        ],
        "stream": False,
        "options": {"num_predict": 16},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Ollama not reachable: {exc}")

    text = json.loads(raw)["message"]["content"]
    assert isinstance(text, str)
    assert text.strip()
