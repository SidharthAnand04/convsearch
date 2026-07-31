from __future__ import annotations

import json

import pytest

from convsearch.config.settings import Settings
from convsearch.diagnostics.llm_readiness import probe_llm_readiness


class Response:
    def __init__(self, models: list[dict[str, str]]) -> None:
        self._body = json.dumps({"models": models}).encode()

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _settings() -> Settings:
    return Settings()


def _assert_no_secret(readiness: object, secret: str) -> None:
    output = str(readiness)
    assert secret not in output


def test_ollama_missing_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("convsearch.diagnostics.llm_readiness.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "convsearch.diagnostics.llm_readiness.urllib.request.urlopen",
        lambda *args, **kwargs: Response([]),
    )

    readiness = probe_llm_readiness(_settings())

    assert readiness.ready is False
    assert readiness.backend is None
    assert readiness.remediation == ("winget install Ollama.Ollama", "ollama pull gemma3:1b")


def test_ollama_server_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("convsearch.diagnostics.llm_readiness.shutil.which", lambda _: "ollama")
    monkeypatch.setattr(
        "convsearch.diagnostics.llm_readiness.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("down")),
    )

    readiness = probe_llm_readiness(_settings())

    assert readiness.ready is False
    assert readiness.backend is None
    assert readiness.remediation == ("ollama serve", "ollama pull gemma3:1b")


def test_ollama_model_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("convsearch.diagnostics.llm_readiness.shutil.which", lambda _: "ollama")
    monkeypatch.setattr(
        "convsearch.diagnostics.llm_readiness.urllib.request.urlopen",
        lambda *args, **kwargs: Response([{"name": "other:latest"}]),
    )

    readiness = probe_llm_readiness(_settings())

    assert readiness.ready is False
    assert readiness.backend is None
    assert readiness.remediation == ("ollama pull gemma3:1b",)


def test_ollama_fully_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("convsearch.diagnostics.llm_readiness.shutil.which", lambda _: "ollama")
    monkeypatch.setattr(
        "convsearch.diagnostics.llm_readiness.urllib.request.urlopen",
        lambda *args, **kwargs: Response([{"name": "gemma3:1b"}]),
    )

    readiness = probe_llm_readiness(_settings())

    assert readiness.ready is True
    assert readiness.backend == "ollama"
    assert readiness.remediation == ()


def test_anthropic_fallback_does_not_leak_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "not-a-real-anthropic-secret"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setattr("convsearch.diagnostics.llm_readiness.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "convsearch.diagnostics.llm_readiness.urllib.request.urlopen",
        lambda *args, **kwargs: Response([]),
    )

    readiness = probe_llm_readiness(_settings())

    assert readiness.ready is True
    assert readiness.backend == "anthropic"
    assert readiness.remediation == ()
    _assert_no_secret(readiness, secret)


def test_http_exception_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("convsearch.diagnostics.llm_readiness.shutil.which", lambda _: "ollama")
    monkeypatch.setattr(
        "convsearch.diagnostics.llm_readiness.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    readiness = probe_llm_readiness(_settings())

    assert readiness.ready is False
    assert readiness.backend is None
    assert readiness.remediation == ("ollama serve", "ollama pull gemma3:1b")
