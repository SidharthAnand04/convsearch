from __future__ import annotations

import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Any

from convsearch.diagnostics.doctor import Check


@dataclass(frozen=True)
class LlmReadiness:
    ready: bool
    backend: str | None
    checks: tuple[Check, ...]
    remediation: tuple[str, ...]
    summary: str


def probe_llm_readiness(settings: Any, *, timeout: float = 1.5) -> LlmReadiness:
    """Report whether the configured LLM backend can serve a request now."""
    llm = settings.llm
    binary_found = shutil.which("ollama") is not None
    checks: list[Check] = [
        Check("ollama_binary", binary_found, "installed" if binary_found else "not found on PATH")
    ]

    server_reachable, model_names = _ollama_models(llm.ollama_host, timeout)
    checks.append(
        Check(
            "ollama_server",
            server_reachable,
            "reachable" if server_reachable else "unreachable",
        )
    )
    model_present = server_reachable and _model_present(llm.ollama_model, model_names)
    checks.append(
        Check(
            "ollama_model",
            model_present,
            "available" if model_present else f"missing: {llm.ollama_model}",
        )
    )

    anthropic_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append(
        Check(
            "anthropic_api_key",
            anthropic_available,
            "configured" if anthropic_available else "not configured",
        )
    )
    local_only = llm.backend == "ollama"
    checks.append(
        Check(
            "llm_backend",
            True,
            f"mode={llm.backend}; local_only={'yes' if local_only else 'no'}",
        )
    )

    backend = _effective_backend(llm.backend, model_present, anthropic_available)
    remediation = _remediation(
        binary_found,
        server_reachable,
        model_present,
        llm.ollama_model,
        backend,
    )
    summary = _summary(backend, llm.backend, binary_found, server_reachable, model_present)
    return LlmReadiness(
        ready=backend is not None,
        backend=backend,
        checks=tuple(checks),
        remediation=remediation,
        summary=summary,
    )


def _ollama_models(host: str, timeout: float) -> tuple[bool, tuple[str, ...]]:
    url = f"{host.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False, ()
    models = data.get("models", [])
    if not isinstance(models, list):
        return True, ()
    return True, tuple(
        model["name"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    )


def _model_present(configured: str, installed: tuple[str, ...]) -> bool:
    configured_name = _without_latest(configured)
    return any(_without_latest(name) == configured_name for name in installed)


def _without_latest(model: str) -> str:
    return model[:-7] if model.endswith(":latest") else model


def _effective_backend(mode: str, ollama_available: bool, anthropic_available: bool) -> str | None:
    if mode == "ollama":
        return "ollama" if ollama_available else None
    if mode == "anthropic":
        return "anthropic" if anthropic_available else None
    if ollama_available:
        return "ollama"
    return "anthropic" if anthropic_available else None


def _remediation(
    binary_found: bool,
    server_reachable: bool,
    model_present: bool,
    model: str,
    backend: str | None,
) -> tuple[str, ...]:
    if backend == "anthropic":
        return ()
    if not binary_found:
        return ("winget install Ollama.Ollama", f"ollama pull {model}")
    if not server_reachable:
        return ("ollama serve", f"ollama pull {model}")
    if not model_present:
        return (f"ollama pull {model}",)
    return ()


def _summary(
    backend: str | None,
    mode: str,
    binary_found: bool,
    server_reachable: bool,
    model_present: bool,
) -> str:
    if backend == "ollama":
        return "Ask and Plan are ready to use the local Ollama backend."
    if backend == "anthropic":
        return "Ask and Plan are ready to use the Anthropic cloud backend."
    if not binary_found:
        return (
            "Ask and Plan cannot run because Ollama is not installed and no cloud fallback "
            "is available."
        )
    if not server_reachable:
        return (
            "Ask and Plan cannot run because the Ollama server is not reachable and no "
            "cloud fallback is available."
        )
    if not model_present:
        return (
            "Ask and Plan cannot run because the configured Ollama model is not installed "
            "and no cloud fallback is available."
        )
    return f"Ask and Plan cannot run with the configured {mode} backend."
