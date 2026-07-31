from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class EmbeddingModelError(RuntimeError):
    """The embedding model could not be loaded. The message names what to do about it."""


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int | None

    def encode(self, texts: list[str], batch_size: int) -> NDArray[np.float32]: ...

    def encode_documents(self, texts: Sequence[str], batch_size: int) -> NDArray[np.float32]: ...

    def encode_query(self, text: str) -> NDArray[np.float32]: ...


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_id: str, device: str = "auto", *, query_cache_size: int = 256) -> None:
        from importlib import import_module

        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as exc:
            raise EmbeddingModelError(
                "sentence-transformers is not installed, so text cannot be embedded. Run "
                "`uv sync --extra ml --group dev` in the convsearch checkout."
            ) from exc
        sentence_transformer = cast(Any, sentence_transformers).SentenceTransformer
        kwargs = {} if device == "auto" else {"device": device}
        try:
            self._model = sentence_transformer(model_id, **kwargs)
        except Exception as exc:
            # Constructing the model downloads it on first use and reads the HuggingFace cache
            # afterwards, so this fails on a typo'd model name, no network on first run, a
            # half-downloaded cache, or a device string this machine does not have. Every one
            # of those looked identical before: a bare exception type in a 500 body.
            raise EmbeddingModelError(
                f"could not load the embedding model {model_id!r} on device {device!r}: "
                f"{type(exc).__name__}: {exc}. On first run the model is downloaded once and "
                "needs network access; afterwards check `embedding_model` / "
                "`embedding_device` in the workspace's config.yaml and the HuggingFace cache."
            ) from exc
        self.model_id = model_id
        self.dimension: int | None = int(self._model.get_sentence_embedding_dimension() or 0)
        self._query_cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._query_cache_size = query_cache_size
        self._cache_lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        return self.encode_documents(texts, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int) -> NDArray[np.float32]:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Embed a query, memoised.

        Profiling a 300-conversation workspace put this at ~57ms of a ~96ms search — 60% of
        the total, and far more than the FAISS lookup (2.5ms) or FTS5 (11ms). The embedding
        of a given string under a given model never changes, so repeated queries (re-running
        a search, reopening the popup, paging) can skip the forward pass entirely.

        Bounded so a long session cannot grow without limit. Callers must not mutate the
        returned array; every current caller copies via np.asarray/reshape.
        """
        with self._cache_lock:
            cached = self._query_cache.get(text)
            if cached is not None:
                self._query_cache.move_to_end(text)
                return cached
        vector = self.encode_documents([text], batch_size=1)
        with self._cache_lock:
            self._query_cache[text] = vector
            if len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        return vector


class DeterministicEmbeddingProvider:
    def __init__(self, dimension: int = 16, model_id: str = "deterministic-test") -> None:
        self.dimension: int | None = dimension
        self.model_id = model_id

    def encode(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        return self.encode_documents(texts, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int) -> NDArray[np.float32]:
        dimension = self.dimension or 16
        vectors = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                index = sum(ord(char) for char in token) % dimension
                vectors[row, index] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def encode_query(self, text: str) -> NDArray[np.float32]:
        return self.encode_documents([text], batch_size=1)
