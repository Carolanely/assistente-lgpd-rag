"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity)."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        normalized = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding para capturar parafrases."""

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        if "GEMINI_API_KEY" in os.environ:
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
        else:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self._embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

    def _embed(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        emb = np.array(r.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada quando a query e semanticamente similar."""
        if not self._queries:
            return None

        query_embedding = self._embed(query)
        matrix = np.vstack(self._embeddings)
        similarities = matrix @ query_embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.threshold:
            return self._answers[best_idx]
        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def clear(self) -> None:
        self._queries.clear()
        self._embeddings.clear()
        self._answers.clear()

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}
