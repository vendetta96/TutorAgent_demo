"""In-process vector store (numpy cosine similarity) with JSON persistence.

The embedder is injected, so the store and the incremental ingestion logic are
deterministic and testable with a fake embedder.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from tutor.knowledge.chunking import Chunk, chunk_document

DEFAULT_INDEX_PATH = Path("data/knowledge_index.json")


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, chunks: list[Chunk] | None = None, vectors: np.ndarray | None = None, model: str = ""):
        self.chunks: list[Chunk] = list(chunks or [])
        self.vectors: np.ndarray = (
            vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32)
        )
        self.model = model
        self._normalized = self._normalize(self.vectors)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def ids(self) -> set[str]:
        return {c.id for c in self.chunks}

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def search(self, query_vector: list[float] | np.ndarray, k: int = 3, min_score: float = 0.0) -> list[SearchHit]:
        if not self.chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        scores = self._normalized @ (q / norm)
        order = np.argsort(-scores)[:k]
        return [
            SearchHit(self.chunks[i], float(scores[i]))
            for i in order
            if float(scores[i]) >= min_score
        ]

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        for i, chunk in enumerate(self.chunks):
            if chunk.id == chunk_id:
                return self.vectors[i]
        return None

    def save(self, path: Path | str = DEFAULT_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors.astype(np.float32).tolist(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX_PATH) -> "VectorStore":
        path = Path(path)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in payload.get("chunks", [])]
        vectors = np.asarray(payload.get("vectors", []), dtype=np.float32)
        if vectors.ndim != 2:
            vectors = np.zeros((0, 0), dtype=np.float32)
        return cls(chunks, vectors, payload.get("model", ""))


@dataclass
class IngestResult:
    total_chunks: int
    embedded: int
    reused: int
    removed: int
    sources: list[str]


async def ingest_documents(
    documents: dict[str, str],
    embedder: Embedder,
    existing: VectorStore | None = None,
    *,
    max_chars: int = 900,
) -> tuple[VectorStore, IngestResult]:
    """Build a store from {source: text}. Chunks already embedded (same content hash
    and model) are reused, so re-ingesting after small edits only embeds what changed."""
    chunks: list[Chunk] = []
    for source, text in sorted(documents.items()):
        chunks.extend(chunk_document(text, source, max_chars=max_chars))

    reusable = existing if existing is not None and existing.model == embedder.model else VectorStore()
    vectors: list[np.ndarray | None] = [reusable.vector_for(c.id) for c in chunks]

    to_embed = [i for i, v in enumerate(vectors) if v is None]
    if to_embed:
        embedded = await embedder.embed([chunks[i].text for i in to_embed])
        for i, vec in zip(to_embed, embedded):
            vectors[i] = np.asarray(vec, dtype=np.float32)

    matrix = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)  # type: ignore[arg-type]
    store = VectorStore(chunks, matrix, embedder.model)
    removed = len(reusable.ids - store.ids) if existing is not None else 0
    return store, IngestResult(
        total_chunks=len(chunks),
        embedded=len(to_embed),
        reused=len(chunks) - len(to_embed),
        removed=removed,
        sources=sorted(documents),
    )


class KnowledgeBase:
    """Runtime retrieval facade: embeds the query and searches the store."""

    def __init__(self, store: VectorStore, embedder: Embedder | None, *, k: int = 3, min_score: float = 0.3):
        self.store = store
        self.embedder = embedder
        self.k = k
        self.min_score = min_score

    @property
    def available(self) -> bool:
        return self.embedder is not None and len(self.store) > 0

    async def search(self, query: str) -> list[SearchHit]:
        if not self.available or not query.strip():
            return []
        assert self.embedder is not None
        [vector] = await self.embedder.embed([query])
        return self.store.search(vector, k=self.k, min_score=self.min_score)

    async def passages(self, query: str) -> list[str]:
        hits = await self.search(query)
        return [f"{h.chunk.heading}: {h.chunk.text}" if h.chunk.heading else h.chunk.text for h in hits]
