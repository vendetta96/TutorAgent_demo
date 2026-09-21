import hashlib

import numpy as np
import pytest

from tutor.knowledge.chunking import chunk_document, chunk_id
from tutor.knowledge.store import KnowledgeBase, VectorStore, ingest_documents

DOC = """# Earthquakes

Earthquakes happen when tectonic plates slip suddenly along a fault.

The energy released travels as seismic waves through the ground.

## Tsunamis

A tsunami is a series of giant waves usually caused by an undersea earthquake.

# Volcanoes

Volcanoes erupt when magma rises through the Earth's crust.
"""


class FakeEmbedder:
    """Deterministic embedder: a bag-of-words hash vector, so similar texts are close."""

    model = "fake-embed-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str, dims: int = 64) -> list[float]:
        v = np.zeros(dims, dtype=np.float32)
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % dims] += 1.0
        return v.tolist()


def test_chunking_splits_on_headings_and_is_deterministic():
    chunks = chunk_document(DOC, "geo.md")
    headings = [c.heading for c in chunks]
    assert headings == ["Earthquakes", "Tsunamis", "Volcanoes"]
    assert all(c.source == "geo.md" for c in chunks)
    assert "tectonic plates" in chunks[0].text and "seismic waves" in chunks[0].text
    again = chunk_document(DOC, "geo.md")
    assert [c.id for c in again] == [c.id for c in chunks]
    assert chunks[0].id == chunk_id("geo.md", "Earthquakes", chunks[0].text)


def test_chunking_packs_paragraphs_up_to_max_chars_with_overlap():
    body = "\n\n".join(f"Paragraph number {i} about floods." for i in range(10))
    chunks = chunk_document("# Floods\n\n" + body, "f.md", max_chars=80)
    assert len(chunks) > 1
    assert all(len(c.text) <= 160 for c in chunks)
    # Overlap: consecutive chunks share a paragraph.
    first_last = chunks[0].text.split("\n\n")[-1]
    assert first_last in chunks[1].text


def test_chunking_hard_splits_oversized_paragraph():
    long = "Rain fell. " * 300
    chunks = chunk_document(long, "rain.txt", max_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)


def test_empty_document_yields_no_chunks():
    assert chunk_document("", "empty.md") == []
    assert chunk_document("# Title only\n\n", "t.md") == []


async def test_ingest_search_and_incremental_reuse(tmp_path):
    embedder = FakeEmbedder()
    store, result = await ingest_documents({"geo.md": DOC}, embedder)
    assert result.total_chunks == 3 and result.embedded == 3 and result.reused == 0
    assert len(store) == 3 and store.model == "fake-embed-v1"

    hits = store.search(FakeEmbedder._vec("giant waves undersea earthquake tsunami"), k=2)
    assert hits[0].chunk.heading == "Tsunamis"
    assert hits[0].score > hits[1].score

    # Persist and reload.
    path = tmp_path / "index.json"
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 3 and loaded.model == "fake-embed-v1"
    assert loaded.search(FakeEmbedder._vec("magma crust erupt"), k=1)[0].chunk.heading == "Volcanoes"

    # Re-ingest with one new section: only the new chunk is embedded.
    embedder.calls.clear()
    updated = DOC + "\n# Droughts\n\nA drought is a long period with very little rain.\n"
    store2, result2 = await ingest_documents({"geo.md": updated}, embedder, existing=loaded)
    assert result2.total_chunks == 4 and result2.embedded == 1 and result2.reused == 3
    assert embedder.calls == [["A drought is a long period with very little rain."]]

    # Removing a document drops its chunks.
    store3, result3 = await ingest_documents({"other.md": "# X\n\nSomething else."}, embedder, existing=store2)
    assert result3.removed == 4 and len(store3) == 1


async def test_reuse_is_skipped_when_embedding_model_changes():
    embedder = FakeEmbedder()
    store, _ = await ingest_documents({"geo.md": DOC}, embedder)
    other = FakeEmbedder()
    other.model = "fake-embed-v2"
    _, result = await ingest_documents({"geo.md": DOC}, other, existing=store)
    assert result.embedded == 3 and result.reused == 0


def test_search_on_empty_store_and_zero_vector():
    store = VectorStore()
    assert store.search([1.0, 0.0]) == []
    store2 = VectorStore(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32))
    assert store2.search(np.zeros(4)) == []


def test_load_missing_index_returns_empty_store(tmp_path):
    store = VectorStore.load(tmp_path / "missing.json")
    assert len(store) == 0


async def test_knowledge_base_passages_respect_threshold():
    embedder = FakeEmbedder()
    store, _ = await ingest_documents({"geo.md": DOC}, embedder)
    kb = KnowledgeBase(store, embedder, k=2, min_score=0.2)
    passages = await kb.passages("what causes a tsunami")
    assert passages and passages[0].startswith("Tsunamis:")
    strict = KnowledgeBase(store, embedder, k=2, min_score=0.99)
    assert await strict.passages("what causes a tsunami") == []
    assert await kb.passages("   ") == []


def test_knowledge_base_unavailable_without_embedder_or_chunks():
    assert not KnowledgeBase(VectorStore(), None).available
    assert not KnowledgeBase(VectorStore(), FakeEmbedder()).available


@pytest.mark.parametrize("k", [1, 2, 3])
def test_search_returns_at_most_k(k):
    embedder = FakeEmbedder()
    chunks = chunk_document(DOC, "geo.md")
    vectors = np.asarray([FakeEmbedder._vec(c.text) for c in chunks], dtype=np.float32)
    store = VectorStore(chunks, vectors, embedder.model)
    assert len(store.search(FakeEmbedder._vec("earthquake"), k=k)) == k
