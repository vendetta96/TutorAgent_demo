"""Knowledge ingestion CLI.

    uv run python -m tutor.knowledge.ingest              # ingest ./knowledge into data/knowledge_index.json
    uv run python -m tutor.knowledge.ingest --dir docs   # a different folder
    uv run python -m tutor.knowledge.ingest --query "what causes a tsunami"   # try a search

Re-running is incremental: unchanged chunks keep their embeddings.
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from tutor.knowledge.embedder import OpenAIEmbedder
from tutor.knowledge.store import DEFAULT_INDEX_PATH, KnowledgeBase, VectorStore, ingest_documents

DEFAULT_KNOWLEDGE_DIR = Path("knowledge")
SUPPORTED = {".md", ".markdown", ".txt"}


def read_documents(directory: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            docs[str(path.relative_to(directory)).replace("\\", "/")] = path.read_text(encoding="utf-8")
    return docs


async def run_ingest(directory: Path, index_path: Path, embedder: OpenAIEmbedder) -> None:
    documents = read_documents(directory)
    if not documents:
        print(f"No .md/.txt documents found in {directory}")
        return
    existing = VectorStore.load(index_path)
    store, result = await ingest_documents(documents, embedder, existing)
    store.save(index_path)
    print(
        f"Ingested {len(result.sources)} documents -> {result.total_chunks} chunks "
        f"({result.embedded} embedded, {result.reused} reused, {result.removed} removed). "
        f"Index: {index_path}"
    )


async def run_query(index_path: Path, embedder: OpenAIEmbedder, query: str) -> None:
    kb = KnowledgeBase(VectorStore.load(index_path), embedder, min_score=0.0)
    for hit in await kb.search(query):
        print(f"[{hit.score:.3f}] {hit.chunk.source} / {hit.chunk.heading}\n    {hit.chunk.text[:200]}...")


def main(argv: list[str] | None = None) -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Ingest knowledge documents for the tutor agent.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--query", type=str, default=None, help="Search the index instead of ingesting")
    args = parser.parse_args(argv)

    embedder = OpenAIEmbedder()
    if args.query:
        asyncio.run(run_query(args.index, embedder, args.query))
    else:
        asyncio.run(run_ingest(args.dir, args.index, embedder))


if __name__ == "__main__":
    main()
