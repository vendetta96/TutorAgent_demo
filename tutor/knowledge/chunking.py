"""Split Markdown / plain-text documents into retrieval chunks. Pure and deterministic."""

import hashlib
import re
from dataclasses import dataclass, asdict

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    heading: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(id=data["id"], source=data["source"], heading=data["heading"], text=data["text"])


def chunk_id(source: str, heading: str, text: str) -> str:
    digest = hashlib.sha1(f"{source}\n{heading}\n{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _sections(text: str) -> list[tuple[str, str]]:
    """Split on Markdown headings; returns (heading, body) pairs in order."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            sections.append((match.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(body).strip()) for heading, body in sections if "\n".join(body).strip()]


def _paragraphs(body: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def chunk_document(text: str, source: str, *, max_chars: int = 900, overlap_paragraphs: int = 1) -> list[Chunk]:
    """Chunk by heading, then pack paragraphs up to max_chars with a small paragraph overlap."""
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        paragraphs = _paragraphs(body)
        i = 0
        while i < len(paragraphs):
            packed: list[str] = []
            size = 0
            j = i
            while j < len(paragraphs) and (size + len(paragraphs[j]) <= max_chars or not packed):
                packed.append(paragraphs[j])
                size += len(paragraphs[j]) + 2
                j += 1
            chunk_text = "\n\n".join(packed)
            if len(chunk_text) > max_chars * 2:
                # A single oversized paragraph: hard-split on sentence boundaries.
                for piece in _split_long(chunk_text, max_chars):
                    chunks.append(Chunk(chunk_id(source, heading, piece), source, heading, piece))
            else:
                chunks.append(Chunk(chunk_id(source, heading, chunk_text), source, heading, chunk_text))
            if j >= len(paragraphs):
                break
            i = max(i + 1, j - overlap_paragraphs)
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces
