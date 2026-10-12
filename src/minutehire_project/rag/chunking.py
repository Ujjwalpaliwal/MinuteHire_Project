"""Header-aware markdown chunking.

Career documents are structured with markdown headings. Splitting on headings first keeps
each chunk topically coherent, which improves retrieval precision compared to blind
fixed-size splitting. Within a section we pack paragraphs up to a target size and add a
small overlap so that context spanning a boundary is not lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    section: str
    index: int


def _words(text: str) -> list[str]:
    return text.split()


def _split_long_block(block: str, max_words: int, overlap: int) -> list[str]:
    """Split a single oversized paragraph/section into overlapping word windows."""
    words = _words(block)
    if len(words) <= max_words:
        return [block] if block.strip() else []
    pieces: list[str] = []
    start = 0
    step = max(1, max_words - overlap)
    while start < len(words):
        pieces.append(" ".join(words[start : start + max_words]))
        start += step
    return pieces


def _pack_paragraphs(paragraphs: list[str], max_words: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(_words(para))
        if para_len > max_words:
            # Flush current buffer, then emit the long paragraph as its own windows.
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            chunks.extend(_split_long_block(para, max_words, overlap))
            continue
        if current_len + para_len > max_words and current:
            chunks.append("\n\n".join(current))
            # Overlap: carry trailing words of the previous chunk forward.
            tail = " ".join(_words("\n\n".join(current))[-overlap:]) if overlap else ""
            current = [tail] if tail else []
            current_len = len(_words(tail))
        current.append(para)
        current_len += para_len
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in (c.strip() for c in chunks) if c]


def chunk_markdown(
    text: str,
    source: str,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """Chunk one markdown document into a list of overlapping, section-aware chunks."""
    lines = text.splitlines()

    doc_title = source
    sections: list[tuple[str, str]] = []  # (section_heading, body)
    current_heading = ""
    buffer: list[str] = []

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1 and not doc_title_was_set(doc_title, source):
                doc_title = heading
                continue
            if buffer:
                sections.append((current_heading, "\n".join(buffer)))
                buffer = []
            current_heading = heading
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer)))

    chunks: list[Chunk] = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        # Prepend the heading to each piece so the chunk is self-describing.
        paragraphs = [p for p in body.split("\n\n")]
        for piece in _pack_paragraphs(paragraphs, chunk_size, overlap):
            labelled = f"{heading}\n{piece}" if heading else piece
            chunks.append(
                Chunk(
                    text=labelled,
                    source=source,
                    title=doc_title,
                    section=heading,
                    index=len(chunks),
                )
            )
    return chunks


def doc_title_was_set(doc_title: str, source: str) -> bool:
    return doc_title != source
