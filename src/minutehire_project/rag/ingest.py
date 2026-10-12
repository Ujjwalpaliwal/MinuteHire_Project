"""Ingestion: load knowledge-base documents, chunk them, embed and index them.

Run via `python -m minutehire_project.rag.ingest` or the helper script
`scripts/build_index.py`. The FastAPI app can also call `ensure_index()` at startup so a
fresh deploy self-populates.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from minutehire_project.config import settings
from minutehire_project.rag.chunking import chunk_markdown
from minutehire_project.rag.embeddings import embed_texts
from minutehire_project.rag.vectorstore import (
    collection_count,
    reset_collection,
    upsert_chunks,
)

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


def load_documents(kb_dir: Path) -> list[tuple[str, str]]:
    """Return a list of (source_name, full_text)."""
    docs: list[tuple[str, str]] = []
    if not kb_dir.exists():
        logger.warning("Knowledge base directory does not exist: %s", kb_dir)
        return docs
    for path in sorted(kb_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{source}::{index}::{digest}"


def build_index(rebuild: bool = False) -> int:
    """Build (or rebuild) the vector index. Returns the number of chunks indexed."""
    docs = load_documents(settings.kb_dir)
    if not docs:
        return 0

    reset_collection()

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for source, text in docs:
        for chunk in chunk_markdown(
            text,
            source=source,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        ):
            ids.append(_chunk_id(source, chunk.index, chunk.text))
            texts.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source,
                    "title": chunk.title,
                    "section": chunk.section,
                }
            )

    if not texts:
        return 0

    logger.info("Embedding %d chunks from %d documents...", len(texts), len(docs))
    embeddings = embed_texts(texts)
    upsert_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info("Index built with %d chunks.", len(texts))
    return len(texts)


def ensure_index() -> int:
    """Build the index if the collection is empty. Cheap no-op otherwise."""
    count = collection_count()
    if count == 0:
        logger.info("Vector collection empty; building index from knowledge base.")
        return build_index()
    logger.info("Vector collection already has %d chunks.", count)
    return count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    total = build_index(rebuild=True)
    print(f"Indexed {total} chunks into '{settings.collection_name}'.")


if __name__ == "__main__":
    main()
