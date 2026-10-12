"""Retrieval: embed the query and fetch the most relevant, deduplicated chunks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from minutehire_project.config import settings
from minutehire_project.rag import vectorstore
from minutehire_project.rag.embeddings import embed_query


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    section: str
    score: float  # cosine similarity in [0, 1], higher is better


def _search(query: str, top_k: int) -> list[RetrievedChunk]:
    embedding = embed_query(query)
    # Over-fetch candidates, then deduplicate and trim.
    n = max(top_k, settings.candidate_k)
    documents, metadatas, distances = vectorstore.query(embedding, n_results=n)

    seen: set[str] = set()
    results: list[RetrievedChunk] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        key = doc[:200]
        if key in seen:
            continue
        seen.add(key)
        # Chroma cosine distance = 1 - cosine similarity.
        similarity = 1.0 - float(dist)
        results.append(
            RetrievedChunk(
                text=doc,
                source=(meta or {}).get("source", "unknown"),
                title=(meta or {}).get("title", ""),
                section=(meta or {}).get("section", ""),
                score=max(0.0, min(1.0, similarity)),
            )
        )
        if len(results) >= top_k:
            break
    return results


async def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Async wrapper so blocking embedding + search never stalls the event loop."""
    k = top_k or settings.top_k
    return await asyncio.to_thread(_search, query, k)
