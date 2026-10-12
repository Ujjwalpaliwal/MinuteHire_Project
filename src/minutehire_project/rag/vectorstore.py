"""ChromaDB persistent vector store wrapper.

Chroma is embedded (no separate server), persists to disk, and supports cosine similarity
search. It is intentionally swappable: the retriever depends on this module only, so moving
to Pinecone/FAISS later means reimplementing these few functions.
"""

from __future__ import annotations

import threading

from minutehire_project.config import settings

_client = None
_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                import chromadb

                settings.chroma_dir.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return _client


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


def upsert_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    collection = get_collection()
    batch = 256
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )


def reset_collection() -> None:
    client = get_client()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    get_collection()


def query(
    embedding: list[float], n_results: int
) -> tuple[list[str], list[dict], list[float]]:
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return documents, metadatas, distances
