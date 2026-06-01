from __future__ import annotations

from app.config.settings import settings
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import get_client


def retrieve_context(query: str, top_k: int | None = None) -> list[str]:
    client = get_client()
    embedder = EmbeddingModel()
    vector = embedder.embed_query(query)
    limit = top_k or settings.top_k
    results = client.search(
        collection_name=settings.collection_name,
        query_vector=vector,
        limit=limit,
    )
    return [point.payload.get("text", "") for point in results]
