from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.config.settings import settings


_LOCAL_CLIENT: QdrantClient | None = None
_LOCAL_PATH_CLIENTS: dict[str, QdrantClient] = {}


def get_client() -> QdrantClient:
    global _LOCAL_CLIENT
    if settings.qdrant_url == ":memory:":
        if _LOCAL_CLIENT is None:
            _LOCAL_CLIENT = QdrantClient(":memory:")
        return _LOCAL_CLIENT
    if settings.qdrant_url.startswith("local:"):
        path = settings.qdrant_url.removeprefix("local:")
        if path not in _LOCAL_PATH_CLIENTS:
            _LOCAL_PATH_CLIENTS[path] = QdrantClient(path=path)
        return _LOCAL_PATH_CLIENTS[path]
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    collections = client.get_collections().collections
    exists = any(c.name == settings.collection_name for c in collections)
    if exists:
        return
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
