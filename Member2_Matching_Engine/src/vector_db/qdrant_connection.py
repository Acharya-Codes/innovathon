from __future__ import annotations

import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "internships")


def get_qdrant_client(url: Optional[str] = None) -> QdrantClient:
    endpoint = url or QDRANT_URL
    return QdrantClient(url=endpoint)


def is_qdrant_reachable(client: Optional[QdrantClient] = None) -> bool:
    local_client = client or get_qdrant_client()
    try:
        local_client.get_collections()
        return True
    except Exception:
        return False


def ensure_collection(
    collection_name: str = COLLECTION_NAME,
    vector_size: int = 384,
    distance: str = "Cosine",
    client: Optional[QdrantClient] = None,
) -> QdrantClient:
    local_client = client or get_qdrant_client()
    try:
        collections = local_client.get_collections().collections
        existing_names = {collection.name for collection in collections}
        if collection_name in existing_names:
            try:
                info = local_client.get_collection(collection_name)
                params = getattr(info.config, "params", None)
                vectors = getattr(params, "vectors", None)
                if vectors is not None:
                    current_size = getattr(vectors, "size", None)
                    current_distance = getattr(vectors, "distance", None)
                    if current_size == vector_size and str(current_distance).lower() == str(models.Distance.COSINE).lower():
                        return local_client
            except Exception:
                pass

            try:
                local_client.delete_collection(collection_name=collection_name)
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - connection-level error
        raise RuntimeError(f"Unable to reach Qdrant at {QDRANT_URL}: {exc}") from exc

    try:
        local_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    except Exception as exc:
        if "already exists" in str(exc).lower():
            return local_client
        raise RuntimeError(f"Could not create Qdrant collection '{collection_name}': {exc}") from exc
    return local_client
