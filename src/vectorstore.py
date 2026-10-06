"""
Kết nối tới Qdrant (self-host bằng Docker, xem docker-compose-qdrant.yml).
Qdrant CHỈ lưu trữ + tìm kiếm vector — rất nhẹ, chạy tốt trên máy CPU
yếu. Phần NẶNG (tính embedding, rerank) chạy trên Colab GPU T4 thông
qua colab_client.py.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIM

_client = QdrantClient(url=QDRANT_URL)


def ensure_collection():
    existing = [c.name for c in _client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        return

    _client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"[qdrant] Đã tạo collection '{QDRANT_COLLECTION}'.")


def collection_is_empty() -> bool:
    ensure_collection()
    info = _client.get_collection(QDRANT_COLLECTION)
    return info.points_count == 0


def count_points() -> int:
    ensure_collection()
    return _client.get_collection(QDRANT_COLLECTION).points_count


def upsert_chunks(vectors: list[list[float]], payloads: list[dict]):
    ensure_collection()

    points = [
        qmodels.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
        for vec, payload in zip(vectors, payloads)
    ]
    _client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search(query_vector: list[float], top_k: int) -> list[dict]:
    ensure_collection()

    hits = _client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        {
            "id": hit.id,
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", ""),
            "chunk_index": hit.payload.get("chunk_index", -1),
            "score": hit.score,
        }
        for hit in hits
    ]


def scroll_all(limit: int = 1000) -> list[dict]:
    """Lấy TOÀN BỘ chunk đã lưu (không cần query), dùng cho việc xem lại
    nội dung đã embed — xem viewer.py."""
    ensure_collection()

    points, _ = _client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {
            "id": p.id,
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", ""),
            "chunk_index": p.payload.get("chunk_index", -1),
        }
        for p in points
    ]


def clear_collection():
    """Xoá toàn bộ collection (dùng khi muốn ingest lại từ đầu)."""
    existing = [c.name for c in _client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        _client.delete_collection(QDRANT_COLLECTION)
        print(f"[qdrant] Đã xoá collection '{QDRANT_COLLECTION}'.")
