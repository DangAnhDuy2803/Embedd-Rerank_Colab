"""
Client gọi model bge-m3 (embedding) và bge-reranker-v2-m3 (rerank) đang
được host trên Google Colab (GPU T4), thay vì chạy trực tiếp trên CPU
máy cá nhân.

Colab chỉ đóng vai trò "GPU worker" tạm thời: mỗi lần notebook được
chạy lại, URL public sẽ đổi -> cập nhật COLAB_URL trong config.py.
"""

import httpx

from config import COLAB_URL, COLAB_API_KEY, RAG_REQUEST_TIMEOUT


def _headers() -> dict:
    return {"x-api-key": COLAB_API_KEY} if COLAB_API_KEY else {}


def check_colab_health() -> bool:
    try:
        resp = httpx.get(f"{COLAB_URL}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Gửi danh sách văn bản tới Colab, nhận về dense vectors (bge-m3)."""
    if not texts:
        return []

    with httpx.Client(timeout=RAG_REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{COLAB_URL}/embed",
            json={"texts": texts, "is_query": is_query},
            headers=_headers(),
        )
        resp.raise_for_status()
        return resp.json()["dense_vectors"]


def rerank(query: str, documents: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
    """
    Gọi Colab /rerank. Trả về list (index_trong_documents, score) đã
    sắp xếp giảm dần theo độ liên quan.
    """
    if not documents:
        return []

    with httpx.Client(timeout=RAG_REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{COLAB_URL}/rerank",
            json={"query": query, "documents": documents, "top_k": top_k},
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

    return [(idx, data["scores"][idx]) for idx in data["ranked_indices"]]
