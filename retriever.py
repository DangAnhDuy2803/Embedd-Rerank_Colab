"""
Pipeline: embed câu hỏi (Colab) -> tìm trong Qdrant -> rerank (Colab).
"""

from config import TOP_N_CANDIDATE, TOP_K_FINAL
from colab_client import embed_texts, rerank as colab_rerank
from vectorstore import search as qdrant_search


def retrieve(query: str) -> list[dict]:
    query_vector = embed_texts([query], is_query=True)[0]

    candidates = qdrant_search(query_vector, top_k=TOP_N_CANDIDATE)
    if not candidates:
        return []

    documents = [c["text"] for c in candidates]
    reranked = colab_rerank(query, documents, top_k=TOP_K_FINAL)

    return [
        {
            "text": candidates[idx]["text"],
            "source": candidates[idx]["source"],
            "score": score,
        }
        for idx, score in reranked
    ]
