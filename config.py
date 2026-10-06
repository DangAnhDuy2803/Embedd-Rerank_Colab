"""
Cấu hình chung. Các thông tin nhạy cảm được đọc từ file .env
(xem .env.example để biết các biến cần thiết).
"""

import os
from dotenv import load_dotenv

# Tự động tìm và load file .env ở thư mục gốc của project
load_dotenv()

# ── Model bge-m3 / bge-reranker-v2-m3 — chạy trên Google Colab (GPU T4) ──
# URL này THAY ĐỔI mỗi lần bạn chạy lại Colab notebook (Cell 5) -> nhớ
# cập nhật lại trong file .env mỗi lần đó.
COLAB_URL = os.getenv("COLAB_URL", "https://xxxx-xxxx.trycloudflare.com")

# Phải TRÙNG với API_KEY đặt trong Colab (colab/colab_server.py, Cell 4)
COLAB_API_KEY = os.getenv("COLAB_API_KEY")

RAG_REQUEST_TIMEOUT = 30   # giây, chờ phản hồi từ Colab

# ── Gemini API (sinh câu trả lời cuối cùng) ─────────────────────────
# Lấy API key miễn phí tại: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# ── Dữ liệu nguồn ────────────────────────────────────────────────────
FILE_PATH = "sample.txt"

# ── Qdrant (vector database — self-host bằng Docker, xem
#    docker-compose-qdrant.yml). Chỉ Qdrant chạy trong Docker; 2 model
#    embedding/reranker chạy trên Colab (không phải máy cá nhân, không
#    cần Docker cho chúng). ─────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "colab_rag_documents")
EMBEDDING_DIM = 1024   # bge-m3 dense vector có 1024 chiều

# ── Tham số truy xuất (retrieval) ───────────────────────────────────
TOP_N_CANDIDATE = 20   # số ứng viên lấy ra từ Qdrant trước khi rerank
TOP_K_FINAL = 4         # số đoạn tốt nhất giữ lại sau rerank, đưa cho Gemini
