"""
Nạp dữ liệu từ file vào Qdrant, dùng Colab (GPU T4) để tính embedding.

Chạy:
    python ingest.py

Hỗ trợ .txt, .md trực tiếp. Yêu cầu Colab đang chạy (xem README.md).
"""

import sys

from config import FILE_PATH
from colab_client import embed_texts, check_colab_health
from vectorstore import upsert_chunks, clear_collection


def load_chunks(file_path: str) -> list[str]:
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def main():
    print("[ingest] Kiểm tra kết nối Colab...")
    if not check_colab_health():
        print("[ingest] KHÔNG kết nối được Colab. Kiểm tra lại COLAB_URL")
        print("         trong config.py và đảm bảo Colab notebook đang chạy.")
        sys.exit(1)
    print("[ingest] Colab OK.")

    if "--reset" in sys.argv:
        clear_collection()

    chunks = load_chunks(FILE_PATH)
    print(f"[ingest] Đọc được {len(chunks)} chunks từ '{FILE_PATH}'.")

    batch_size = 12   # Colab có GPU nên dùng batch lớn hơn bản CPU
    total_batches = (len(chunks) - 1) // batch_size + 1

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = embed_texts(batch, is_query=False)

        payloads = [
            {"text": chunk, "source": FILE_PATH, "chunk_index": i + j}
            for j, chunk in enumerate(batch)
        ]

        upsert_chunks(vectors, payloads)
        print(f"[ingest] Đã upsert batch {i // batch_size + 1}/{total_batches}")

    print(f"[ingest] Hoàn tất. {len(chunks)} chunks đã có trong Qdrant.")
    print("[ingest] Xem lại nội dung đã embed bằng: python viewer.py")


if __name__ == "__main__":
    main()
