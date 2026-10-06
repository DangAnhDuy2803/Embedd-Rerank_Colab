"""
Chạy chương trình chính: hỏi-đáp tương tác tại terminal.

    python main.py

YÊU CẦU TRƯỚC KHI CHẠY:
    1. Colab notebook (colab/colab_server.py) đang chạy, COLAB_URL
       trong config.py đã được cập nhật đúng URL hiện tại.
    2. Qdrant đang chạy (docker compose -f docker-compose-qdrant.yml up -d)
    3. Đã chạy `python ingest.py` ít nhất 1 lần để nạp dữ liệu.

Luồng xử lý cho mỗi câu hỏi:
    1. Embed câu hỏi (bge-m3, qua Colab GPU)
    2. Tìm trong Qdrant -> lọc top ứng viên
    3. Rerank (bge-reranker-v2-m3, qua Colab GPU) -> top K chính xác nhất
    4. Đưa top K đoạn văn + lịch sử hội thoại + câu hỏi vào Gemini -> sinh câu trả lời tự nhiên

Lệnh đặc biệt trong session:
    history  — xem lại toàn bộ lịch sử hội thoại (kể cả các lần chạy trước)
    clear    — xóa lịch sử hội thoại khỏi bộ nhớ VÀ file JSON
    exit     — thoát chương trình
"""

import sys
import json
import pathlib
from datetime import datetime

from src.colab_client import check_colab_health
from src.vectorstore import collection_is_empty, count_points
from src.retriever import retrieve
from src.gemini_client import generate_answer

# Số lượt hội thoại tối đa được đưa vào prompt (để tránh prompt quá dài)
MAX_HISTORY_TURNS = 10

# File JSON lưu lịch sử hội thoại bền vững qua các lần chạy
HISTORY_DIR = pathlib.Path("history")
HISTORY_FILE = HISTORY_DIR / "chat_history.json"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)  # tự tạo folder nếu chưa có


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    """Đọc lịch sử từ file JSON. Trả về list rỗng nếu file chưa tồn tại."""
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        print(f"[!] Không đọc được {HISTORY_FILE}, bắt đầu lịch sử mới.")
    return []


def save_history(history: list[dict]) -> None:
    """Ghi toàn bộ history vào file JSON (ghi đè)."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"[!] Không ghi được lịch sử: {e}")


def print_history(history: list[dict]) -> None:
    """In lịch sử hội thoại ra terminal."""
    if not history:
        print("  (Chưa có lịch sử hội thoại nào.)")
        return
    print()
    for i in range(0, len(history), 2):
        turn_num = i // 2 + 1
        user_entry = history[i]
        user_msg = user_entry["content"]
        ts = user_entry.get("timestamp", "")
        ts_label = f" ({ts})" if ts else ""
        assistant_msg = (
            history[i + 1]["content"] if i + 1 < len(history) else "(chưa có trả lời)"
        )
        print(f"  [{turn_num}]{ts_label}")
        print(f"  ❓ {user_msg}")
        print(f"  💬 {assistant_msg}")
        print()


def main():
    print("=" * 60)
    print(" Kiểm tra kết nối Colab...")
    print("=" * 60)

    if not check_colab_health():
        print("\n[!] Không kết nối được tới Colab.")
        print("[!] Kiểm tra: Colab notebook có đang chạy không?")
        print("[!] Kiểm tra: COLAB_URL trong config.py có đúng không?")
        print("    (URL đổi mỗi lần bạn chạy lại Colab)")
        sys.exit(1)

    print("[✓] Colab đang hoạt động.\n")

    if collection_is_empty():
        print("[!] Qdrant chưa có dữ liệu nào.")
        print("[!] Hãy chạy `python ingest.py` trước, rồi chạy lại main.py.")
        sys.exit(1)

    print(f"[✓] Qdrant có {count_points()} chunk sẵn sàng.\n")

    # Load lịch sử từ file JSON (nếu có từ lần chạy trước)
    conversation_history: list[dict] = load_history()
    if conversation_history:
        turns = len(conversation_history) // 2
        print(f"[✓] Đã tải {turns} lượt hội thoại từ '{HISTORY_FILE}'.\n")

    print("=" * 60)
    print(" Sẵn sàng! Gõ câu hỏi và Enter.")
    print(" Lệnh đặc biệt: 'history' | 'clear' | 'exit'")
    print(f" Lịch sử được lưu vào: {HISTORY_FILE.resolve()}")
    print("=" * 60)

    while True:
        query = input("\n❓ Câu hỏi của bạn: ").strip()

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Tạm biệt!")
            break

        # Xem lịch sử hội thoại
        if query.lower() == "history":
            print("\n📜 Lịch sử hội thoại:")
            print_history(conversation_history)
            continue

        # Xóa lịch sử khỏi bộ nhớ VÀ file JSON
        if query.lower() == "clear":
            conversation_history.clear()
            save_history(conversation_history)
            print("🗑️  Đã xóa lịch sử hội thoại. Bắt đầu cuộc trò chuyện mới.")
            continue

        try:
            print("\n[1/2] Đang tìm đoạn liên quan (Qdrant + rerank qua Colab)...")
            results = retrieve(query)

            print("--- Các đoạn được chọn ---")
            for r in results:
                print(f"  [{r['score']:.4f}] {r['text'][:80]}...")

            print("\n[2/2] Đang gọi Gemini để sinh câu trả lời...")

            # Chỉ truyền MAX_HISTORY_TURNS lượt gần nhất vào prompt
            recent_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]
            answer = generate_answer(query, results, history=recent_history)

            print("\n💬 Trả lời:")
            print(answer)

            # Lưu lượt hội thoại vừa xong vào history (kèm timestamp)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_history.append({"role": "user", "content": query, "timestamp": now})
            conversation_history.append({"role": "assistant", "content": answer, "timestamp": now})

            # Ghi ngay vào file JSON sau mỗi lượt (không chờ đến khi exit)
            save_history(conversation_history)

        except Exception as exc:
            print(f"\n[Lỗi] {exc}")
            print("Gợi ý: Colab có thể đã ngắt phiên hoặc URL đã đổi.")


if __name__ == "__main__":
    main()
