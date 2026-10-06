# RAG: Colab GPU (embed + rerank) + Qdrant + Gemini — hỏi đáp ở terminal

Kiến trúc đầy đủ của bản này:

```
┌──────────────── Máy cá nhân ────────────────┐
│  ingest.py / main.py / viewer.py (Python)   │
│  Qdrant (Docker, vector database)           │
└───────────────────┬──────────────────────────┘
                     │ HTTP (embed, rerank)
                     ▼
        ┌─────────────────────────────┐
        │  Google Colab (GPU T4)       │
        │  bge-m3 + bge-reranker-v2-m3 │
        └─────────────────────────────┘
                     │
                     ▼ (câu trả lời cuối)
            Gemini API (gemini-2.5-flash)
```

- **Embed + rerank**: chạy trên **GPU T4 của Colab** (nhanh), máy cá nhân chỉ gọi HTTP sang — không dùng CPU/GPU máy bạn cho 2 việc này nữa.
- **Qdrant**: tự host bằng Docker trên máy cá nhân — lưu trữ vector bền vững, không mất khi tắt chương trình.
- **Gemini**: sinh câu trả lời tự nhiên dựa trên các đoạn văn bản tìm được.
- **viewer.py**: công cụ mới — xem lại nội dung các chunk đã được embed và lưu trong Qdrant.

---

## Bước 1: Chạy Colab notebook (GPU worker)

1. Vào https://colab.research.google.com → New Notebook
2. `Runtime` → `Change runtime type` → chọn **T4 GPU** → Save
3. Mở file `colab/colab_server.py` trong bộ này. File gồm 6 khối `### CELL n`, mỗi khối nằm trong dấu `"""..."""`. Tạo **6 cell riêng** trong Colab, copy đúng nội dung bên trong dấu `"""` (bỏ dấu `"""`) theo thứ tự:
   - Cell 1: cài thư viện
   - Cell 2: kiểm tra GPU (`CUDA available: True`)
   - Cell 3: load 2 model (lần đầu tải ~3GB, mất vài phút)
   - Cell 4: định nghĩa FastAPI app — **sửa `API_KEY = "..."` thành chuỗi bí mật của bạn**
   - Cell 5: mở tunnel Cloudflare (hoặc Cell 6 nếu dùng ngrok)
4. Chạy lần lượt Cell 1 → 5. Cell 5 in ra:
   ```
   PUBLIC URL: https://xxxx-xxxx.trycloudflare.com
   ```
   Copy URL này lại.
5. **Giữ tab Colab mở** trong lúc dùng.

> Nếu gặp thông báo "Không thể kết nối với phần phụ trợ GPU" (hết hạn mức GPU miễn phí), đợi vài giờ tới 24h rồi thử lại, hoặc tạm bấm "Kết nối mà không cần GPU" để chạy chậm hơn bằng CPU của Colab.

## Bước 2: Cài thư viện ở máy cá nhân

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# hoặc: source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

## Bước 3: Chạy Qdrant (Docker)

```bash
docker compose -f docker-compose-qdrant.yml up -d
```

Kiểm tra: http://localhost:6333/dashboard

## Bước 4: Cấu hình biến môi trường (`.env`)

Sao chép file mẫu và điền giá trị thật:

```bash
copy .env.example .env   # Windows
# hoặc: cp .env.example .env  # macOS/Linux
```

Mở `.env` và sửa các giá trị:

```dotenv
# URL từ Bước 1 — thay đổi mỗi lần chạy lại Colab
COLAB_URL=https://xxxx-xxxx.trycloudflare.com

# Phải TRÙNG với API_KEY đặt trong Colab Cell 4
COLAB_API_KEY=chuoi-bi-mat-cua-ban

# Lấy tại https://aistudio.google.com/apikey
GEMINI_API_KEY=AIza...

# Giữ nguyên nếu chạy Qdrant mặc định trên máy
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=colab_rag_documents
```

> **Quan trọng**: File `.env` đã được thêm vào `.gitignore` — sẽ **không bị commit lên git**.
> Chỉ commit file `.env.example` (không chứa giá trị thật) để chia sẻ với team.

> **Nhắc nhở**: mỗi lần chạy lại Colab (session mới), `COLAB_URL` sẽ đổi — cập nhật lại trong `.env`.

## Bước 5: Nạp dữ liệu vào Qdrant

```bash
python ingest.py
```

Dùng `python ingest.py --reset` nếu muốn xoá dữ liệu cũ và nạp lại từ đầu.

## Bước 6: Xem lại nội dung đã embed (tính năng mới)

```bash
# In ra terminal
python viewer.py

# Xuất ra file HTML đẹp hơn để xem trên trình duyệt
python viewer.py --export
```

Lệnh `--export` tạo ra file `chunks_view.html` — mở bằng trình duyệt để xem từng chunk kèm nguồn file và số thứ tự, dễ đọc hơn terminal.

## Bước 7: Chạy hỏi-đáp

```bash
python main.py
```

Gõ câu hỏi, Enter, xem kết quả. Gõ `exit` để thoát.

---

## Cấu trúc file

| File / Thư mục | Vai trò |
|---|---|
| `.env` | ⚠️ Biến môi trường nhạy cảm — **không commit lên git** |
| `.env.example` | Template biến môi trường — commit lên git để chia sẻ với team |
| `.gitignore` | Loại trừ `.env` và các file không cần thiết khỏi git |
| `config.py` | Đọc biến từ `.env` và định nghĩa các tham số cấu hình |
| `colab/colab_server.py` | Code copy vào Colab — host bge-m3 + reranker qua HTTP |
| `src/` | **Gói mã nguồn backend**: chứa các module xử lý nội bộ |
| ├── `src/colab_client.py` | Gọi `/embed`, `/rerank` trên Colab |
| ├── `src/gemini_client.py` | Gọi Gemini API, ghép prompt kèm ngữ cảnh & lịch sử |
| ├── `src/vectorstore.py` | Wrapper Qdrant (tạo collection, upsert, search, scroll toàn bộ) |
| └── `src/retriever.py` | Pipeline: embed câu hỏi → tìm Qdrant → rerank |
| `main.py` | Chatbot hỏi-đáp ở terminal (hỗ trợ ghi nhớ lịch sử hội thoại) |
| `ingest.py` | Đọc file → gọi Colab embed → lưu vào Qdrant |
| `viewer.py` | Xem lại nội dung các chunk đã embed (terminal hoặc HTML) |
| `history/` | Thư mục lưu lịch sử hội thoại (`chat_history.json`) |
| `docker-compose-qdrant.yml` | Chạy Qdrant bằng Docker |
| `sample.txt` | Dữ liệu mẫu để test ngay |

## Xử lý sự cố

- **`Không kết nối được tới Colab`**: Colab đã ngắt phiên (idle quá 90 phút / hết 12h) hoặc bạn quên cập nhật `COLAB_URL` mới trong `.env` sau khi chạy lại notebook.
- **`401 Unauthorized`**: `COLAB_API_KEY` trong `.env` không khớp với `API_KEY` trong Colab Cell 4.
- **`GEMINI_API_KEY` hoặc `COLAB_API_KEY` trả về `None`**: chưa tạo file `.env` hoặc chưa điền giá trị — chạy `copy .env.example .env` rồi sửa lại.
- **`Qdrant chưa có dữ liệu nào`**: quên chạy `python ingest.py` trước khi chạy `main.py`.
- **`Connection refused` tới Qdrant**: Qdrant chưa chạy — `docker compose -f docker-compose-qdrant.yml up -d`.
- **Colab báo hết hạn mức GPU**: đợi vài giờ tới 24h, đổi tài khoản Google, hoặc tạm dùng chế độ CPU của Colab (chậm hơn nhưng vẫn chạy được).

## Nếu muốn nâng cấp sau này

- **Production ổn định hơn**: thay Colab bằng GPU thuê (RunPod/Vast.ai) — chỉ cần đổi `COLAB_URL` thành URL cố định của GPU thuê, code `colab_client.py` không cần sửa gì (vẫn cùng giao thức HTTP `/embed`, `/rerank`).
- **Ingest nhiều loại file hơn** (.pdf, .docx): thêm thư viện `pypdf`, `python-docx` và mở rộng `load_chunks()` trong `ingest.py`.
- **Tích hợp vào chatbot LangGraph chính**: logic `colab_client.py` + `vectorstore.py` có thể đóng gói thành 1 LangChain `@tool`, y như đã làm ở repo chatbot trước đó.
