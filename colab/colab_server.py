"""
=====================================================================
 COLAB BGE SERVER — bge-m3 (embedding) + bge-reranker-v2-m3 (rerank)
=====================================================================
KHÔNG chạy file này bằng `python colab_server.py` trên máy cá nhân.
File này chỉ để COPY từng khối (### CELL n) vào từng cell của một
Google Colab Notebook (Runtime > Change runtime type > T4 GPU).

Mục đích: dùng GPU T4 miễn phí của Colab để chạy 2 model nặng, máy cá
nhân (CPU yếu) chỉ gửi HTTP request sang và nhận kết quả.

LƯU Ý:
- Colab free tự ngắt sau ~90 phút không tương tác, tối đa ~12h/phiên.
- Mỗi lần chạy lại notebook, URL public sẽ THAY ĐỔI -> phải cập nhật
  lại COLAB_URL trong config.py ở máy cá nhân.
- Có thể bị chặn tạm thời ("hết hạn mức GPU") nếu dùng nhiều trong thời
  gian ngắn -> đợi vài giờ tới 24h, hoặc đổi tài khoản Google.
=====================================================================
"""

# ### CELL 1 — Cài đặt thư viện ######################################
"""
!pip install -q fastapi uvicorn pyngrok nest_asyncio FlagEmbedding \
    sentence-transformers pydantic python-multipart
"""

# ### CELL 2 — Kiểm tra GPU ##########################################
"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""

# ### CELL 3 — Load model (bge-m3 + bge-reranker-v2-m3) ##############
"""
from FlagEmbedding import BGEM3FlagModel, FlagReranker

embed_model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=True,        # fp16 -> nhanh hơn, tiết kiệm VRAM trên T4
    device='cuda',
)

reranker = FlagReranker(
    'BAAI/bge-reranker-v2-m3',
    use_fp16=True,
    device='cuda',
)

print("Models loaded OK.")
"""

# ### CELL 4 — Định nghĩa FastAPI app ################################
"""
import nest_asyncio
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Đặt 1 API key tự chọn (chuỗi bí mật do bạn tự nghĩ ra) để không ai
# khác gọi được server Colab của bạn. Đặt CÙNG giá trị này vào
# COLAB_API_KEY trong config.py ở máy cá nhân.
API_KEY = "doi-cai-nay-thanh-chuoi-bi-mat-cua-ban"

app = FastAPI(title="BGE Colab Server")


def check_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False


class EmbedResponse(BaseModel):
    dense_vectors: List[List[float]]
    dim: int


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None


class RerankResponse(BaseModel):
    scores: List[float]
    ranked_indices: List[int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)

    output = embed_model.encode(
        req.texts,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense = output["dense_vecs"]
    return EmbedResponse(
        dense_vectors=[v.tolist() for v in dense],
        dim=len(dense[0]) if len(dense) else 1024,
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)

    pairs = [[req.query, doc] for doc in req.documents]
    scores = reranker.compute_score(pairs, normalize=True)

    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if req.top_k:
        ranked = ranked[: req.top_k]

    return RerankResponse(scores=scores, ranked_indices=ranked)
"""

# ### CELL 5 — Mở tunnel public (Cloudflare Tunnel, không cần token) #
"""
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64

import subprocess, time, re, threading

def run_server():
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_server, daemon=True).start()
time.sleep(3)

proc = subprocess.Popen(
    ["./cloudflared-linux-amd64", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

for line in proc.stdout:
    print(line, end="")
    match = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", line)
    if match:
        print("\\n==================================================")
        print("PUBLIC URL:", match.group(0))
        print("Copy URL này vào config.py: COLAB_URL = \\"...\\"")
        print("==================================================\\n")
"""

# ### CELL 6 (thay thế Cell 5 nếu Cloudflare bị chặn) ################
"""
from pyngrok import ngrok
import nest_asyncio, threading

ngrok.set_auth_token("NGROK_AUTH_TOKEN_CUA_BAN")  # lấy free tại ngrok.com

nest_asyncio.apply()

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_server, daemon=True).start()

public_url = ngrok.connect(8000)
print("PUBLIC URL:", public_url)
"""
