"""
Gọi Gemini API (google-genai SDK) để sinh câu trả lời tự nhiên, dựa
trên các đoạn văn bản đã lấy được từ bước retrieve + rerank.
Hỗ trợ conversation history để chatbot nhớ các lượt hỏi-đáp trước.
"""

from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


PROMPT_TEMPLATE = """\
Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp.
Chỉ trả lời dựa trên nội dung trong phần "Tài liệu tham khảo" bên dưới.
Nếu tài liệu không chứa thông tin liên quan, hãy nói rõ là không tìm
thấy thông tin, không tự bịa ra câu trả lời.
Khi trả lời, được phép tham chiếu tới các lượt hội thoại trước nếu có liên quan.

Tài liệu tham khảo:
{context}

{history_block}Câu hỏi hiện tại: {question}

Trả lời (ngắn gọn, rõ ràng, bằng tiếng Việt):"""

_HISTORY_HEADER = "Lịch sử hội thoại (để tham khảo ngữ cảnh):\n"


def _format_history(history: list[dict]) -> str:
    """Chuyển list [{role, content}] thành chuỗi văn bản cho prompt."""
    if not history:
        return ""
    lines = [_HISTORY_HEADER]
    for i, turn in enumerate(history, start=1):
        role_label = "Người dùng" if turn["role"] == "user" else "Trợ lý"
        lines.append(f"  [{i}] {role_label}: {turn['content']}")
    lines.append("")  # dòng trống phân cách
    return "\n".join(lines) + "\n"


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    history: list[dict] | None = None,
) -> str:
    """
    Sinh câu trả lời từ Gemini.

    Args:
        question: Câu hỏi hiện tại của người dùng.
        retrieved_chunks: Các đoạn văn bản lấy từ Qdrant + rerank.
        history: Danh sách các lượt hội thoại trước, mỗi lượt là dict
                 {"role": "user"|"assistant", "content": "..."}
    """
    if not retrieved_chunks:
        return "Không tìm thấy thông tin liên quan trong tài liệu."

    context = "\n\n".join(
        f"[Đoạn {i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks)
    )

    history_block = _format_history(history or [])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        history_block=history_block,
        question=question,
    )

    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text
