"""
Xem lại nội dung các chunk đã được embed và lưu trong Qdrant.

Cách dùng:
    python viewer.py                  -> in danh sách chunk ra terminal
    python viewer.py --export         -> xuất ra file HTML để xem trên trình duyệt
    python viewer.py --search "từ khóa"  -> chỉ hiển thị chunk chứa từ khóa (lọc text thường, không phải semantic search)
"""

import sys
import html as html_lib
import pathlib
import webbrowser

from vectorstore import scroll_all, count_points


def print_to_terminal(chunks: list[dict], keyword: str | None = None):
    if keyword:
        keyword_lower = keyword.lower()
        chunks = [c for c in chunks if keyword_lower in c["text"].lower()]

    if not chunks:
        print("Không có chunk nào để hiển thị.")
        return

    chunks = sorted(chunks, key=lambda c: (c["source"], c["chunk_index"]))

    print(f"\n{'='*70}")
    print(f" Tổng số chunk: {len(chunks)}")
    print(f"{'='*70}\n")

    for c in chunks:
        print(f"--- [source: {c['source']}] [chunk #{c['chunk_index']}] (id: {c['id']}) ---")
        print(c["text"])
        print()


def export_to_html(chunks: list[dict], out_path: str = "chunks_view.html"):
    chunks = sorted(chunks, key=lambda c: (c["source"], c["chunk_index"]))

    rows = ""
    for c in chunks:
        text_escaped = html_lib.escape(c["text"]).replace("\n", "<br>")
        rows += f"""
        <div class="chunk">
            <div class="meta">Nguồn: <b>{html_lib.escape(c['source'])}</b> &nbsp;|&nbsp; Chunk #{c['chunk_index']} &nbsp;|&nbsp; id: {c['id']}</div>
            <div class="text">{text_escaped}</div>
        </div>
        """

    page = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Xem nội dung chunk đã embed</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 20px; }}
  .summary {{ color: #666; margin-bottom: 24px; }}
  .chunk {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; }}
  .meta {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
  .text {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; }}
</style>
</head>
<body>
  <h1>Nội dung các chunk đã embed vào Qdrant</h1>
  <div class="summary">Tổng số chunk: {len(chunks)}</div>
  {rows}
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    abs_path = pathlib.Path(out_path).resolve()
    uri = abs_path.as_uri()  # "file:///D:/..."

    print(f"[viewer] Đã xuất {len(chunks)} chunk ra file: {abs_path}")
    print(f"[viewer] Mở trong trình duyệt: {uri}")
    webbrowser.open(uri)


def main():
    total = count_points()
    print(f"[viewer] Qdrant hiện có {total} chunk.")

    chunks = scroll_all(limit=max(total, 1))

    if "--export" in sys.argv:
        export_to_html(chunks)
        return

    keyword = None
    if "--search" in sys.argv:
        idx = sys.argv.index("--search")
        if idx + 1 < len(sys.argv):
            keyword = sys.argv[idx + 1]

    print_to_terminal(chunks, keyword=keyword)


if __name__ == "__main__":
    main()
