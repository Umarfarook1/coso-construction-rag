"""Render a cited page with the cited region highlighted, so a reader can check a citation in one glance."""
from __future__ import annotations

from pathlib import Path

from .chunk import Chunk
from .config import DOCS, RAW_DIR, ROOT


def render(chunk: Chunk, out_dir: Path | None = None, dpi: int = 110) -> Path:
    import pymupdf

    pdf = RAW_DIR / DOCS[chunk.doc]["file"]
    if not pdf.exists():
        raise SystemExit(f"{pdf.name} not found: run `uv run --with gdown python scripts/download_corpus.py` to view pages")
    doc = pymupdf.open(pdf)
    page = doc[chunk.pdf_page - 1]
    for b in chunk.bboxes:
        r = pymupdf.Rect(*b) + (-3, -3, 3, 3)
        page.draw_rect(r, color=(0.95, 0.45, 0.0), width=2)
        page.draw_rect(r, color=None, fill=(1, 0.85, 0.3), fill_opacity=0.18, overlay=True)
    out_dir = out_dir or ROOT / "runs" / "views"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chunk.id.replace(':', '_').replace('/', '_')}.png"
    page.get_pixmap(dpi=dpi).save(path)
    return path
