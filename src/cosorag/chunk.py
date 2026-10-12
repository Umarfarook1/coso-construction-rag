"""The unit of retrieval and citation. Every chunk knows exactly where it came from."""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str                      # stable id, e.g. "gcc:10CC:2", "bsr:S:1.8.2", "nit:p3:b4"
    doc: str                     # gcc | bsr | nit
    pdf_page: int                # 1-based page in the source PDF (what a viewer jumps to)
    page_label: str              # page number as printed on the page ("28", "S-21"); may differ from pdf_page
    locator: str                 # human citation target: "Clause 10CC", "Item S-1.8.2", "Para 11 (fee table)"
    text: str                    # display text, original language
    index_text: str = ""         # text the retrievers see (adds breadcrumbs / English gloss); defaults to text
    title: str = ""              # clause title, BSR chapter path, notice section
    method: str = "native"       # native | ocr | vlm  (how the text was obtained)
    bboxes: list[list[float]] = field(default_factory=list)  # [x0,y0,x1,y1] in PDF points on pdf_page
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.index_text:
            self.index_text = self.text

    def cite(self) -> str:
        from .config import DOCS

        page = f"p. {self.page_label}" if self.page_label and self.page_label != str(self.pdf_page) else f"p. {self.pdf_page}"
        if self.page_label and self.page_label != str(self.pdf_page):
            page += f" (PDF p. {self.pdf_page})"
        return f"{DOCS[self.doc]['title']}, {self.locator}, {page}"

    def label(self) -> str:
        """cite() for people: no repeated page when the locator already starts with it ("p. 3, condition 1").
        The prompt keeps cite() so the evaluated runs stay reproducible from the cache."""
        c = self.cite()
        return c.rsplit(", ", 1)[0] if self.locator.startswith(f"p. {self.pdf_page},") else c


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(path, "wt", encoding="utf-8") if path.suffix == ".gz" else path.open("w", encoding="utf-8")) as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
