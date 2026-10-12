"""Route every page of the whole COSO corpus: trust the text layer, or re-OCR? (needs the PDFs)
  uv run python scripts/check_text_layers.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cosorag.config import RAW_DIR  # noqa: E402
from cosorag.textlayer import quality  # noqa: E402


def main() -> None:
    import pymupdf

    report = {}
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        doc = pymupdf.open(pdf)
        reasons, routes = Counter(), Counter()
        for page in doc:
            q = quality(page.get_text())
            routes[q["route"]] += 1
            reasons.update(q["reasons"])
        report[pdf.name] = {"pages": doc.page_count, "ocr": routes["ocr"], "reasons": dict(reasons)}
        print(f"{pdf.name:58} {doc.page_count:4} pages | re-OCR {routes['ocr']:4} | {dict(reasons)}")
    out = ROOT / "evals" / "results" / "text_layers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
