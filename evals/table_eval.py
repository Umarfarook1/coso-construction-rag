"""Schedule-of-rates row extraction accuracy against 63 hand-checked rows (evals/gold/bsr_rows_gt.json).
A row is correct when item number, unit and rate match exactly and the row's own description starts with the
expected words and ends with the expected last word (so multi-line descriptions are captured whole).
Needs the PDF (scripts/download_corpus.py).
  uv run python evals/table_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cosorag.config import DOCS, RAW_DIR  # noqa: E402
from cosorag.ingest import bsr  # noqa: E402


def main() -> None:
    gt = json.loads((ROOT / "evals/gold/bsr_rows_gt.json").read_text(encoding="utf-8"))
    items, _ = bsr.parse_items(RAW_DIR / DOCS["bsr"]["file"])
    by_page: dict[int, dict[str, bsr.Item]] = {}
    for it in items:
        by_page.setdefault(it.pdf_page, {})[it.no] = it
    ok = total = 0
    fields = {"found": 0, "unit": 0, "rate": 0, "description": 0}
    misses = []
    for page, rows in gt.items():
        for no, unit, rate, first, last in rows:
            total += 1
            it = by_page.get(int(page), {}).get(no)
            if it is None:
                misses.append(f"p{page} {no}: not found"); continue
            fields["found"] += 1
            u = it.unit.rstrip(".") == unit.rstrip(".")
            r = it.rate == rate
            d = it.own.startswith(first) and it.own.rstrip(". ").endswith(last.rstrip("."))
            fields["unit"] += u; fields["rate"] += r; fields["description"] += d
            if u and r and d:
                ok += 1
            else:
                misses.append(f"p{page} {no}: unit={it.unit!r} rate={it.rate!r} desc={it.own[:60]!r}")
    print(f"rows fully correct: {ok}/{total}  |  " + ", ".join(f"{k} {v}/{total}" for k, v in fields.items()))
    for m in misses:
        print("  ", m)


if __name__ == "__main__":
    main()
