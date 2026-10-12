"""Cross-check the schedule-of-rates parser against an independent parser over the whole document.

Two parsers built differently (anchor rows vs header-box column edges) are compared on every (page, item number)
both extract. Agreement on unit and rate is evidence beyond the 63 hand-checked rows; disagreements are listed
so they can be checked against the page image.
  uv run python evals/bsr_crosscheck.py [--show 15]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals" / "ref_parser"))

import pymupdf  # noqa: E402

from bsr_tables import header_edges, m_header_cols, merge  # noqa: E402
from cosorag.config import DOCS, RAW_DIR  # noqa: E402
from cosorag.ingest import bsr  # noqa: E402


def norm_unit(u: str) -> str:
    return u.lower().replace(".", "").replace(" ", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()
    pdf = RAW_DIR / DOCS["bsr"]["file"]
    items, _ = bsr.parse_items(pdf)
    mine = {(it.pdf_page, it.no): it for it in items if it.rate}
    doc = pymupdf.open(pdf)
    both = agree_rate = agree_unit = only_ref = 0
    diffs = []
    for i, page in enumerate(doc):
        if not page.get_text().strip() or len(header_edges(page)) < 5:
            continue
        for r in merge(m_header_cols(page)):
            no, unit, rate = r[0], r[2], r[3]
            if not no or not rate:
                continue
            it = mine.get((i + 1, no))
            if it is None:
                only_ref += 1
                continue
            both += 1
            ok_r = it.rate == rate or rate in it.rate.split()
            ok_u = norm_unit(it.unit) == norm_unit(unit) or norm_unit(unit) in norm_unit(it.unit)
            agree_rate += ok_r
            agree_unit += ok_u
            if not (ok_r and ok_u):
                diffs.append({"page": i + 1, "label": it.page_label, "item": no, "mine": [it.unit, it.rate], "ref": [unit, rate]})
    out = {"rows_both": both, "rate_agree": agree_rate, "unit_agree": agree_unit, "ref_only_rows": only_ref,
           "mine_rated_rows": len(mine), "disagreements": len(diffs)}
    print(json.dumps(out))
    random.Random(0).shuffle(diffs)
    for d in diffs[: a.show]:
        print(d)
    (ROOT / "evals" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "evals" / "results" / "bsr_crosscheck.json").write_text(json.dumps({**out, "sample": diffs[:40]}, indent=1))


if __name__ == "__main__":
    main()
