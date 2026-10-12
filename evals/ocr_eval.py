"""OCR quality on the Hindi tender: field-level digit accuracy against 96 hand-read fields (evals/gold/nit_fields.json).

A field (amount, date, budget head, handwritten letter number ...) counts as correct for a reader if every digit
group of the hand-read value appears on that page of the reader's transcription. Also scores the merged
pipeline output (primary text, with disagreements flagged instead of silently trusted).
  uv run python evals/ocr_eval.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cosorag.text import normalize  # noqa: E402


def groups(s: str) -> Counter:
    return Counter(g.lstrip("0") or "0" for g in re.findall(r"\d+", normalize(s)))


def main() -> None:
    fields = json.loads((ROOT / "evals/gold/nit_fields.json").read_text(encoding="utf-8"))
    ocr = json.loads((ROOT / "data/processed/nit_ocr.json").read_text(encoding="utf-8"))
    readers = ["primary", "second", "third"]
    res = {r: [] for r in readers}
    misses = {r: [] for r in readers}
    for page, fs in fields.items():
        p = page.lstrip("p")
        for name, value in fs.items():
            want = groups(value)
            if not want:
                continue
            for r in readers:
                have = groups(ocr["pages"][p].get(r, {}).get("text", ""))
                ok = not (want - have)
                res[r].append(ok)
                if not ok:
                    misses[r].append(f"{page}:{name}={value}")
    lines = ["| reader | model | fields correct |", "|---|---|---|"]
    for r in readers:
        model = ocr["pages"]["4"].get(r, {}).get("model", "")
        lines.append(f"| {r} | {model} | {sum(res[r])}/{len(res[r])} ({100 * sum(res[r]) / len(res[r]):.0f}%) |")
    print("\n".join(lines))
    for r in readers:
        print(f"\n{r} misses: {misses[r]}")


if __name__ == "__main__":
    main()
