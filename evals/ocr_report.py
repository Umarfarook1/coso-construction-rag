"""Write evals/results/ocr.md from the bake-off outputs and the field-level check (no model calls)."""
from __future__ import annotations

import io
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "evals" / "ocr_bakeoff"


def capture(script: Path, *args: str) -> str:
    buf = io.StringIO()
    old = sys.argv
    sys.argv = [str(script), *args]
    try:
        with redirect_stdout(buf):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old
    return buf.getvalue()


def main() -> None:
    score = capture(B / "score.py", str(B / "refs"), str(B / "out")).splitlines()[1:]
    out = ["Bake-off on 6 hand-transcribed regions (`evals/ocr_bakeoff/`; engine outputs and references included):", "",
           "| engine | CER | numbers found | word F1 | sec/page |", "|---|---|---|---|---|"]
    for line in score:
        p = [x.strip() for x in line.split(" | ")]
        if len(p) >= 5:
            out.append(f"| {p[0]} | {p[1]} | {p[2]} | {p[3]} | {p[5] if len(p) > 5 else ''} |")
    fields = capture(ROOT / "evals" / "ocr_eval.py").split("\n\n")[0]
    out += ["", "Field-level digit accuracy on the full tender against 94 hand-read fields (`evals/ocr_eval.py`):", "", fields, "",
            "Disagreements between readers were put to Claude Opus 5.5 with the page image as targeted questions. Two were "
            "real misreads by the primary reader and were corrected (p. 3 condition 3: \"08.00\" to \"06.00\", the bid "
            "deadline; p. 4 fee table: \"161 lakh\" to \"151 lakh\"); six were alignment false alarms where the primary "
            "reading was confirmed. The 151 slab boundary is not one of the 94 hand-read fields, so 94/94 does not count "
            "that error."]
    (ROOT / "evals" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "evals" / "results" / "ocr.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
