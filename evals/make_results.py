"""Compose evals/RESULTS.md from the committed run files (no model calls, no keys).
  uv run python evals/make_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from harness import fmt, mean, paired_boot, report, summarize  # noqa: E402

RUNS = ROOT / "runs"
CONFIGS = [("v1-norerank", "baseline-norerank", "first version, no reranker"),
           ("v2-norerank", "v2-norerank", "after error analysis, no reranker"),
           ("v2-rerank-base", "v2-rerank-base", "+ off-the-shelf cross-encoder (not adapted)"),
           ("final", "final", "+ fine-tuned cross-encoder and gate (submitted system)")]


def rows(name):
    p = RUNS / name / "results.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else None


def table(runs: list[tuple[str, list]]) -> list[str]:
    out = ["| system | n | judge: correct | key facts all found | citation hit | negatives handled | over-refusal | answers w/ unsupported number |",
           "|---|---|---|---|---|---|---|---|"]
    for label, rs in runs:
        s = summarize(rs)
        out.append(f"| {label} | {s['n']} | {fmt(s['judge_pass'])} | {fmt(s['correct_det'])} | {fmt(s['citation_hit'])} | "
                   f"{fmt(s['refusal_recall'])} | {fmt(s['over_refusal'])} | {fmt(s['unsupported_any'])} |")
    return out


def compare(a: list, b: list, label: str) -> str:
    ja = {r["id"]: r.get("judge", 0.0) for r in a}
    jb = {r["id"]: r.get("judge", 0.0) for r in b}
    ids = sorted(set(ja) & set(jb))
    xa, xb = [ja[i] for i in ids], [jb[i] for i in ids]
    p = paired_boot(xa, xb)
    wins = sum(y > x for x, y in zip(xa, xb)); losses = sum(y < x for x, y in zip(xa, xb))
    return (f"- {label}: judge {mean(xa):.3f} -> {mean(xb):.3f} (n={len(ids)}; {wins} questions better, {losses} worse; "
            f"paired bootstrap P(no improvement) = {p:.2f})")


def main() -> None:
    md = ["# Results", "", "All numbers below are recomputed from files in `runs/` and `evals/results/` by `evals/make_results.py`.",
          "Intervals are 95% bootstrap intervals. The gold set is a development set (see `evals/README.md`); the blind set is the clean estimate.", ""]

    md += ["## End to end, gold set (96 questions)", ""]
    have = [(f"{lab}: {desc}", rows(name)) for lab, name, desc in CONFIGS if rows(name)]
    md += table(have) + [""]
    base, ft = rows("v2-norerank"), rows("final")
    if base and ft:
        md += ["Paired comparisons (same questions):", compare(base, ft, "no reranker -> fine-tuned reranker")]
        if rows("v2-rerank-base"):
            md += [compare(rows("v2-rerank-base"), ft, "off-the-shelf reranker -> fine-tuned reranker")]
        md += [""]

    blind = [(lab, rows(name)) for lab, name in (("no reranker", "blind-norerank"), ("submitted system", "blind-final")) if rows(name)]
    if blind:
        md += ["## Blind set (30 questions, written after the code freeze, run once)", ""] + table(blind) + [""]
        if len(blind) == 2:
            md += [compare(blind[0][1], blind[1][1], "no reranker -> fine-tuned reranker"), ""]

    if ft:
        md += ["## Submitted system by slice", "", report(RUNS / "final"), ""]

    rr = ROOT / "evals" / "results" / "reranker_test.json"
    if rr.exists():
        d = json.loads(rr.read_text())
        md += ["## Adaptation before/after on held-out synthetic queries (TEST page blocks, never trained on)", "",
               f"{d['n_pos']} real queries, {d['n_near_miss']} near-miss queries. Candidates: hybrid top-20 plus each document's top-3.", "",
               "| reranking | Hit@1 | MRR@10 |", "|---|---|---|"]
        for k, lab in (("hybrid", "none (hybrid order)"), ("base", "off-the-shelf cross-encoder"), ("ft", "fine-tuned cross-encoder")):
            md.append(f"| {lab} | {d['ranking'][k]['hit1']:.3f} | {d['ranking'][k]['mrr10']:.3f} |")
        md += ["", "| gate (real vs near-miss query) | AUROC on the source passage | AUROC on the top candidate |", "|---|---|---|"]
        for k, lab in (("base", "off-the-shelf"), ("ft", "fine-tuned")):
            md.append(f"| {lab} | {d['gate'][k]['auroc_pair']:.3f} | {d['gate'][k]['auroc_top']:.3f} |")
        md += [""]

    ret = ROOT / "evals" / "results" / "retrieval_gold.json"
    if ret.exists():
        d = json.loads(ret.read_text(encoding="utf-8"))
        md += ["## Retrieval on the gold set (answerable questions, locator-level)", "",
               "| mode | n | Hit@1 | Hit@5 | MRR@10 | page Hit@5 |", "|---|---|---|---|---|---|"]
        for m, s in d["summary"].items():
            md.append(f"| {m} | {s['n']} | {s['hit1']:.3f} | {s['hit5']:.3f} | {s['mrr10']:.3f} | {s['page_hit5']:.3f} |")
        md += [""]

    ocr = ROOT / "evals" / "results" / "ocr.md"
    if ocr.exists():
        md += ["## OCR on the Hindi tender", "", ocr.read_text(encoding="utf-8"), ""]
    (ROOT / "evals" / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
