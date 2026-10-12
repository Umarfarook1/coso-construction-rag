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
           ("v2-rerank-base", "v2-rerank-base", "+ off-the-shelf cross-encoder, re-sorted"),
           ("v2-rerank-base-fused", "v2-rerank-base-fused", "+ off-the-shelf cross-encoder, fused with hybrid order"),
           ("ft-gate-in-prompt", "final-v1-gate-in-prompt", "+ fine-tuned cross-encoder, re-sorted, gate note in the prompt (first version)"),
           ("final", "final", "+ fine-tuned cross-encoder, fused with hybrid order, gate as review flag (submitted)")]


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
        md += ["Paired comparisons (same questions):", compare(base, ft, "no reranker -> submitted")]
        for name, label in (("v2-rerank-base-fused", "off-the-shelf fused -> fine-tuned fused (the adaptation alone)"),
                            ("final-v1-gate-in-prompt", "first fine-tuned version -> submitted")):
            if rows(name):
                md += [compare(rows(name), ft, label)]
        md += [""]

    blind = [(lab, rows(name)) for lab, name in (("no reranker", "blind-norerank"),
                                                 ("off-the-shelf cross-encoder, fused", "blind-rerank-base-fused"),
                                                 ("first fine-tuned version (gate note in the prompt)", "blind-final-v1-gate-in-prompt"),
                                                 ("submitted system (fine-tuned, fused)", "blind-final")) if rows(name)]
    if blind:
        md += ["## Blind set (30 questions, written after the code freeze)", "",
               "Each configuration was run on it exactly once. The first fine-tuned version was frozen and run on the",
               "blind set before its gold-set regression was analysed; the fix (fusion, gate as a flag) was chosen on the",
               "gold set only, and the blind questions were never read. Both runs are reported.", ""] + table(blind) + [""]
        if rows("blind-norerank") and rows("blind-final"):
            md += [compare(rows("blind-norerank"), rows("blind-final"), "no reranker -> submitted")]
        if rows("blind-rerank-base-fused") and rows("blind-final"):
            md += [compare(rows("blind-rerank-base-fused"), rows("blind-final"), "off-the-shelf fused -> fine-tuned fused (the adaptation alone)")]
        md += [""]

    if ft:
        md += ["## Submitted system by slice", "", report(RUNS / "final"), ""]

    rr = ROOT / "evals" / "results" / "reranker_test.json"
    if rr.exists():
        d = json.loads(rr.read_text())
        md += ["## Adaptation before/after on held-out synthetic queries (TEST page blocks, never trained on)", "",
               f"{d['n_pos']} real queries, {d['n_near_miss']} near-miss queries. Candidates: hybrid top-20 plus each document's top-3.", "",
               "| reranking | Hit@1 | Hit@8 (in context) | MRR@10 |", "|---|---|---|---|"]
        for k, lab in (("hybrid", "none (hybrid order)"), ("base", "off-the-shelf cross-encoder, re-sorted"), ("ft", "fine-tuned cross-encoder, re-sorted"),
                       ("base_fused", "off-the-shelf, fused with hybrid order (RRF)"),
                       ("ft_fused", "fine-tuned, fused with hybrid order (RRF; submitted)")):
            if k in d["ranking"]:
                r = d["ranking"][k]
                md.append(f"| {lab} | {r['hit1']:.3f} | {r.get('hit8', float('nan')):.3f} | {r['mrr10']:.3f} |")
        md += ["", "| gate (real vs near-miss query) | AUROC on the source passage | AUROC on the top candidate |", "|---|---|---|"]
        for k, lab in (("base", "off-the-shelf"), ("ft", "fine-tuned")):
            md.append(f"| {lab} | {d['gate'][k]['auroc_pair']:.3f} | {d['gate'][k]['auroc_top']:.3f} |")
        md += [""]

    gg = ROOT / "evals" / "results" / "gate_gold.json"
    if gg.exists():
        d = json.loads(gg.read_text(encoding="utf-8"))
        md += ["## Does the gate transfer to real questions? (gold set, `evals/gate_gold.py`)", "",
               f"Top cross-encoder score per question; {d['n_answerable']} answerable vs {d['n_not']} not answerable. "
               f"Threshold {d['threshold_from_dev']} was chosen on synthetic DEV queries.", "",
               "| model | AUROC answerable vs not | answerable below threshold | not answerable below threshold |", "|---|---|---|---|"]
        for m, s in d["models"].items():
            md.append(f"| {m} | {s['auroc']:.3f} | {s['answerable_below_threshold']:.0%} | {s['not_answerable_below_threshold']:.0%} |")
        md += [""]

    ret = ROOT / "evals" / "results" / "retrieval_gold.json"
    if ret.exists():
        d = json.loads(ret.read_text(encoding="utf-8"))
        md += ["## Retrieval on the gold set (answerable questions, locator-level)", "",
               "| mode | n | Hit@1 | Hit@5 | Hit@8 (in context) | MRR@10 | page Hit@5 |", "|---|---|---|---|---|---|---|"]
        for m, s in d["summary"].items():
            md.append(f"| {m} | {s['n']} | {s['hit1']:.3f} | {s['hit5']:.3f} | {s['hit8']:.3f} | {s['mrr10']:.3f} | {s['page_hit5']:.3f} |")
        md += [""]

    ocr = ROOT / "evals" / "results" / "ocr.md"
    if ocr.exists():
        md += ["## OCR on the Hindi tender", "", ocr.read_text(encoding="utf-8"), ""]
    (ROOT / "evals" / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
