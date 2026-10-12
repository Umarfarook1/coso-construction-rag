"""Evaluation harness: retrieval metrics (no LLM) and end-to-end answer metrics (deterministic first, judge second).

Gold questions live in evals/gold/{gcc,bsr,nit}.jsonl. They were drafted from the source pages (not from system
outputs) before any system run, and every answer was re-checked against the page; see evals/README.md.
Eval split: questions whose gold page is on a DEV page block (and a hash-chosen quarter of the Hindi notice)
are used for tuning; everything else is TEST and is only ever reported.

  uv run python evals/harness.py retrieval                     # all retrieval modes, no keys needed
  uv run python evals/harness.py e2e --name final              # end-to-end with the configured LLM
  uv run python evals/harness.py report runs/final             # reprint a finished run (no keys)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cosorag.config import ROOT  # noqa: E402
from judge import fact_found, judge  # noqa: E402

GOLD_DIR = ROOT / "evals" / "gold"
RUNS = ROOT / "runs"
REFUSAL_RE = re.compile(r"not (?:found|contain|mention|specif|provide|state|list|includ|available|present|give)|does not exist|"
                        r"no (?:such|information|mention)|cannot (?:be )?(?:answer|determin|find)|isn't (?:listed|stated)|"
                        r"is not (?:listed|stated|given|available)|unable to", re.I)


# ---------------------------------------------------------------------------------------------- gold
def load_gold(gold_dir: Path | None = None) -> list[dict]:
    items = []
    for f in sorted((gold_dir or GOLD_DIR).glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                items.append(json.loads(line))
    for g in items:
        if g["id"].startswith("blind"):
            g["eval_split"] = "blind"
        elif g["doc"] == "nit":
            h = int(hashlib.sha1(g["id"].encode()).hexdigest(), 16)
            g["eval_split"] = "dev" if h % 4 == 0 else "test"
        else:
            g["eval_split"] = "dev" if g.get("page_split") == "dev" else "test"
    return items


def _norm_loc(s: str) -> str:
    s = re.sub(r"\s+", "", s or "").lower()
    return s.replace("item", "").replace("schedulef,", "")


def chunk_relevant(c, g: dict) -> bool:
    """A chunk is relevant if it is on a gold page and (when the gold names a clause / item) carries that locator.
    For a schedule-of-rates child item, its parent "group" chunk also counts: it lists the child row and rate."""
    for ref in g.get("gold", []):
        if c.doc != ref["doc"]:
            continue
        loc = ref.get("locator", "")
        if c.doc == "bsr" and loc.lower().startswith("item"):
            want = _norm_loc(loc)
            have = _norm_loc(c.locator)
            if have == want or (c.meta.get("kind") == "group" and want.startswith(have + ".")):
                return True
            continue
        if c.doc == "gcc" and loc.lower().startswith("clause"):
            if _norm_loc(c.locator) == _norm_loc(loc) and abs(c.pdf_page - ref["pdf_page"]) <= 3:
                return True
            continue
        if c.pdf_page == ref["pdf_page"]:
            return True
    return False


def page_relevant(c, g: dict) -> bool:
    return any(c.doc == r["doc"] and c.pdf_page == r["pdf_page"] for r in g.get("gold", []))


# ---------------------------------------------------------------------------------------------- stats
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, n=2000, seed=0):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return (ms[int(0.025 * n)], ms[int(0.975 * n)])


def paired_boot(a, b, n=2000, seed=0):
    """P(mean(b) - mean(a) <= 0) under a paired bootstrap; a, b are per-question scores in the same order."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    worse = sum(1 for _ in range(n) if mean([d[rng.randrange(len(d))] for _ in d]) <= 0)
    return worse / n


def fmt(xs) -> str:
    lo, hi = boot_ci(xs)
    return f"{mean(xs):.3f} [{lo:.2f}, {hi:.2f}]"


# ---------------------------------------------------------------------------------------------- retrieval
def rank_metrics(ranked_chunks, g) -> dict:
    loc = [chunk_relevant(c, g) for c in ranked_chunks]
    pag = [page_relevant(c, g) for c in ranked_chunks]
    first = next((i for i, x in enumerate(loc) if x), None)
    return {"hit1": float(loc[:1] == [True]), "hit5": float(any(loc[:5])), "hit10": float(any(loc[:10])),
            "mrr10": 1.0 / (first + 1) if first is not None and first < 10 else 0.0,
            "page_hit5": float(any(pag[:5]))}


def run_retrieval(split: str = "test", modes=None) -> dict:
    from cosorag import rerank as rr
    from cosorag.index import Index
    from cosorag.retrieve import retrieve

    ix = Index.build()
    gold = [g for g in load_gold() if g["answerable"] and g["gold"] and (split == "all" or g["eval_split"] == split)]
    modes = modes or ["bm25", "dense", "hybrid", "hybrid+xenc_base", "hybrid+xenc_ft"]
    ft = rr.default_reranker()
    res: dict[str, list] = {m: [] for m in modes}
    for g in gold:
        cache = {}
        for m in modes:
            base_mode = m.split("+")[0]
            if base_mode not in cache:
                cache[base_mode] = retrieve(ix, g["question"], k=20, mode=base_mode, pool_all_docs=True)
            hits = list(cache[base_mode].hits)
            if "+xenc" in m:
                name = rr.BASE_XENC if m.endswith("base") else ft
                if name is None:
                    continue
                pinned = [h for h in hits if "id" in h.sources]
                rest = [h for h in hits if "id" not in h.sources]
                ps = rr.score(name, g["question"], [ix.chunks[h.idx].index_text for h in rest])
                rest = [h for _, h in sorted(zip(ps, rest), key=lambda t: -t[0])]
                hits = pinned + rest
            ranked = [ix.chunks[h.idx] for h in hits]
            res[m].append({"id": g["id"], "doc": g["doc"], "type": g["type"], **rank_metrics(ranked, g)})
    summary = {}
    for m, rows in res.items():
        if not rows:
            continue
        summary[m] = {k: round(mean([r[k] for r in rows]), 3) for k in ("hit1", "hit5", "hit10", "mrr10", "page_hit5")}
        summary[m]["n"] = len(rows)
        summary[m]["by_doc"] = {d: round(mean([r["hit5"] for r in rows if r["doc"] == d]), 3)
                                for d in sorted({r["doc"] for r in rows})}
    return {"split": split, "summary": summary, "rows": res}


# ---------------------------------------------------------------------------------------------- end to end
def score_answer(g: dict, a) -> dict:
    text = a.answer or ""
    refused = not a.answerable   # the model's own flag; hedged answers are not refusals
    row = {"id": g["id"], "doc": g["doc"], "type": g["type"], "lang": g.get("lang", "en"), "answerable_gold": g["answerable"],
           "eval_split": g["eval_split"], "page_split": g.get("page_split", ""), "refused": refused,
           "answer": text, "citations": [c["cite"] for c in a.citations], "checks": {k: v for k, v in a.checks.items() if k != "bad_quotes"},
           "n_bad_quotes": len(a.checks.get("bad_quotes", [])), "timings": a.timings, "gate": a.gate,
           "parse_error": bool(a.checks.get("parse_error"))}
    if g["answerable"]:
        facts = g.get("key_facts", [])
        found = [fact_found(f, text) for f in facts]
        row["facts_recall"] = mean([float(x) for x in found]) if found else float("nan")
        row["correct_det"] = float(all(found)) if found else float("nan")
        cited = [a.sources[c["n"] - 1] for c in a.citations]
        # locator-level for contract clauses and rate items, page-level for the tender (no locators there)
        row["citation_hit"] = float(any(chunk_relevant(c, g) for c in cited))
        row["citation_precision"] = mean([float(chunk_relevant(c, g)) for c in cited]) if cited else 0.0
        row["unsupported_numbers"] = len(a.checks.get("unsupported_numbers", []))
    return row


def run_e2e(name: str, split: str = "test", limit: int = 0, rerank: bool = True, mode: str = "hybrid",
            gold_dir: Path | None = None, use_judge: bool = True) -> Path:
    from cosorag import rerank as rr
    from cosorag.answer import answer
    from cosorag.index import Index
    from cosorag.llm import LLM

    ix = Index.build()
    llm = LLM()
    gold = [g for g in load_gold(gold_dir) if split == "all" or g["eval_split"] == split]
    if not gold:
        raise SystemExit(f"no gold questions selected (split={split}, gold_dir={gold_dir}); use --split all")
    if rerank and os.environ.get("COSORAG_RERANKER") != "none":
        if rr.default_reranker() is None or (rr.gate_threshold(rr.default_reranker()) is None
                                             and os.environ.get("COSORAG_ALLOW_NO_GATE") != "1"):
            raise SystemExit("reranker or gate.json missing: train + eval_reranker --write-gate first, or set COSORAG_RERANKER=none")
    if limit:
        gold = gold[:limit]
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, g in enumerate(gold, 1):
        a = answer(ix, g["question"], llm=llm, rerank=rerank, mode=mode)
        row = score_answer(g, a)
        row["question"] = g["question"]
        row["gold_answer"] = g["answer"]
        row["raw"] = a.raw
        if use_judge:
            v = judge(g["question"], g, a.answer)
            row["judge"], row["judge_verdict"], row["judge_reason"] = v["score"], v.get("verdict"), v.get("reason")
        rows.append(row)
        if i % 10 == 0:
            print(f"  {i}/{len(gold)}  {time.time() - t0:.0f}s")
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    xenc = rr.default_reranker() if rerank else None
    (out / "config.json").write_text(json.dumps({"llm": llm.name, "rerank": rerank, "reranker": xenc,
                                                 "gate_threshold": rr.gate_threshold(xenc) if xenc else None, "mode": mode, "split": split,
                                                 "n": len(rows)}, indent=2))
    print(report(out))
    return out


def rejudge(run_dir: Path) -> None:
    gold = {g["id"]: g for g in load_gold()} | {g["id"]: g for g in load_gold(ROOT / "evals" / "gold_blind")}
    path = Path(run_dir) / "results.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    for r in rows:
        g = gold[r["id"]]
        v = judge(g["question"], g, r["answer"])
        r["judge"], r["judge_verdict"], r["judge_reason"] = v["score"], v.get("verdict"), v.get("reason")
        if g["answerable"]:
            found = [fact_found(f, r["answer"]) for f in g.get("key_facts", [])]
            r["correct_det"] = float(all(found)) if found else float("nan")
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(report(run_dir))


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable_gold"]]
    una = [r for r in rows if not r["answerable_gold"]]
    s = {
        "n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
        "correct_det": [r["correct_det"] for r in ans if r.get("correct_det") == r.get("correct_det")],
        "answered_correct": [float(r.get("correct_det") == 1.0 and not r["refused"]) for r in ans],
        "citation_hit": [r["citation_hit"] for r in ans],
        "citation_precision": [r["citation_precision"] for r in ans],
        "over_refusal": [float(r["refused"]) for r in ans],
        # negatives are "handled" when the judge says the answer declined or rejected the premise; ambiguous
        # item numbers (the right behaviour is to list each chapter) are reported in their own slice
        "refusal_recall": [float(r.get("judge", 0.0) == 1.0) for r in una if r["type"] != "ambiguous_item"],
        "parse_errors": sum(1 for r in rows if r.get("parse_error")),
        "unsupported_any": [float(r.get("unsupported_numbers", 0) > 0) for r in ans],
        "judge_pass": [r["judge"] for r in rows if isinstance(r.get("judge"), (int, float))],
    }
    refused = [r for r in rows if r["refused"]]
    s["refusal_precision"] = [float(not r["answerable_gold"]) for r in refused]
    lat = [r["timings"].get("retrieve_s", 0) + r["timings"].get("rerank_s", 0) + r["timings"].get("generate_s", 0)
           for r in rows if not r["timings"].get("cached")]
    s["latency_p50"] = statistics.median(lat) if lat else float("nan")
    s["latency_p95"] = sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else float("nan")
    return s


def report(run_dir: Path) -> str:
    rows = [json.loads(l) for l in (Path(run_dir) / "results.jsonl").open(encoding="utf-8")]
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    lines = [f"## {Path(run_dir).name}  ({cfg['llm']}, reranker={cfg.get('reranker', cfg['rerank'])}, split={cfg['split']}, n={len(rows)})", "",
             "| slice | n | judge: correct | key facts all found | citation hit | negatives handled | over-refusal | answers w/ unsupported number |",
             "|---|---|---|---|---|---|---|---|"]
    slices = [("all", rows)] + [(f"split={sp}", [r for r in rows if r["eval_split"] == sp]) for sp in ("dev", "test", "blind")] \
        + [(d, [r for r in rows if r["doc"] == d]) for d in ("gcc", "bsr", "nit")]
    for name, rs in slices:
        if not rs:
            continue
        s = summarize(rs)
        lines.append(f"| {name} | {s['n']} | {fmt(s['judge_pass'])} | {fmt(s['correct_det'])} | {fmt(s['citation_hit'])} | "
                     f"{fmt(s['refusal_recall'])} | {fmt(s['over_refusal'])} | {fmt(s['unsupported_any'])} |")
    s = summarize(rows)
    lines += ["", f"refusal precision {fmt(s['refusal_precision'])}; parse errors {s['parse_errors']}; latency p50 {s['latency_p50']:.1f}s, p95 {s['latency_p95']:.1f}s"
              + (f"; judge pass {fmt(s['judge_pass'])}" if s["judge_pass"] else "")]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval"); r.add_argument("--split", default="test"); r.add_argument("--out", default="")
    e = sub.add_parser("e2e"); e.add_argument("--name", required=True); e.add_argument("--split", default="test")
    e.add_argument("--limit", type=int, default=0); e.add_argument("--no-rerank", action="store_true")
    e.add_argument("--mode", default="hybrid"); e.add_argument("--gold", default=""); e.add_argument("--no-judge", action="store_true")
    p = sub.add_parser("report"); p.add_argument("run")
    j = sub.add_parser("rejudge", help="add judge verdicts to an existing run"); j.add_argument("run")
    a = ap.parse_args()
    if a.cmd == "retrieval":
        res = run_retrieval(a.split)
        for m, s in res["summary"].items():
            print(f"{m:18} n={s['n']} hit@1 {s['hit1']:.3f} hit@5 {s['hit5']:.3f} hit@10 {s['hit10']:.3f} "
                  f"MRR@10 {s['mrr10']:.3f} page-hit@5 {s['page_hit5']:.3f}  by-doc hit@5 {s['by_doc']}")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    elif a.cmd == "e2e":
        run_e2e(a.name, a.split, a.limit, rerank=not a.no_rerank, mode=a.mode,
                gold_dir=Path(a.gold) if a.gold else None, use_judge=not a.no_judge)
    elif a.cmd == "rejudge":
        rejudge(Path(a.run))
    else:
        print(report(Path(a.run)))


if __name__ == "__main__":
    main()
