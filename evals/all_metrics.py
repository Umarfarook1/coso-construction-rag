"""Every standard RAG metric, per stage and per configuration, in one file the web view renders
(src/cosorag/web/assets/metrics.json).

Retrieval and reranking, on the 84 answerable gold questions, with the harness's relevance definition (a passage is
relevant when it carries the gold clause or rate item, or sits on the gold tender page):
  Hit@k        at least one relevant passage in the top k
  Recall@k     share of the question's gold references found in the top k
  Precision@k  share of the top k passages that are relevant
  MRR@10       1 / rank of the first relevant passage (0 if none in the top 10)
  nDCG@10      rank-discounted gain, one gain per gold reference, normalised by the ideal ranking
Answers, from the evaluated runs in runs/ (gold: 96 questions; blind: 30, aggregates only).
  uv run python evals/all_metrics.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, RUNS, boot_ci, chunk_relevant, load_gold, mean, page_relevant, summarize  # noqa: E402

OUT = ROOT / "src" / "cosorag" / "web" / "assets" / "metrics.json"
E5, COHERE = "intfloat/multilingual-e5-small", "bedrock:cohere.embed-multilingual-v3"
FT, COHERE_RR = str(ROOT / "models" / "xenc-coso"), "bedrock:cohere.rerank-v3-5:0"


def rank_stats(ranked, g) -> dict:
    refs = g["gold"]
    seen, gains, rel = set(), [], []
    for c in ranked[:10]:
        r = chunk_relevant(c, g)
        rel.append(r)
        new = [i for i, ref in enumerate(refs) if i not in seen and chunk_relevant(c, {"gold": [ref]})]
        gains.append(1.0 if new else 0.0)
        seen.update(new)
    allrel = [chunk_relevant(c, g) for c in ranked[:10]]
    first = next((i for i, x in enumerate(allrel) if x), None)

    def recall(k):
        got = set()
        for c in ranked[:k]:
            got.update(i for i, ref in enumerate(refs) if chunk_relevant(c, {"gold": [ref]}))
        return len(got) / len(refs)

    dcg = sum(gn / math.log2(i + 2) for i, gn in enumerate(gains))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(refs), 10)))
    out = {f"hit{k}": float(any(allrel[:k])) for k in (1, 3, 5, 8, 10)}
    out.update({f"recall{k}": recall(k) for k in (1, 5, 8, 10)})
    out.update({f"precision{k}": sum(allrel[:k]) / k for k in (1, 5, 8)})
    out["mrr10"] = 1 / (first + 1) if first is not None else 0.0
    out["ndcg10"] = dcg / idcg if idcg else 0.0
    out["page_hit5"] = float(any(page_relevant(c, g) for c in ranked[:5]))
    return out


def retrieval() -> list[dict]:
    from cosorag import rerank as rr
    from cosorag.index import Index
    from cosorag.retrieve import retrieve

    ixs = {"e5": Index.build(model_name=E5), "cohere": Index.build(model_name=COHERE)}
    gold = [g for g in load_gold() if g["answerable"] and g["gold"]]
    configs = [  # (key, label, stage, embedder, mode, reranker, fused)
        ("bm25", "BM25 keyword search", "retrieval", "e5", "bm25", None, False),
        ("dense-e5", "Dense: multilingual-e5-small", "retrieval", "e5", "dense", None, False),
        ("dense-cohere", "Dense: Cohere Embed Multilingual v3", "retrieval", "cohere", "dense", None, False),
        ("hybrid-e5", "Hybrid (e5): BM25 + dense + exact ids + per-document lists", "retrieval", "e5", "hybrid", None, False),
        ("hybrid-cohere", "Hybrid (Cohere): BM25 + dense + exact ids + per-document lists", "retrieval", "cohere", "hybrid", None, False),
        ("e5-orig-sorted", "Hybrid (e5) + original cross-encoder, re-sorted", "rerank", "e5", "hybrid", rr.BASE_XENC, False),
        ("e5-orig-fused", "Hybrid (e5) + original cross-encoder, fused", "rerank", "e5", "hybrid", rr.BASE_XENC, True),
        ("e5-ft-sorted", "Hybrid (e5) + fine-tuned cross-encoder, re-sorted", "rerank", "e5", "hybrid", FT, False),
        ("e5-ft-fused", "Hybrid (e5) + fine-tuned cross-encoder, fused (evaluated system)", "rerank", "e5", "hybrid", FT, True),
        ("e5-cohere-fused", "Hybrid (e5) + Cohere Rerank 3.5, fused", "rerank", "e5", "hybrid", COHERE_RR, True),
        ("cohere-ft-fused", "Hybrid (Cohere) + fine-tuned cross-encoder, fused", "rerank", "cohere", "hybrid", FT, True),
        ("cohere-cohere-fused", "Hybrid (Cohere) + Cohere Rerank 3.5, fused (live demo)", "rerank", "cohere", "hybrid", COHERE_RR, True),
    ]
    rows = {c[0]: [] for c in configs}
    for n, g in enumerate(gold, 1):
        cache: dict = {}
        for key, _, _, emb, mode, rer, fused in configs:
            ix = ixs[emb]
            if (emb, mode) not in cache:
                cache[(emb, mode)] = retrieve(ix, g["question"], k=20, mode=mode, pool_all_docs=True).hits
            hits = list(cache[(emb, mode)])
            if rer:
                pinned = [h for h in hits if "id" in h.sources]
                rest = [h for h in hits if "id" not in h.sources]
                if (emb, rer) not in cache:
                    cache[(emb, rer)] = rr.score(rer, g["question"], [ix.chunks[h.idx].index_text for h in rest])
                ps = cache[(emb, rer)]
                rest = rr.fused_order(rest, ps) if fused else [h for _, h in sorted(zip(ps, rest), key=lambda t: -t[0])]
                hits = pinned + rest
            rows[key].append({"doc": g["doc"], **rank_stats([ix.chunks[h.idx] for h in hits], g)})
        if n % 10 == 0:
            print(f"  retrieval {n}/{len(gold)}", flush=True)
    out = []
    for key, label, stage, emb, mode, rer, fused in configs:
        rs = rows[key]
        m = {k: round(mean([r[k] for r in rs]), 3) for k in rs[0] if k != "doc"}
        m["by_doc_hit5"] = {d: round(mean([r["hit5"] for r in rs if r["doc"] == d]), 3) for d in ("gcc", "bsr", "nit")}
        out.append({"key": key, "label": label, "stage": stage, "n": len(rs), **m})
    return out


GEN_RUNS = [  # (run dir, label, writer, retrieval, set)
    ("gen-claude-sonnet-5-5-bedrock-only", "Claude Sonnet 5.5, Bedrock-only pipeline (live demo)", "Claude Sonnet 5.5", "Cohere embed + Cohere rerank", "gold"),
    ("gen-claude-opus-5-5", "Claude Opus 5.5", "Claude Opus 5.5", "e5 + fine-tuned cross-encoder", "gold"),
    ("gen-claude-sonnet-5-5", "Claude Sonnet 5.5", "Claude Sonnet 5.5", "e5 + fine-tuned cross-encoder", "gold"),
    ("gen-claude-haiku-5-5", "Claude Haiku 5.5", "Claude Haiku 5.5", "e5 + fine-tuned cross-encoder", "gold"),
    ("gen-deepseek-v3", "DeepSeek V3", "DeepSeek V3", "e5 + fine-tuned cross-encoder", "gold"),
    ("gen-qwen3-235b", "Qwen3 235B", "Qwen3 235B", "e5 + fine-tuned cross-encoder", "gold"),
    ("final", "gpt-oss-120b (first submission)", "gpt-oss-120b", "e5 + fine-tuned cross-encoder", "gold"),
    ("gen-gpt-oss-20b", "gpt-oss-20b", "gpt-oss-20b", "e5 + fine-tuned cross-encoder", "gold"),
    ("v2-rerank-base-fused", "gpt-oss-120b, original cross-encoder", "gpt-oss-120b", "e5 + original cross-encoder", "gold"),
    ("v2-norerank", "gpt-oss-120b, no reranker", "gpt-oss-120b", "e5, no reranker", "gold"),
    ("final-v1-gate-in-prompt", "gpt-oss-120b, fine-tuned re-sorted + gate in prompt (first try)", "gpt-oss-120b", "e5 + fine-tuned, re-sorted", "gold"),
    ("blind-live-bedrock-sonnet-5-5", "Claude Sonnet 5.5, Bedrock-only pipeline (live demo)", "Claude Sonnet 5.5", "Cohere embed + Cohere rerank", "blind"),
    ("blind-sonnet-5-5", "Claude Sonnet 5.5", "Claude Sonnet 5.5", "e5 + fine-tuned cross-encoder", "blind"),
    ("blind-final", "gpt-oss-120b (first submission)", "gpt-oss-120b", "e5 + fine-tuned cross-encoder", "blind"),
    ("blind-rerank-base-fused", "gpt-oss-120b, original cross-encoder", "gpt-oss-120b", "e5 + original cross-encoder", "blind"),
    ("blind-norerank", "gpt-oss-120b, no reranker", "gpt-oss-120b", "e5, no reranker", "blind"),
]


def generation() -> list[dict]:
    out = []
    for run, label, writer, retr, which in GEN_RUNS:
        p = RUNS / run / "results.jsonl"
        if not p.exists():
            print("  missing run", run)
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8")]
        s = summarize(rows)
        ans = [r for r in rows if r["answerable_gold"]]
        answered = [r for r in rows if not r["refused"]]
        jp = s["judge_pass"]
        lo, hi = boot_ci(jp)
        out.append({
            "run": run, "label": label, "writer": writer, "retrieval": retr, "set": which, "n": len(rows),
            "judged_correct": round(mean(jp), 3), "judged_ci": [round(lo, 3), round(hi, 3)],
            "key_fact_recall": round(mean([r["facts_recall"] for r in ans if r.get("facts_recall") == r.get("facts_recall")]), 3),
            "all_key_facts": round(mean(s["correct_det"]), 3),
            "citation_hit": round(mean(s["citation_hit"]), 3),
            "citation_precision": round(mean(s["citation_precision"]), 3),
            "negatives_handled": round(mean(s["refusal_recall"]), 3), "negatives_n": len(s["refusal_recall"]),
            "refusal_precision": round(mean(s["refusal_precision"]), 3) if s["refusal_precision"] else None,
            "over_refusal": round(mean(s["over_refusal"]), 3),
            "unsupported_number": round(mean(s["unsupported_any"]), 3),
            "quotes_verified": round(mean([float(r.get("n_bad_quotes", 0) == 0) for r in answered]), 3),
            "parse_errors": s["parse_errors"],
            "latency_p50": round(s["latency_p50"], 1) if s["latency_p50"] == s["latency_p50"] else None,
        })
    return out


STATIC = {
    "reranker_synthetic": {
        "note": "Held-out synthetic questions from TEST page blocks the reranker never saw: 400 questions, 200 near-misses.",
        "rows": [
            {"label": "No reranker (hybrid order)", "hit1": 0.537, "hit8": 0.800, "mrr10": 0.632, "auroc_source": None, "auroc_top": None},
            {"label": "Original cross-encoder, re-sorted", "hit1": 0.652, "hit8": 0.848, "mrr10": 0.723, "auroc_source": 0.450, "auroc_top": 0.407},
            {"label": "Fine-tuned cross-encoder, re-sorted", "hit1": 0.708, "hit8": 0.882, "mrr10": 0.779, "auroc_source": 0.838, "auroc_top": 0.772},
            {"label": "Original cross-encoder, fused", "hit1": 0.590, "hit8": 0.850, "mrr10": 0.684, "auroc_source": None, "auroc_top": None},
            {"label": "Fine-tuned cross-encoder, fused", "hit1": 0.640, "hit8": 0.873, "mrr10": 0.726, "auroc_source": None, "auroc_top": None},
        ]},
    "gate_gold": {"note": "Top cross-encoder score as an answerability signal on the 96 real gold questions; threshold 0.70 from synthetic dev pages.",
                  "rows": [{"label": "Original cross-encoder", "auroc": 0.529, "answerable_below": 0.21, "unanswerable_below": 0.25},
                           {"label": "Fine-tuned cross-encoder", "auroc": 0.727, "answerable_below": 0.48, "unanswerable_below": 0.75}]},
    "ocr": {"note": "Tender scan: 6 hand-transcribed regions (61 lines, 86 numbers); field check on 94 hand-read numeric fields.",
            "rows": [{"label": "Claude Sonnet 4.6 (vision)", "cer": 0.006, "numbers": "86/86", "fields": "94/94", "role": "primary reader"},
                     {"label": "Llama 4 Maverick", "cer": 0.020, "numbers": "81/86", "fields": "92/94", "role": "third reader"},
                     {"label": "Gemma 3 27B", "cer": 0.043, "numbers": "86/86", "fields": "70/94", "role": "second reader"},
                     {"label": "Tesseract 5 hin+eng", "cer": 0.128, "numbers": "51/86", "fields": None, "role": "bake-off"},
                     {"label": "Pixtral Large", "cer": 0.201, "numbers": "71/86", "fields": None, "role": "bake-off"},
                     {"label": "RapidOCR PP-OCRv5 Devanagari", "cer": 0.228, "numbers": "78/86", "fields": None, "role": "bake-off"}]},
    "parsing": {"rows": [{"label": "Rate rows exactly right, hand-checked (in-sample)", "value": "63/63"},
                         {"label": "Rate agreement with an independently written parser", "value": "99.1% of 6,510"},
                         {"label": "Unit agreement with the independent parser", "value": "98.7%"},
                         {"label": "Generic table extractors on the same 63 rows", "value": "54% and 56%"},
                         {"label": "Tender numbers after three-reader voting", "value": "94/94"}]},
}


def markdown(m: dict) -> str:
    """The same tables as the web view, for readers of the repository (evals/METRICS.md)."""
    f3 = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
    pc = lambda v: "" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    keys = ["hit1", "hit3", "hit5", "hit8", "recall5", "recall8", "recall10", "precision1", "precision5", "mrr10", "ndcg10", "page_hit5"]
    out = ["# All metrics", "", "Generated by `evals/all_metrics.py` (also rendered in the web view, How it works > All metrics).",
           "Retrieval and reranking: 84 answerable gold questions, locator-level relevance. Answers: 96 gold questions and 30 blind",
           "questions (blind runs once per configuration, aggregates only).", "",
           "## Retrieval and reranking", "",
           "| configuration | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    out += [f"| {r['label']} | " + " | ".join(f3(r[k]) for k in keys) + " |" for r in m["retrieval"]]
    gk = [("judged correct", "judged_correct"), ("key-fact recall", "key_fact_recall"), ("all key facts", "all_key_facts"),
          ("citation hit", "citation_hit"), ("citation precision", "citation_precision"), ("negatives handled", "negatives_handled"),
          ("refusal precision", "refusal_precision"), ("over-refusal", "over_refusal"), ("unsupported number", "unsupported_number"),
          ("quotes verified", "quotes_verified")]
    for which, n in (("gold", 96), ("blind", 30)):
        out += ["", f"## Answers, {which} set ({n} questions)", "",
                "| configuration | retrieval | " + " | ".join(h for h, _ in gk) + " |", "|---|---|" + "---|" * len(gk)]
        out += [f"| {g['label']} | {g['retrieval']} | " + " | ".join(
            (pc(g[k]) + (f" [{pc(g['judged_ci'][0])}, {pc(g['judged_ci'][1])}]" if k == "judged_correct" else "")) for _, k in gk) + " |"
                for g in m["generation"] if g["set"] == which]
    sy, gg, ocr = m["reranker_synthetic"], m["gate_gold"], m["ocr"]
    out += ["", "## Fine-tuning: the cross-encoder before and after", "", sy["note"], "",
            "| reranker | hit1 | hit8 | mrr10 | near-miss AUROC (source) | near-miss AUROC (top) |", "|---|---|---|---|---|---|"]
    out += [f"| {r['label']} | {f3(r['hit1'])} | {f3(r['hit8'])} | {f3(r['mrr10'])} | {f3(r['auroc_source'])} | {f3(r['auroc_top'])} |" for r in sy["rows"]]
    out += ["", gg["note"], "", "| score | AUROC | answerable under 0.70 | unanswerable under 0.70 |", "|---|---|---|---|"]
    out += [f"| {r['label']} | {f3(r['auroc'])} | {pc(r['answerable_below'])} | {pc(r['unanswerable_below'])} |" for r in gg["rows"]]
    out += ["", "## Reading the documents", "", ocr["note"], "", "| OCR engine | CER | numbers exact | fields (94) | role |", "|---|---|---|---|---|"]
    out += [f"| {r['label']} | {pc(r['cer'])} | {r['numbers']} | {r['fields'] or ''} | {r['role']} |" for r in ocr["rows"]]
    out += ["", "| table parsing and voting | result |", "|---|---|"] + [f"| {r['label']} | {r['value']} |" for r in m["parsing"]["rows"]]
    if m.get("router"):
        ro = m["router"]
        out += ["", "## Input guardrail: routing before retrieval", "", ro["note"], "",
                f"Accuracy {pc(ro['accuracy'])} on {ro['n']} messages; follow-up rewrites that kept the subject {ro['rewrite_ok']}; "
                f"gold questions routed to the documents {ro['gold_docs']}, blind {ro['blind_docs']}; "
                f"{pc(ro['rules_share'])} answered by the greeting list without a model; router median {ro['latency_p50']} s "
                "(it runs alongside the search for a first question).", "",
                "| route | messages | recall | precision |", "|---|---|---|---|"]
        out += [f"| {k} | {v['n']} | {pc(v['recall'])} | {pc(v['precision'])} |" for k, v in ro["per_route"].items()]
    out += ["", "## Metric definitions", "",
            "- **hit@k**: at least one relevant passage in the top k (hit8 is what the answer writer sees).",
            "- **recall@k**: share of the question's gold references (clauses, rows, pages) in the top k.",
            "- **precision@k**: share of the top k that is relevant; low by design, most questions have one or two right passages.",
            "- **mrr10**: mean of 1 / rank of the first relevant passage. **ndcg10**: one gain per gold reference, discounted by rank, normalised to 1.",
            "- **page_hit5**: the right page in the top 5 even when the exact clause or row is not.",
            "- **judged correct**: a Claude Sonnet 4.6 judge compares the answer with the gold answer (95% bootstrap interval).",
            "- **key-fact recall / all key facts**: plain string matching of the gold key facts; all key facts is a strict lower bound.",
            "- **citation hit**: a citation points at the gold clause, rate row or tender page. **citation precision**: share of citations that do.",
            "- **negatives handled**: unanswerable, near-miss and false-premise questions declined or corrected. **over-refusal**: answerable ones declined.",
            "- **unsupported number**: an answer contains a number that is neither in a cited source nor one arithmetic step away.",
            "- **quotes verified**: every quote in the answer was found in the source it was attributed to."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if "--md" in sys.argv:
        (ROOT / "evals" / "METRICS.md").write_text(markdown(json.loads(OUT.read_text(encoding="utf-8"))), encoding="utf-8")
        raise SystemExit(0)
    gen = generation()
    ret = retrieval() if "--no-retrieval" not in sys.argv else json.loads(OUT.read_text(encoding="utf-8"))["retrieval"]
    router = ROOT / "evals" / "results" / "router.json"   # evals/router_eval.py
    extra = {"router": {k: v for k, v in json.loads(router.read_text(encoding="utf-8")).items() if k != "rows"}} if router.exists() else {}
    OUT.write_text(json.dumps({"retrieval": ret, "generation": gen, **STATIC, **extra}, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in ret:
        print(f"{r['label'][:62]:62} hit1 {r['hit1']:.3f} r@8 {r['recall8']:.3f} p@5 {r['precision5']:.3f} mrr {r['mrr10']:.3f} ndcg {r['ndcg10']:.3f}")
    for gn in gen:
        print(f"{gn['set']:5} {gn['label'][:60]:60} correct {gn['judged_correct']:.3f} cit {gn['citation_hit']:.3f} neg {gn['negatives_handled']:.3f}")
