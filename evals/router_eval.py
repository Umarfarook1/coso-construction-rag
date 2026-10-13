"""Evaluate the input guardrail (cosorag.guard) that routes every chat message before retrieval.

Three checks:
  1. evals/router_cases.jsonl: 56 labelled messages (greetings, questions about the assistant, unrelated requests and
     prompt injections, and tricky document questions: one-word, Hinglish, greeting + question, false premise,
     follow-ups with a conversation). Accuracy per route, and for follow-ups whether the rewrite kept the subject.
  2. The 96 gold questions: every one must be routed to the documents (a misrouted real question is the costly error).
  3. The 30 blind questions: the same check, reported as a count only (the blind set is never read by hand).
  uv run python evals/router_eval.py            -> evals/results/router.json
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, load_gold  # noqa: E402

from cosorag.guard import ROUTES, route  # noqa: E402
from cosorag.models import MODELS  # noqa: E402

OUT = ROOT / "evals" / "results" / "router.json"


def main() -> None:
    llm = MODELS["claude-haiku-5-5"].client()
    cases = [json.loads(line) for line in (ROOT / "evals" / "router_cases.jsonl").open(encoding="utf-8") if line.strip()]
    rows, conf = [], Counter()
    for c in cases:
        r = route(llm, c["message"], c.get("history"))
        ok = r.kind == c["route"]
        rew = None
        if c.get("expect_in_rewrite"):
            low = r.question.lower()
            rew = all(any(alt.lower() in low for alt in e.split("|")) for e in c["expect_in_rewrite"])
        conf[(c["route"], r.kind)] += 1
        rows.append({"id": c["id"], "message": c["message"], "expected": c["route"], "got": r.kind, "ok": ok,
                     "by": r.by, "seconds": r.seconds, "rewrite": r.question if c.get("history") else None, "rewrite_ok": rew})
        if not ok or rew is False:
            print(f"MISS {c['id']:5} expected {c['route']:9} got {r.kind:9} {c['message'][:60]!r} -> {r.question[:80]!r}")
    per = {}
    for k in ROUTES:
        tp = conf[(k, k)]
        n_exp = sum(v for (e, _), v in conf.items() if e == k)
        n_got = sum(v for (_, g), v in conf.items() if g == k)
        per[k] = {"n": n_exp, "recall": round(tp / n_exp, 3) if n_exp else None, "precision": round(tp / n_got, 3) if n_got else None}
    model_secs = [r["seconds"] for r in rows if r["by"] not in ("rules", "none", "fallback")]
    rw = [r["rewrite_ok"] for r in rows if r["rewrite_ok"] is not None]

    def docs_count(questions: list[str]) -> int:
        return sum(route(llm, q).kind == "docs" for q in questions)

    gold = load_gold()
    gold_docs = 0
    for g in gold:
        r = route(llm, g["question"])
        gold_docs += r.kind == "docs"
        if r.kind != "docs":
            print(f"GOLD MISROUTED {g['id']}: {r.kind}")
    blind_path = ROOT / "evals" / "gold_blind" / "blind.jsonl"
    blind = [json.loads(line)["question"] for line in blind_path.open(encoding="utf-8") if line.strip()]
    blind_docs = docs_count(blind)   # count only; the blind questions are never printed

    out = {
        "note": (f"{len(cases)} labelled messages (evals/router_cases.jsonl), written for development: the router prompt was "
                 "changed once, after the first run sent small talk (\"how are you?\") to the documents. Router model Claude "
                 "Haiku 5.5; plain greetings are matched without a model call."),
        "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 3), "n": len(rows),
        "per_route": per,
        "confusion": [{"expected": e, "got": g, "n": v} for (e, g), v in sorted(conf.items())],
        "rewrite_ok": f"{sum(rw)}/{len(rw)}",
        "rules_share": round(sum(r["by"] == "rules" for r in rows) / len(rows), 3),
        "latency_p50": round(statistics.median(model_secs), 2) if model_secs else None,
        "gold_docs": f"{gold_docs}/{len(gold)}", "blind_docs": f"{blind_docs}/{len(blind)}",
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"accuracy {out['accuracy']} ({len(rows)}), rewrites {out['rewrite_ok']}, gold to docs {out['gold_docs']}, "
          f"blind to docs {out['blind_docs']}, model p50 {out['latency_p50']} s")
    for k, v in per.items():
        print(f"  {k:9} n={v['n']:3} recall={v['recall']} precision={v['precision']}")


if __name__ == "__main__":
    main()
