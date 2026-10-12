"""Does the gate transfer from synthetic queries to real ones? On every gold question (answerable or not), the
top cross-encoder score over the same candidate pool answer() uses, for the off-the-shelf and the fine-tuned
model: AUROC of that score for "answerable" vs "not answerable", and how many answerable questions fall below
the threshold chosen on synthetic DEV queries. No LLM, CPU only.
  uv run python evals/gate_gold.py  ->  evals/results/gate_gold.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from harness import load_gold  # noqa: E402

from cosorag import rerank as rr  # noqa: E402
from cosorag.index import Index  # noqa: E402
from cosorag.retrieve import retrieve  # noqa: E402


def auroc(pos: list[float], neg: list[float]) -> float:
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main() -> None:
    ix = Index.build()
    gold = load_gold()
    models = {"off-the-shelf": rr.BASE_XENC, "fine-tuned": str(rr.FT_XENC_DIR)}
    thr = rr.gate_threshold(str(rr.FT_XENC_DIR))
    tops = {m: {"answerable": [], "not": []} for m in models}
    for g in gold:
        hits = retrieve(ix, g["question"], k=20, mode="hybrid", pool_all_docs=True).hits
        cands = [ix.chunks[h.idx].index_text for h in hits if "id" not in h.sources]
        for m, name in models.items():
            top = max(rr.score(name, g["question"], cands), default=0.0)
            tops[m]["answerable" if g["answerable"] else "not"].append(round(top, 4))
    out = {"n_answerable": len(tops["fine-tuned"]["answerable"]), "n_not": len(tops["fine-tuned"]["not"]),
           "threshold_from_dev": thr, "models": {}}
    for m in models:
        pos, neg = tops[m]["answerable"], tops[m]["not"]
        out["models"][m] = {"auroc": round(auroc(pos, neg), 3),
                            "answerable_below_threshold": round(sum(p < thr for p in pos) / len(pos), 3) if thr else None,
                            "not_answerable_below_threshold": round(sum(n < thr for n in neg) / len(neg), 3) if thr else None}
    out["scores"] = tops
    print(json.dumps({k: v for k, v in out.items() if k != "scores"}, indent=1))
    (ROOT / "evals" / "results" / "gate_gold.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
