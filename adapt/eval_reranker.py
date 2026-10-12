"""Before/after for the adaptation, on held-out synthetic queries from DEV/TEST pages (never trained on).

  ranking : the hybrid retriever's top-20 is reordered by (a) nothing, (b) the base cross-encoder,
            (c) the fine-tuned one, each alone and fused with the hybrid order (RRF, as answer() does);
            Hit@1, Hit@8 (inside the generator's context) and MRR@10 of the source chunk (or its clause /
            parent item).
  gate    : can the score tell a real question from a near-miss (one attribute changed, so the passage no
            longer answers)? AUROC on (query, source passage) pairs, and on the top candidate's score.
The gate threshold is chosen on DEV (max balanced accuracy on the top score) and written to gate.json.
  uv run python adapt/eval_reranker.py --split test --n 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cosorag import rerank as rr  # noqa: E402
from cosorag.config import ROOT  # noqa: E402
from cosorag.index import Index  # noqa: E402
from cosorag.retrieve import retrieve  # noqa: E402


def auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)) if pos and neg else float("nan")


def relevant(c, gold) -> bool:
    if c.id == gold.id:
        return True
    if gold.doc in ("gcc", "nit"):
        return c.doc == gold.doc and c.locator.split(" (cont")[0] == gold.locator.split(" (cont")[0]
    if gold.doc == "bsr" and c.doc == "bsr" and c.meta.get("kind") == "group" and c.meta.get("chapter") == gold.meta.get("chapter"):
        return gold.meta.get("item_no", "").startswith(c.meta.get("item_no", "~") + ".")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=400, help="positive queries for the ranking eval")
    ap.add_argument("--ft", default=str(rr.FT_XENC_DIR))
    ap.add_argument("--write-gate", action="store_true", help="choose the gate threshold (use on dev)")
    a = ap.parse_args()
    rng = random.Random(1)
    ix = Index.build()
    qs = [json.loads(l) for l in (ROOT / f"adapt/data/queries_{a.split}.jsonl").open(encoding="utf-8")]
    missing = sum(q["gold"] not in ix.by_id for q in qs)   # chunk ids that changed after re-ingestion
    qs = [q for q in qs if q["gold"] in ix.by_id]
    pos = [q for q in qs if q["kind"] == "pos"]
    nm = [q for q in qs if q["kind"] == "near_miss"]
    rng.shuffle(pos); rng.shuffle(nm)
    pos, nm = pos[: a.n], nm[: a.n // 2]
    models = {"base": rr.BASE_XENC, "ft": a.ft}
    out = {"split": a.split, "n_pos": len(pos), "n_near_miss": len(nm), "skipped_stale_ids": missing}

    # ---- ranking ------------------------------------------------------------------------------
    rank = {k: {"hit1": [], "hit8": [], "mrr10": []} for k in ("hybrid", "base", "ft", "base_fused", "ft_fused")}
    top = {k: {"pos": [], "nm": []} for k in ("base", "ft")}
    for kind, items in (("pos", pos), ("nm", nm)):
        for q in items:
            gold = ix.chunks[ix.by_id[q["gold"]]]
            # mirror answer(): id-pinned hits stay first and unscored; the gate looks at the rest
            hits = retrieve(ix, q["q"], k=20, mode="hybrid", pool_all_docs=True).hits
            pinned = [ix.chunks[h.idx] for h in hits if "id" in h.sources]
            cands = [ix.chunks[h.idx] for h in hits if "id" not in h.sources]
            orders = {"hybrid": pinned + cands}
            for name, model in models.items():
                ps = rr.score(model, q["q"], [c.index_text for c in cands])
                orders[name] = pinned + [c for _, c in sorted(zip(ps, cands), key=lambda t: -t[0])]
                orders[f"{name}_fused"] = pinned + rr.fused_order(cands, ps)   # what answer() uses
                if not pinned:
                    top[name][kind].append(max(ps) if ps else 0.0)
            if kind == "pos":
                for name, order in orders.items():
                    rel = [relevant(c, gold) for c in order]
                    first = next((i for i, x in enumerate(rel) if x), None)
                    rank[name]["hit1"].append(float(bool(rel[:1]) and rel[0]))
                    rank[name]["hit8"].append(float(any(rel[:8])))   # in the generator's context (k=8)
                    rank[name]["mrr10"].append(1 / (first + 1) if first is not None and first < 10 else 0.0)
    out["ranking"] = {k: {m: round(sum(v) / len(v), 3) for m, v in d.items()} for k, d in rank.items()}

    # ---- gate: pair scores on the source passage ------------------------------------------------
    pair = {}
    for name, model in models.items():
        sp = [rr.score(model, q["q"], [ix.chunks[ix.by_id[q["gold"]]].index_text])[0] for q in pos]
        sn = [rr.score(model, q["q"], [ix.chunks[ix.by_id[q["gold"]]].index_text])[0] for q in nm]
        pair[name] = {"auroc_pair": round(auroc(sp, sn), 3), "auroc_top": round(auroc(top[name]["pos"], top[name]["nm"]), 3)}
    out["gate"] = pair

    if a.write_gate:
        best = (0.0, 0.5)
        for t in [i / 100 for i in range(5, 96)]:
            tpr = sum(s >= t for s in top["ft"]["pos"]) / len(top["ft"]["pos"])
            tnr = sum(s < t for s in top["ft"]["nm"]) / len(top["ft"]["nm"])
            best = max(best, ((tpr + tnr) / 2, t))
        gate = {"threshold": best[1], "balanced_accuracy_dev": round(best[0], 3), "chosen_on": a.split}
        (Path(a.ft) / "gate.json").write_text(json.dumps(gate, indent=2))
        out["gate_threshold"] = gate
    print(json.dumps(out, indent=1))
    res = ROOT / "evals" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"reranker_{a.split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
