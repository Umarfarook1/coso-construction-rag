"""Turn synthetic queries into labelled (query, passage, label) pairs for the cross-encoder.

  positive   : (query, the chunk it was written from)                        label 1
  hard neg   : (query, a chunk the base hybrid retriever ranks high)          label 0
  near-miss  : (near-miss query, the chunk it was written from)               label 0

False-negative guard: a rate row's parent "group" chunks list the row itself, and a contract clause or tender
condition is split into several chunks, so those never serve as negatives for their own query. Sibling rows
(same parent, different size) DO serve as hard negatives: telling 15 mm from 20 mm is the point.
Leakage guard: queries AND negatives come only from TRAIN page blocks (cosorag/splits.py), so DEV/TEST
passages are never seen in training, as positives or as negatives.
Usage: uv run python adapt/build_pairs.py [--hard 3]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosorag.index import BASE_MODEL, Index  # noqa: E402
from cosorag.config import ROOT  # noqa: E402
from cosorag.retrieve import retrieve  # noqa: E402
from cosorag.splits import page_split  # noqa: E402

SYN = ROOT / "adapt" / "data" / "synthetic.jsonl"
VAGUE = re.compile(r"\b(these|this item|the above|such an?|this passage|this clause|this section|listed here)\b", re.I)
OUT = ROOT / "adapt" / "data"


def same_answer(src, c) -> bool:
    """True if chunk c may legitimately answer a query written for src (so it must not be a negative)."""
    if c.id == src.id or c.id.split("#")[0] == src.id.split("#")[0]:
        return True
    if src.doc != c.doc:
        return False
    if src.doc == "bsr" and src.meta.get("item_no") and c.meta.get("item_no"):
        if src.meta.get("chapter") != c.meta.get("chapter"):
            return False
        a, b = src.meta["item_no"], c.meta["item_no"]
        return a.startswith(b + ".") or b.startswith(a + ".")   # ancestor group lists the row; descendants
    return src.locator.split(" (cont")[0] == c.locator.split(" (cont")[0]   # same clause / tender condition


def clean(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[‐-―−]", "-", q)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", type=int, default=3)
    ap.add_argument("--max-train-queries", type=int, default=3000)
    a = ap.parse_args()
    rng = random.Random(7)
    ix = Index.build(BASE_MODEL)
    recs = [json.loads(l) for l in SYN.open(encoding="utf-8")]
    out = {"train": [], "dev": [], "test": []}
    queries = {"train": [], "dev": [], "test": []}   # for retrieval eval: (query, gold chunk id, kind)
    seen_q = set()
    for rec in recs:
        if rec["chunk_id"] not in ix.by_id:
            continue
        src = ix.chunks[ix.by_id[rec["chunk_id"]]]
        split = rec["split"]
        for q in rec["queries"]:
            text = clean(str(q["q"]))
            key = text.lower()
            if key in seen_q or VAGUE.search(text):
                continue  # vague queries ("one of these pumps") are answered equally well by sibling rows
            seen_q.add(key)
            queries[split].append({"q": text, "gold": src.id, "style": q.get("style", ""), "doc": src.doc, "kind": "pos"})
        for nm in rec.get("near_miss", []):
            text = clean(str(nm["q"]))
            if text.lower() in seen_q:
                continue
            seen_q.add(text.lower())
            queries[split].append({"q": text, "gold": src.id, "style": "near_miss", "doc": src.doc, "kind": "near_miss",
                                   "changed": nm.get("changed", "")})
    # cap train size for CPU budget (stratified by doc)
    if len(queries["train"]) > a.max_train_queries:
        rng.shuffle(queries["train"])
        queries["train"] = queries["train"][: a.max_train_queries]
    for split, qs in queries.items():
        for i, item in enumerate(qs):
            src = ix.chunks[ix.by_id[item["gold"]]]
            if item["kind"] == "near_miss":
                out[split].append({"query": item["q"], "passage": src.index_text, "label": 0.0, "kind": "near_miss"})
                continue
            out[split].append({"query": item["q"], "passage": src.index_text, "label": 1.0, "kind": "pos"})
            if split != "train":
                continue  # dev/test negatives come from the live candidate list at eval time
            r = retrieve(ix, item["q"], k=30, mode="hybrid", pool_all_docs=True)
            negs = [ix.chunks[h.idx] for h in r.hits if not same_answer(src, ix.chunks[h.idx])
                    and page_split(ix.chunks[h.idx].doc, ix.chunks[h.idx].pdf_page) == "train"]
            for c in negs[: a.hard]:
                out[split].append({"query": item["q"], "passage": c.index_text, "label": 0.0, "kind": "hard"})
            if i % 500 == 0:
                print(f"  {split}: {i}/{len(qs)}")
    for split in out:
        rng.shuffle(out[split])
        with (OUT / f"pairs_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in out[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (OUT / f"queries_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in queries[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kinds = {}
        for r in out[split]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(split, len(out[split]), kinds, "queries:", len(queries[split]))


if __name__ == "__main__":
    main()
