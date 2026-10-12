"""Generate synthetic (query, chunk) training data with an open-weight model (gpt-oss-120b, Apache 2.0).

For each sampled chunk the model writes queries the chunk ANSWERS (specific, paraphrase, cross-lingual) and
NEAR-MISS queries the chunk does NOT answer although it looks like it should (one attribute changed: a size,
a clause number, a period, a material). Near-misses are the training signal for the grounding gate: the
failure that matters most in rates and contracts is a confident answer with a real citation to the wrong row.

Only chunks on TRAIN pages feed training; DEV/TEST pages produce held-out synthetic eval queries.
Usage: uv run python adapt/gen_queries.py --per-doc gcc=400,bsr=1600,nit=60 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosorag.chunk import load_chunks  # noqa: E402
from cosorag.config import PROCESSED_DIR, ROOT  # noqa: E402
from cosorag.llm import LLM, parse_json  # noqa: E402
from cosorag.splits import page_split  # noqa: E402

OUT = ROOT / "adapt" / "data" / "synthetic.jsonl"

SYSTEM = """You create search queries for training a retrieval model over Indian construction documents.
Reasoning: low
You are given ONE passage with its citation header. Write realistic queries that a site engineer, estimator, contracts or claims manager at an Indian infrastructure contractor would type.

Return ONLY JSON:
{"queries": [{"q": "...", "style": "specific"}, {"q": "...", "style": "paraphrase"}, {"q": "...", "style": "crosslingual"}],
 "near_miss": [{"q": "...", "changed": "what was changed so the passage no longer answers it"}]}

Rules:
- "specific": answerable from this passage alone; name the distinguishing detail (size, material, unit, clause topic, amount, date) so that this passage, not a sibling, is the answer. Do NOT copy the item code or clause number in more than half of your queries.
- "paraphrase": same need in plain words or site jargon, do not copy more than 4 consecutive words from the passage.
- "crosslingual": if the passage is English, write the query in Hindi (Devanagari) or Hinglish (romanized Hindi mixed with English); if the passage is Hindi, write it in English.
- "near_miss": 1-2 queries that look like this passage should answer them but it does NOT: change exactly one attribute to a value the passage does not contain (e.g. 20 mm instead of 15 mm, Clause 10C instead of 10CC, 12 months instead of 9, a different material or work), or ask for a detail the passage lacks. They must be plausible questions, not nonsense.
- 6-25 words per query. No answers, only queries."""

DOC_HINT = {
    "gcc": "Passage from the CPWD General Conditions of Contract 2020 (a construction contract).",
    "bsr": "Passage from the Rajasthan Housing Board Basic Schedule of Rates 2024. A row's description inherits its parent's (joined with '>'). Rates are in rupees per unit.",
    "nit": "Passage from a Rajasthan PWD e-tender notice (Hindi, OCR of a scan) for the Yatri Niwas building work in Jaipur.",
}


def sample_chunks(per_doc: dict[str, int], seed: int = 13):
    chunks = load_chunks(PROCESSED_DIR / "chunks.jsonl.gz")
    rng = random.Random(seed)
    picked = []
    for doc, n in per_doc.items():
        pool = [c for c in chunks if c.doc == doc and len(c.text) >= 40]
        if doc == "bsr":  # stratify by chapter, prefer rated items, keep some group/notes chunks
            by_ch: dict[str, list] = {}
            for c in pool:
                by_ch.setdefault(c.meta.get("chapter", "text"), []).append(c)
            share = {k: max(15, int(n * len(v) / len(pool))) for k, v in by_ch.items()}
            for k, v in by_ch.items():
                rng.shuffle(v)
                picked += v[: share[k]]
        else:
            rng.shuffle(pool)
            picked += pool[:n]
    return picked


def gen_one(llm: LLM, c) -> dict | None:
    user = f"{DOC_HINT[c.doc]}\nCitation: {c.cite()}\n\nPassage:\n{c.text[:2200]}"
    for attempt in range(2):
        try:
            r = llm.chat(SYSTEM, user if attempt == 0 else user + "\n(Return valid JSON only.)", max_tokens=2400, temperature=0.7)
            d = parse_json(r.text)
            qs = [q for q in d.get("queries", []) if isinstance(q, dict) and 4 <= len(str(q.get("q", "")).split()) <= 40]
            nm = [q for q in d.get("near_miss", []) if isinstance(q, dict) and len(str(q.get("q", "")).split()) >= 4]
            if qs:
                return {"chunk_id": c.id, "doc": c.doc, "split": page_split(c.doc, c.pdf_page), "pdf_page": c.pdf_page,
                        "queries": qs, "near_miss": nm, "model": llm.name, "tokens": [r.input_tokens, r.output_tokens]}
        except Exception as e:  # noqa: BLE001  (logged and skipped; one bad chunk must not stop the run)
            print(f"[warn] {c.id}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", default="gcc=400,bsr=1600,nit=60")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="bedrock:openai.gpt-oss-120b-1:0")
    a = ap.parse_args()
    per_doc = {k: int(v) for k, v in (x.split("=") for x in a.per_doc.split(","))}
    chunks = sample_chunks(per_doc)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["chunk_id"] for l in OUT.open(encoding="utf-8")}
    todo = [c for c in chunks if c.id not in done]
    print(f"{len(chunks)} sampled, {len(todo)} to generate")
    llm = LLM(a.model)
    llm._bedrock() if llm.provider == "bedrock" else llm._openai()  # create the client before threads
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with ThreadPoolExecutor(a.workers) as ex, OUT.open("a", encoding="utf-8") as f:
        futs = {ex.submit(gen_one, llm, c): c for c in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n_ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} done, {n_ok} ok")
    print(f"wrote {n_ok} records -> {OUT}")


if __name__ == "__main__":
    main()
