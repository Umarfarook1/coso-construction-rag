"""Run the gold set (development questions only) through alternative generators, everything else unchanged.

Retrieval, the fine-tuned reranker and the verifier are the submitted system's; only the model that writes the
answer changes. The index and reranker load once and the questions run in parallel threads, so five generators
take minutes, not hours. Results land in runs/gen-<slug>[COSORAG_RUN_TAG]/ in the harness format (`harness.py report` reads them).
The blind set is not used: it stays a one-shot estimate for the submitted configuration.
  uv run python evals/generators.py                 # all generators below
  uv run python evals/generators.py gpt-oss-20b     # one
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import RUNS, load_gold, report, score_answer  # noqa: E402
from judge import judge  # noqa: E402

# slug -> (Bedrock model id, region); same list as the web view's model picker (src/cosorag/models.py)
from cosorag.models import MODELS  # noqa: E402


def run(slugs: list[str], workers: int = 6) -> None:
    from cosorag import rerank as rr
    from cosorag.answer import answer
    from cosorag.index import Index

    ix = Index.build()
    xenc = rr.default_reranker()
    rr.load(xenc)                      # load once, before threads share it
    ix.dense_search("warm up", 1)
    gold = load_gold()
    for slug in slugs:
        m = MODELS[slug]
        llm = m.client()
        t0 = time.time()

        def one(g):
            a = answer(ix, g["question"], llm=llm)
            row = score_answer(g, a)
            row.update(question=g["question"], gold_answer=g["answer"], raw=a.raw)
            v = judge(g["question"], g, a.answer)
            row["judge"], row["judge_verdict"], row["judge_reason"] = v["score"], v.get("verdict"), v.get("reason")
            return row

        with cf.ThreadPoolExecutor(workers) as ex:
            rows = list(ex.map(one, gold))
        out = RUNS / f"gen-{slug}{os.environ.get('COSORAG_RUN_TAG', '')}"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "results.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "config.json").write_text(json.dumps({
            "llm": llm.name, "region": llm.region, "rerank": True, "embedder": ix.model_name,
            "reranker": xenc if xenc.startswith("bedrock:") else "models/xenc-coso",
            "gate_threshold": rr.gate_threshold(xenc), "mode": "hybrid", "split": "all", "n": len(rows)}, indent=2))
        print(f"{slug}: {len(rows)} questions in {time.time() - t0:.0f}s", flush=True)
        print(report(out), flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or [s for s in MODELS if s != "gpt-oss-120b"])
