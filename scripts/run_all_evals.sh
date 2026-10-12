#!/usr/bin/env bash
# The exact procedure behind evals/RESULTS.md (run after training; code and chunks frozen).
# CPU-only steps need no keys; the end-to-end runs need a model (Bedrock here) and the judge.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. adaptation on held-out synthetic queries: pick the gate threshold on DEV, then report on TEST
uv run python adapt/eval_reranker.py --split dev --n 300 --write-gate
uv run python adapt/eval_reranker.py --split test --n 400

# 2. retrieval ablation and gate transfer on the gold set (no LLM)
uv run python evals/harness.py retrieval --split all --out evals/results/retrieval_gold.json
uv run python evals/gate_gold.py

# 3. end to end on the gold set: no reranker, off-the-shelf reranker, fine-tuned reranker (submitted)
COSORAG_RERANKER=none uv run python evals/harness.py e2e --name v2-norerank --split all
COSORAG_ALLOW_NO_GATE=1 COSORAG_RERANKER=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  uv run python evals/harness.py e2e --name v2-rerank-base-fused --split all
uv run python evals/harness.py e2e --name final --split all
# runs/v2-rerank-base and runs/*-v1-gate-in-prompt come from commit d56bd5e, where answer() re-sorted by the
# cross-encoder alone and put a "no source answers" note into the prompt below the gate threshold
# (same commands as above at that commit; see DESIGN.md section 6 for why it changed).

# 4. blind set, once per configuration
COSORAG_RERANKER=none uv run python evals/harness.py e2e --name blind-norerank --split all --gold evals/gold_blind
COSORAG_ALLOW_NO_GATE=1 COSORAG_RERANKER=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 uv run python evals/harness.py e2e --name blind-rerank-base-fused --split all --gold evals/gold_blind
uv run python evals/harness.py e2e --name blind-final --split all --gold evals/gold_blind

# 5. OCR and tables, then the results page
uv run python evals/ocr_report.py
uv run python evals/table_eval.py
uv run python evals/bsr_crosscheck.py
uv run python evals/make_results.py
