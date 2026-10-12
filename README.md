# Cited Q&A over Indian construction documents

Ask a question about a construction contract, an 824-page schedule of rates or a scanned Hindi tender notice,
and get an answer that points to the exact clause, item row or tender condition, with page numbers, or a clear
"the documents do not say".

```
$ cosorag ask "What is the EMD for the Yatri Niwas tender if I am registered outside the division?"
{{EXAMPLE_NIT}}
```

**Documents** (from the COSO test corpus): CPWD General Conditions of Contract 2020 (born-digital contract),
RHB Basic Schedule of Rates 2024 (824 pages of rate tables), Rajasthan PWD e-tender notice 68/2024-25 (6-page
image-only Hindi scan). Why these three and what makes each hard: [DESIGN.md](DESIGN.md#1-the-documents-and-why-they-are-hard).

**Model adaptation:** a multilingual cross-encoder fine-tuned on this corpus (CPU, ~1.5 h) that reranks
candidates and acts as a grounding gate, trained to tell a real question from a near-miss (same item, wrong
size). [Why this component](DESIGN.md#6-model-adaptation-a-fine-tuned-cross-encoder-as-reranker-and-grounding-gate).

## Results

{{RESULTS_TABLE}}

Full tables: [evals/RESULTS.md](evals/RESULTS.md). Method and caveats: [DESIGN.md section 7](DESIGN.md#7-evaluation).

## Setup

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). CPU only; about 1.5 GB of disk for the environment and models.

```bash
git clone <this repo> coso-rag && cd coso-rag      # a short path avoids Windows' 260-character limit
uv sync                                           # installs CPU-only torch
uv run python scripts/download_model.py           # fine-tuned reranker (~235 MB); skip it to run without the gate
```

The processed documents (chunks, OCR output, translations) and the search index are committed, so no OCR,
PDF parsing or embedding is needed to ask questions.

### Ask a question

Pick one way to reach a language model (copy `.env.example` to `.env`):

| Option | What to set | Notes |
|---|---|---|
| No model at all | nothing | `cosorag ask --no-llm "..."` returns the cited passages only |
| Any OpenAI-compatible endpoint | `COSORAG_LLM=openai:<model>`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` | e.g. Ollama `openai:qwen2.5:7b-instruct` at `http://localhost:11434/v1`, Groq, vLLM, OpenRouter |
| AWS Bedrock (used for the results) | `AWS_BEARER_TOKEN_BEDROCK` | gpt-oss-120b (open-weight) in ap-south-1 Mumbai |

```bash
uv run cosorag ask "What is the rate of a 15 mm CP brass pillar cock?"
uv run cosorag ask "मानसून में भी काम पूरा करने की अवधि क्या है?"
uv run cosorag ask --sources "Who decides the compensation for delay under the contract?"
uv run cosorag show bsr:S:2.33.1          # renders the cited page with the row highlighted (needs the PDFs)
uv run cosorag serve                      # local web view at http://127.0.0.1:8000: click a citation to see the page
```

More questions with their real answers, including the ones that should fail: [EXAMPLES.md](EXAMPLES.md).

### Reproduce the evaluation

```bash
uv run python evals/harness.py report runs/final          # reprint the reported run, no keys needed
uv run python evals/harness.py retrieval --split all      # retrieval ablation, CPU only, no keys
uv run python adapt/eval_reranker.py --split test         # adaptation before/after on held-out pages, no keys
uv run python evals/ocr_eval.py                           # OCR field accuracy on the Hindi scan
uv run python evals/harness.py e2e --name mine --split all  # full end-to-end run (needs a model + Bedrock for the judge)
uv run pytest -q
```

### Rebuild everything from the PDFs

```bash
uv run --with gdown python scripts/download_corpus.py     # the COSO corpus into data/raw/
uv run cosorag ingest                                     # parse contract + rates, segment the tender
uv run python scripts/ocr_nit.py                          # re-read the scan (Bedrock vision models)
uv run cosorag index                                      # embed chunks (about 10 min on a laptop CPU)
uv run python adapt/gen_queries.py                        # synthetic training queries (gpt-oss-120b)
uv sync --extra train
uv run python adapt/build_pairs.py && uv run python adapt/train_reranker.py   # ~95 min on a 4-core CPU
uv run python adapt/eval_reranker.py --split dev --write-gate
```

## Repository map

| Path | What |
|---|---|
| `src/cosorag/ingest/` | `gcc.py` clause parser, `bsr.py` rate-row parser, `nit.py` tender segmentation, `vote.py` OCR number voting |
| `src/cosorag/textlayer.py` | per-page check: trust the PDF text layer or re-OCR |
| `src/cosorag/retrieve.py`, `index.py` | exact id lookup, BM25 + dense, RRF, per-document lists |
| `src/cosorag/rerank.py` | cross-encoder reranker and grounding gate |
| `src/cosorag/answer.py` | prompt, quote-then-answer, deterministic verifier |
| `src/cosorag/show.py`, `serve.py` | page render with the cited region highlighted, local web view |
| `scripts/` | corpus download, tender OCR, model download/packaging, `run_all_evals.sh` (the exact evaluation procedure) |
| `adapt/` | synthetic data, pair mining, training, before/after evaluation |
| `evals/` | gold set, blind set, harness, judge, OCR evaluation, results |
| `data/processed/` | chunks, OCR readings, translations (committed) |

## Known limitations

The short version (details and fixes in [DESIGN.md section 9](DESIGN.md#9-known-limitations)):

- **Rate notes are separate chunks.** "Add 50% for coloured chinaware over items 1.1 to 1.7" is only applied if
  retrieval brings the note along with the rate; note scopes are not yet linked to the items they modify.
- **About 3% of rate rows** (unnumbered variants, rate ranges, multi-column hire charges) are stored as one record
  with several rates, and defects in the schedule itself (same material at two rates, a Rs 0.00 rate) are reproduced.
- **Tender citations have no highlight box.** The scan was read by vision models, which return text without
  coordinates, so a tender citation points to the page and condition only.
- **Schedule F is blank in the contract template,** so "who decides" and "what percentage" questions often end in
  "the template leaves this to Schedule F". That is correct for this document, and useless for a real contract.
- **Small evaluation.** 96 development questions and 30 blind ones (intervals of about plus or minus 8 and 15
  points), drafted with an AI assistant from the pages, not by an estimator or contracts manager; an LLM judges correctness.
- **Keyword routing between documents** is fine for three documents and would not survive three hundred.

I used Claude Code as a pair programmer for implementation and first drafts; the design calls and the
evaluation choices are mine.
