# Web view on Vercel

The same FastAPI app as `uv run cosorag serve`, as one Python function in Mumbai (`bom1`). The live demo calls
every model through the Amazon Bedrock API and ships no model weights.

```bash
uv run python evals/all_metrics.py                # writes the metrics the How it works page shows
uv run python deploy/vercel/stage.py              # assembles dist/coso-rag-web (~50 MB: code, chunks, vectors, 3 PDFs)
cd dist/coso-rag-web
vercel link --yes --project coso-rag-web
vercel env add AWS_BEARER_TOKEN_BEDROCK production --sensitive
vercel env add COSORAG_ACCESS_CODE production --sensitive   # the page asks for it; links can carry ?code=...
vercel env add COSORAG_DAILY_LIMIT production               # questions per day per running instance
vercel deploy --prod
```

What is different from running it locally, and why:

* **No local models.** The fine-tuned reranker and the e5 embedder run on PyTorch; with both, a function needs
  about 700 MB of model files and about 2 GB of memory, more than the Hobby plan gives with headroom. The live
  demo uses Bedrock-hosted models instead: Cohere Embed Multilingual v3 (Mumbai) for query vectors, Cohere
  Rerank 3.5 (Tokyo, the nearest region that hosts it) for reranking, and the chosen answer writer (Claude
  Sonnet 5.5 by default). The 10,025 passage vectors are precomputed and shipped (20 MB). On the gold set this
  configuration scores 92.7% judged correct, on the blind set 96.7% (`evals/METRICS.md`).
* **The fine-tuned model** stays available: GitHub release v1.0 (`xenc-coso.zip`), fetched by
  `scripts/download_model.py` and used by `cosorag serve` and `cosorag ask` locally.
* **Read-only filesystem.** The answer cache and page renders go to `/tmp` (`COSORAG_CACHE_DIR`,
  `COSORAG_VIEWS_DIR`); nothing is written into the bundle.
* **Access code and daily cap**, because the function calls paid models with the owner's key. The cap is
  counted per running instance, so it is a brake, not an exact quota; an exact quota needs a shared store.
* **Cold start** loads the passages, BM25 and the vectors: a few seconds. The page also calls `/warm` when it
  opens, so the reader's first question does not pay for it.
