# Web view on Vercel

The same FastAPI app as `uv run cosorag serve`, as one Python function (Fluid compute, Mumbai region next
to the Bedrock model).

```bash
uv run python deploy/vercel/stage.py          # assembles dist/coso-rag-web (~40 MB: code, chunks, vectors, 3 PDFs)
cd dist/coso-rag-web
vercel link --yes --project coso-rag-web
vercel env add AWS_BEARER_TOKEN_BEDROCK production --sensitive
vercel env add COSORAG_ACCESS_CODE production --sensitive   # the page asks for it; links can carry ?code=...
vercel env add COSORAG_DAILY_LIMIT production               # questions per day per running instance
vercel deploy --prod
```

What is different from running it locally, and why:

* **Models are fetched at build time** (`build.py`): the Hobby plan uploads at most 100 MB of source, and the
  two models are about 700 MB. The fine-tuned reranker comes from this repository's v1.0 release, the
  embedder from Hugging Face at the revision the committed vectors were computed with.
* **Large function.** PyTorch and the two models put the bundle well above the standard 500 MB Python limit,
  so the build switches to Vercel's large functions (beta, up to 5 GB).
* **Read-only filesystem.** The answer cache and page renders go to `/tmp` (`COSORAG_CACHE_DIR`,
  `COSORAG_VIEWS_DIR`); nothing is written into the bundle.
* **Access code and daily cap**, because the function calls a paid model with the owner's key. The cap is
  counted per running instance, so it is a brake, not an exact quota; an exact quota needs a shared store.
* **Cold start** loads PyTorch, the index and both models: about 20 s on the first question after a quiet
  spell, then 2-6 s per question. Memory is tight for the Hobby plan's 2 GB (the loaded process measures
  about 2.2 GB on Windows); on Vercel it has run without memory errors, but there is little headroom.
