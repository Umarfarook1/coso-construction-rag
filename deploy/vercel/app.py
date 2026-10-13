"""Vercel entrypoint for the web view. Assembled into a deployable folder by deploy/vercel/stage.py.

The live demo calls every model through the Amazon Bedrock API and ships no model weights: the fine-tuned reranker
and the local embedder need PyTorch and about 2 GB of memory, beyond the free plan. Search uses the precomputed
Cohere vectors in data/index; the cache and page renders go to /tmp (the rest of the filesystem is read-only).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("COSORAG_CACHE_DIR", "/tmp/cosorag/cache")
os.environ.setdefault("COSORAG_VIEWS_DIR", "/tmp/cosorag/views")
os.environ.setdefault("COSORAG_EMBEDDER", "bedrock:cohere.embed-multilingual-v3")
os.environ.setdefault("COSORAG_RERANKER", "bedrock:cohere.rerank-v3-5:0")
sys.path.insert(0, str(HERE / "src"))

from cosorag.serve import create_app  # noqa: E402

app = create_app()
