"""Cross-encoder reranker that doubles as a grounding gate.

A bi-encoder scores query and passage separately, so "20 mm pillar cock" and the 15 mm row look almost the
same. A cross-encoder reads them together and can learn that the size mismatch matters. We fine-tune a small
multilingual one (mMiniLM-L12, 118M params, CPU) on synthetic queries from this corpus, including near-miss
queries the passage does NOT answer, so its top score also tells us whether ANY retrieved passage answers.
"""
from __future__ import annotations

import json
import math
import os

from .config import MODELS_DIR

BASE_XENC = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
FT_XENC_DIR = MODELS_DIR / "xenc-coso"
_CACHE: dict = {}


def default_reranker() -> str | None:
    env = os.environ.get("COSORAG_RERANKER")
    if env:
        return None if env == "none" else env
    return str(FT_XENC_DIR) if (FT_XENC_DIR / "config.json").exists() else None


def load(name: str):
    if name.startswith("bedrock:"):   # hosted reranker: nothing to load
        return None
    if name not in _CACHE:
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        # released weights are fp16 to halve the download; CPU inference runs in fp32
        _CACHE[name] = CrossEncoder(name, device="cpu", max_length=256, model_kwargs={"torch_dtype": torch.float32})
    return _CACHE[name]


def gate_threshold(name: str) -> float | None:
    if name.startswith("bedrock:"):   # no threshold has been calibrated for a hosted reranker: no review flag
        return None
    p = os.path.join(name, "gate.json")
    return json.load(open(p))["threshold"] if os.path.exists(p) else None


def fused_order(items: list, scores: list[float], k: int = 60) -> list:
    """Reciprocal-rank fusion of the retriever's order (items arrive in it) with the cross-encoder's order.
    Re-sorting by the cross-encoder alone dropped rows it misread on long, multi-part questions from the
    context (gold-set citation hit 94% -> 88%); fused, a passage stays near the top if either ranker is sure."""
    xr = {i: r for r, i in enumerate(sorted(range(len(items)), key=lambda i: -scores[i]))}
    return [items[i] for i in sorted(range(len(items)), key=lambda i: -(1 / (k + i) + 1 / (k + xr[i])))]


def _bedrock_score(model_id: str, query: str, passages: list[str]) -> list[float]:
    """Cohere Rerank 3.5 on Bedrock (the live demo's reranker; it is not hosted in Mumbai, so Tokyo by default)."""
    import boto3

    if "client" not in _CACHE:
        _CACHE["client"] = boto3.client("bedrock-runtime", region_name=os.environ.get("COSORAG_RERANK_REGION", "ap-northeast-1"))
    body = {"query": query, "documents": passages, "top_n": len(passages), "api_version": 2}
    res = json.loads(_CACHE["client"].invoke_model(modelId=model_id, body=json.dumps(body))["body"].read())["results"]
    out = [0.0] * len(passages)
    for r in res:
        out[r["index"]] = float(r["relevance_score"])
    return out


def score(name: str, query: str, passages: list[str]) -> list[float]:
    """Relevance probabilities in [0, 1] (sigmoid of the logit)."""
    if not passages:
        return []
    if name.startswith("bedrock:"):
        return _bedrock_score(name.split(":", 1)[1], query, passages)
    logits = load(name).predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False,
                                activation_fn=None, apply_softmax=False)
    return [1 / (1 + math.exp(-float(x))) for x in logits]
