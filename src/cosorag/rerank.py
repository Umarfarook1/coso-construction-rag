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
    if name not in _CACHE:
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        # released weights are fp16 to halve the download; CPU inference runs in fp32
        _CACHE[name] = CrossEncoder(name, device="cpu", max_length=256, model_kwargs={"torch_dtype": torch.float32})
    return _CACHE[name]


def gate_threshold(name: str) -> float | None:
    p = os.path.join(name, "gate.json")
    return json.load(open(p))["threshold"] if os.path.exists(p) else None


def score(name: str, query: str, passages: list[str]) -> list[float]:
    """Relevance probabilities in [0, 1] (sigmoid of the logit)."""
    if not passages:
        return []
    logits = load(name).predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False,
                                activation_fn=None, apply_softmax=False)
    return [1 / (1 + math.exp(-float(x))) for x in logits]
