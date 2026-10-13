"""BM25 + dense index over all chunks. Brute-force cosine is exact and takes milliseconds at ~10k chunks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .chunk import Chunk, load_chunks
from .config import INDEX_DIR, MODELS_DIR, PROCESSED_DIR
from .text import tokenize

BASE_MODEL = "intfloat/multilingual-e5-small"
FT_MODEL_DIR = MODELS_DIR / "e5-small-coso"


def default_model() -> str:
    """Fine-tuned model if present, else the base model. Override with COSORAG_EMBEDDER."""
    env = os.environ.get("COSORAG_EMBEDDER")
    if env:
        return env
    return str(FT_MODEL_DIR) if (FT_MODEL_DIR / "config.json").exists() else BASE_MODEL


_MODELS: dict = {}


def load_embedder(name: str):
    if name not in _MODELS:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        # COSORAG_EMBEDDER_DIR: a local copy of the same model (offline deployments); the vectors stay keyed by name
        _MODELS[name] = SentenceTransformer(os.environ.get("COSORAG_EMBEDDER_DIR") or name, device="cpu")
    return _MODELS[name]


def embed(texts: list[str], model_name: str, kind: str, batch_size: int = 32, show: bool = False) -> np.ndarray:
    """kind is 'query' or 'passage' (e5 prefixes)."""
    m = load_embedder(model_name)
    m.max_seq_length = 256 if kind == "passage" else 64
    return m.encode([f"{kind}: {t}" for t in texts], batch_size=batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=show).astype(np.float32)


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None, model_name: str | None):
        import bm25s

        self.chunks = chunks
        self.by_id = {c.id: i for i, c in enumerate(chunks)}
        self.vectors = vectors
        self.model_name = model_name
        self._qcache: dict = {}
        self.bm25 = bm25s.BM25(k1=1.2, b=0.75)
        self.bm25.index([tokenize(c.index_text) for c in chunks], show_progress=False)

    # ---- persistence -------------------------------------------------------------------------
    @staticmethod
    def vec_path(model_name: str) -> Path:
        tag = Path(model_name).name if os.path.exists(model_name) else model_name.replace("/", "__")
        return INDEX_DIR / f"vectors__{tag}.npy"

    @classmethod
    def build(cls, model_name: str | None = None, chunks_path: Path | None = None, force: bool = False) -> "Index":
        chunks = load_chunks(chunks_path or PROCESSED_DIR / "chunks.jsonl.gz")
        model_name = model_name or default_model()
        vp = cls.vec_path(model_name)
        meta = vp.with_suffix(".json")
        keys = [hashlib.sha1(c.index_text.encode("utf-8")).hexdigest()[:16] for c in chunks]
        cache: dict[str, np.ndarray] = {}
        if vp.exists() and meta.exists() and not force:   # incremental: only re-embed changed chunks
            old = json.loads(meta.read_text())
            cache = dict(zip(old.get("keys", []), np.load(vp).astype(np.float32)))
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            new = embed([chunks[i].index_text for i in missing], model_name, "passage", show=True)
            cache.update({keys[i]: v for i, v in zip(missing, new)})
        vectors = np.stack([cache[k] for k in keys]).astype(np.float32)
        if missing or not meta.exists():
            vp.parent.mkdir(parents=True, exist_ok=True)
            tmp = vp.with_suffix(".tmp.npy")   # write then rename: parallel readers never see half a file
            np.save(tmp, vectors.astype(np.float16))   # committed to git: 10k x 384 x 2 bytes ~ 8 MB
            tmp.replace(vp)
            meta.with_suffix(".tmp").write_text(json.dumps({"model": model_name, "keys": keys}))
            meta.with_suffix(".tmp").replace(meta)
        return cls(chunks, vectors, model_name)

    # ---- search ------------------------------------------------------------------------------
    def bm25_search(self, query: str, k: int = 50, docs: set[str] | None = None) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = np.argsort(-scores)
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) >= k:
                break
            if docs and self.chunks[i].doc not in docs:
                continue
            out.append((int(i), float(scores[i])))
        return out

    def dense_search(self, query: str, k: int = 50, docs: set[str] | None = None) -> list[tuple[int, float]]:
        if query not in self._qcache:   # per-document lists reuse the same query vector
            self._qcache[query] = embed([query], self.model_name, "query")[0]
        q = self._qcache[query]
        scores = self.vectors @ q
        order = np.argsort(-scores)
        out = []
        for i in order:
            if docs and self.chunks[i].doc not in docs:
                continue
            out.append((int(i), float(scores[i])))
            if len(out) >= k:
                break
        return out
