"""Web view: a chat over the three documents, with the cited page highlighted in a side panel.
  uv run cosorag serve      ->  http://127.0.0.1:8000
The reader can pick the model that writes the answer (cosorag.models); retrieval, the fine-tuned reranker and the
verifier are the same for every model. Every message is routed first (cosorag.guard): greetings, questions about the
assistant and unrelated requests get a fixed reply without a search; follow-ups are rewritten into standalone questions.
For a public deployment (deploy/vercel): COSORAG_ACCESS_CODE makes /ask, /page and /warm require that code, and
COSORAG_DAILY_LIMIT caps questions per day (per running instance) so a leaked link cannot run up model costs.
"""
import hmac
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

WEB = Path(__file__).parent / "web"
HTML = (WEB / "index.html").read_text(encoding="utf-8")
ASSETS = {p.name: p for p in (WEB / "assets").glob("*") if p.is_file()}
TYPES = {".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".json": "application/json"}
NAMES = {"bedrock:cohere.embed-multilingual-v3": "Cohere Embed Multilingual v3 (Bedrock)",
         "intfloat/multilingual-e5-small": "multilingual-e5-small (local CPU)",
         "bedrock:cohere.rerank-v3-5:0": "Cohere Rerank 3.5 (Bedrock)"}


def pipeline() -> dict:
    """Which search and rerank components this server runs, without loading them."""
    from .index import default_model
    from .rerank import default_reranker

    emb, rr = default_model(), default_reranker()
    kind = None if rr is None else ("bedrock" if rr.startswith("bedrock:") else "fine-tuned")
    return {"embedder": NAMES.get(emb, Path(emb).name),
            "reranker": NAMES.get(rr, "Fine-tuned cross-encoder (local CPU)") if rr else None, "reranker_kind": kind}


class Turn(BaseModel):
    question: str = Field(max_length=500)
    answer: str = Field(max_length=4000)


class Ask(BaseModel):   # module level: FastAPI must resolve it as the JSON body
    question: str = Field(min_length=1, max_length=500)
    rerank: bool = True
    model: str | None = Field(default=None, max_length=40)
    history: list[Turn] = Field(default_factory=list, max_length=6)


def model_options() -> tuple[list[dict], str | None]:
    """The models a reader may pick, and the default. A COSORAG_LLM override (e.g. a local Ollama model) replaces
    the Bedrock list with that one model, so a reviewer's own endpoint keeps working."""
    from .llm import llm_configured
    from .models import DEFAULT, MODELS

    if not llm_configured():
        return [], None
    spec = os.environ.get("COSORAG_LLM")
    if spec:
        return [{"slug": "configured", "label": spec.split(":", 1)[1], "maker": spec.split(":", 1)[0],
                 "open_weights": None, "where": "", "blurb": "Set by COSORAG_LLM", "gold": None}], "configured"
    return [m.public() for m in MODELS.values()], DEFAULT


def create_app():
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from . import rerank as rr
    from .answer import answer, prepare
    from .chat import standalone
    from .guard import route as route_message, rules
    from .config import RAW_DIR, DOCS
    from .index import Index
    from .llm import LLM
    from .models import MODELS
    from .show import render

    app = FastAPI(title="cosorag")
    state: dict = {}
    clients: dict = {}
    build_lock, count_lock = threading.Lock(), threading.Lock()
    access = os.environ.get("COSORAG_ACCESS_CODE", "")
    limit = int(os.environ.get("COSORAG_DAILY_LIMIT") or 0)
    usage = {"day": "", "n": 0}
    pool = ThreadPoolExecutor(max_workers=4)

    def ix():
        if "ix" not in state:
            with build_lock:   # two cold-start requests must not both load the models (memory)
                if "ix" not in state:
                    state["ix"] = Index.build()
        return state["ix"]

    def check(code: str | None) -> None:
        if access and not hmac.compare_digest((code or "").encode(), access.encode()):
            raise HTTPException(401, "access code required")

    def spend() -> None:
        if not limit:
            return
        with count_lock:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if usage["day"] != today:
                usage.update(day=today, n=0)
            if usage["n"] >= limit:
                raise HTTPException(429, "daily question limit reached; try again tomorrow")
            usage["n"] += 1

    def client(slug: str | None):
        options, default = model_options()
        if not options:
            return None, None
        slug = slug or default
        if slug not in {o["slug"] for o in options}:
            raise HTTPException(422, f"unknown model {slug!r}")
        if slug not in clients:
            clients[slug] = LLM() if slug == "configured" else MODELS[slug].client()
        return clients[slug], slug

    def router(llm):
        """The small, fast model that routes messages: Claude Haiku 5.5 on Bedrock, or the configured endpoint."""
        spec = os.environ.get("COSORAG_ROUTER", "claude-haiku-5-5")
        if llm is None or spec == "off":
            return None
        if os.environ.get("COSORAG_LLM") or spec not in MODELS:
            return llm
        if "router" not in clients:
            clients["router"] = MODELS[spec].client()
        return clients["router"]

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTML

    @app.get("/assets/{name}")
    def asset(name: str):
        p = ASSETS.get(name)
        if p is None:
            raise HTTPException(404, "not found")
        return FileResponse(p, media_type=TYPES.get(p.suffix, "application/octet-stream"),
                            headers={"cache-control": "public, max-age=31536000, immutable"})

    @app.get("/models")
    def models():
        options, default = model_options()
        return {"models": options, "default": default, "access_required": bool(access), "pipeline": pipeline(),
                "pdfs": {d: (RAW_DIR / m["file"]).exists() for d, m in DOCS.items()}}

    @app.post("/warm")
    def warm(x_access_code: str | None = Header(default=None)):
        """Load the index, the embedder and the reranker so the reader's first question does not pay for it."""
        check(x_access_code)
        t0 = time.time()
        index = ix()
        index.dense_search("warm up", 1)
        xenc = rr.default_reranker()
        if xenc:
            rr.load(xenc)
        return {"ready": True, "seconds": round(time.time() - t0, 2)}

    @app.post("/ask")
    def ask(q: Ask, x_access_code: str | None = Header(default=None)):
        check(x_access_code)
        t_start = time.time()
        llm, slug = client(q.model)
        history = [t.model_dump() for t in q.history]
        notice, timings, prepared, text = "", {}, None, q.question.strip()
        if rules(text) is None:
            spend()   # every message that reaches a model counts against the daily limit; plain greetings are free
        rl = router(llm)
        if rl is not None and not history and rules(text) is None:
            # A first question needs no rewrite, so the search starts while the router decides. A greeting or an
            # off-topic message is answered as soon as the router returns; its search result is never used.
            fut = pool.submit(lambda: prepare(ix(), text, rerank=q.rerank))
            r = route_message(rl, text, history)
            if r.kind == "docs":
                try:
                    prepared = fut.result()
                except Exception:  # noqa: BLE001  (answer() repeats the search and reports the error properly)
                    prepared = None
            timings["route_parallel"] = True
        else:
            r = route_message(rl, text, history)
        timings["route_s"] = r.seconds
        if r.kind != "docs":
            label = next((o["label"] for o in model_options()[0] if o["slug"] == slug), "the selected model")
            return {"question": q.question, "standalone": "", "route": r.kind, "route_by": r.by, "lang": r.lang,
                    "answer": r.reply(label), "answerable": None, "model": None, "model_slug": None, "gate": "",
                    "gate_score": None, "flag": False, "timings": {**timings, "total_s": round(time.time() - t_start, 3)},
                    "notice": "", "checks": {},
                    "citations": [], "missing_ids": [], "passages": []}
        asked = r.question
        if llm is not None and history and r.by == "fallback":
            t0 = time.time()   # the router's reply was unusable: rewrite the follow-up with the writer, as before
            try:
                asked = standalone(llm, history, asked)
            except Exception:  # noqa: BLE001  (the rewrite is a convenience; the original question still works)
                asked = q.question.strip()
            timings["rewrite_s"] = round(time.time() - t0, 3)
        try:
            a = answer(ix(), asked, llm=llm, rerank=q.rerank, prepared=prepared)
        except Exception as e:  # noqa: BLE001  (same fallback as the CLI: expired key, no network ...)
            if llm is None:
                raise
            notice = f"The model call failed ({type(e).__name__}); showing the cited passages only."
            a, slug = answer(ix(), asked, llm=None, rerank=q.rerank, prepared=prepared), None
        index = ix()
        pdfs = {d: (RAW_DIR / m["file"]).exists() for d, m in DOCS.items()}
        passages = []
        for h in (a.retrieval.hits if a.retrieval else [])[:8]:
            c = index.chunks[h.idx]
            passages.append({"chunk_id": c.id, "label": c.label(), "doc": c.doc, "xenc": h.sources.get("xenc"),
                             "pinned": "id" in h.sources,
                             "ranks": {k: v for k, v in h.sources.items() if k in ("bm25", "dense")}})
        m = re.search(r"score ([\d.]+)", a.gate or "")
        return {
            "question": q.question, "standalone": asked if asked != q.question.strip() else "",
            "route": "docs", "route_by": r.by, "lang": r.lang,
            "answer": a.answer, "answerable": a.answerable, "model": a.model, "model_slug": slug,
            "gate": a.gate, "gate_score": float(m.group(1)) if m else None, "flag": "low confidence" in (a.gate or ""),
            "timings": {**timings, **a.timings, "total_s": round(time.time() - t_start, 3)}, "notice": notice,
            "checks": {k: v for k, v in a.checks.items() if k in ("ok", "unsupported_numbers", "derived_numbers")},
            "citations": [{**c, "viewable": pdfs.get(c["doc"], False)} for c in a.citations],
            "missing_ids": [lk["id"] for lk in (a.retrieval.id_lookups if a.retrieval else []) if not lk["found"]],
            "passages": passages,
        }

    @app.get("/page/{chunk_id:path}")
    def page(chunk_id: str, x_access_code: str | None = Header(default=None)):
        check(x_access_code)
        i = ix().by_id.get(chunk_id)
        if i is None:
            raise HTTPException(404, "unknown chunk")
        try:
            return FileResponse(render(ix().chunks[i]), headers={"cache-control": "private, max-age=3600"})
        except FileNotFoundError:
            raise HTTPException(404, "source PDF not available") from None

    return app


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
