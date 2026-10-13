"""Minimal web view: ask a question, read the cited answer, click a citation to see the page highlighted.
  uv run cosorag serve      ->  http://127.0.0.1:8000
For a public deployment (deploy/vercel): COSORAG_ACCESS_CODE makes /ask and /page require that code, and
COSORAG_DAILY_LIMIT caps questions per day (per running instance) so a leaked link cannot run up model costs.
"""
import hmac
import os
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

HTML = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


class Ask(BaseModel):   # module level: FastAPI must resolve it as the JSON body
    question: str = Field(min_length=1, max_length=500)
    rerank: bool = True


def create_app():
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from .answer import answer
    from .config import RAW_DIR, DOCS
    from .index import Index
    from .llm import LLM
    from .show import render

    app = FastAPI(title="cosorag")
    state: dict = {}
    build_lock, count_lock = threading.Lock(), threading.Lock()
    access = os.environ.get("COSORAG_ACCESS_CODE", "")
    limit = int(os.environ.get("COSORAG_DAILY_LIMIT") or 0)
    usage = {"day": "", "n": 0}

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

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTML

    @app.post("/ask")
    def ask(q: Ask, x_access_code: str | None = Header(default=None)):
        from .llm import llm_configured

        check(x_access_code)
        spend()
        llm, notice = (LLM() if llm_configured() else None), ""
        try:
            a = answer(ix(), q.question, llm=llm, rerank=q.rerank)
        except Exception as e:  # noqa: BLE001  (same fallback as the CLI: expired key, no network ...)
            if llm is None:
                raise
            notice = f"Model call failed ({type(e).__name__}); showing the cited passages only."
            a = answer(ix(), q.question, llm=None, rerank=q.rerank)
        pdfs = {d: (RAW_DIR / m["file"]).exists() for d, m in DOCS.items()}
        return {
            "answer": a.answer, "answerable": a.answerable, "model": a.model, "gate": a.gate, "timings": a.timings,
            "notice": notice,
            "checks": {k: v for k, v in a.checks.items() if k in ("ok", "unsupported_numbers", "derived_numbers")},
            "citations": [{**c, "viewable": pdfs.get(c["doc"], False)} for c in a.citations],
            "missing_ids": [lk["id"] for lk in (a.retrieval.id_lookups if a.retrieval else []) if not lk["found"]],
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
