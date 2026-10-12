"""Minimal local web view: ask a question, read the cited answer, click a citation to see the page highlighted.
  uv run cosorag serve      ->  http://127.0.0.1:8000
"""
from pathlib import Path

from pydantic import BaseModel

HTML = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


class Ask(BaseModel):   # module level: FastAPI must resolve it as the JSON body
    question: str
    rerank: bool = True


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from .answer import answer
    from .config import RAW_DIR, DOCS
    from .index import Index
    from .llm import LLM
    from .show import render

    app = FastAPI(title="cosorag")
    state: dict = {}

    def ix():
        if "ix" not in state:
            state["ix"] = Index.build()
        return state["ix"]

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTML

    @app.post("/ask")
    def ask(q: Ask):
        from .llm import llm_configured

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
    def page(chunk_id: str):
        i = ix().by_id.get(chunk_id)
        if i is None:
            raise HTTPException(404, "unknown chunk")
        return FileResponse(render(ix().chunks[i]))

    return app


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
