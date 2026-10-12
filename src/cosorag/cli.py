"""cosorag command line.

  cosorag ask "What is the EMD for the Yatri Niwas tender?"     answer with citations
  cosorag ask --no-llm "..."                                     retrieval only (no model, no keys)
  cosorag ingest                                                 PDFs -> data/processed/chunks.jsonl.gz
  cosorag index                                                  build/refresh the dense index
"""
from __future__ import annotations

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _print_answer(a, show_sources: bool) -> None:
    from rich.console import Console
    from rich.panel import Panel

    con = Console()
    status = "[green]verified[/]" if a.checks.get("ok") else ("[yellow]check warnings[/]" if a.checks else "[dim]retrieval only[/]")
    if not a.answerable:
        status = "[cyan]not answerable from the documents[/]"
    con.print(Panel(a.answer, title=f"Answer ({status})", subtitle=a.model or "", expand=True))
    for c in a.citations:
        con.print(f"  [bold][S{c['n']}][/] {c['cite']}  [dim]({c['chunk_id']})[/]")
        if c.get("quote"):
            con.print(f"       [dim]\"{c['quote'][:220]}\"[/]")
    if a.checks.get("unsupported_numbers"):
        con.print(f"  [yellow]numbers not found in the cited sources (possibly computed): {a.checks['unsupported_numbers']}[/]")
    if a.checks.get("bad_quotes"):
        con.print(f"  [yellow]{len(a.checks['bad_quotes'])} quote(s) could not be matched to their source[/]")
    for lk in (a.retrieval.id_lookups if a.retrieval else []):
        if not lk["found"]:
            con.print(f"  [cyan]{lk['id']} does not exist in the indexed documents[/]")
    if a.gate:
        con.print(f"  [dim]grounding gate: {a.gate}[/]")
    if show_sources:
        con.print("\n[bold]Retrieved sources[/]")
        for i, c in enumerate(a.sources, 1):
            con.print(f"  S{i} {c.cite()}  [dim]{c.text[:110].replace(chr(10), ' ')}[/]")
    t = a.timings
    con.print(f"[dim]retrieve {t.get('retrieve_s', 0)}s | rerank {t.get('rerank_s', 0)}s | generate {t.get('generate_s', 0)}s"
              f"{' (cached)' if t.get('cached') else ''}[/]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cosorag", description="Cited Q&A over Indian construction documents")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--no-llm", action="store_true", help="retrieval only, no model call")
    p_ask.add_argument("--llm", default=None, help="override COSORAG_LLM, e.g. openai:qwen2.5:7b or bedrock:openai.gpt-oss-120b-1:0")
    p_ask.add_argument("--doc", action="append", choices=["gcc", "bsr", "nit"], help="restrict to a document")
    p_ask.add_argument("--no-rerank", action="store_true", help="skip the fine-tuned reranker / grounding gate")
    p_ask.add_argument("--sources", action="store_true", help="print all retrieved sources")
    p_ing = sub.add_parser("ingest", help="parse the PDFs into chunks")
    p_ing.add_argument("--doc", action="append", choices=["gcc", "bsr", "nit"])
    p_show = sub.add_parser("show", help="render a cited page with the region highlighted")
    p_show.add_argument("chunk_id", help="e.g. bsr:S:2.33.1 or gcc:Clause-2:0 (printed after each answer)")
    sub.add_parser("serve", help="local web view at http://127.0.0.1:8000")
    p_idx = sub.add_parser("index", help="build the dense index")
    p_idx.add_argument("--model", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "ingest":
        from .ingest import run

        run(set(a.doc) if a.doc else None)
        return
    if a.cmd == "serve":
        from .serve import main as serve

        serve()
        return
    if a.cmd == "show":
        from .chunk import load_chunks
        from .config import PROCESSED_DIR
        from .show import render

        chunks = {c.id: c for c in load_chunks(PROCESSED_DIR / "chunks.jsonl.gz")}
        if a.chunk_id not in chunks:
            raise SystemExit(f"unknown chunk id {a.chunk_id}")
        print(render(chunks[a.chunk_id]))
        return
    if a.cmd == "index":
        from .index import Index

        ix = Index.build(a.model)
        print(f"index ready: {len(ix.chunks)} chunks, model {ix.model_name}")
        return
    if a.cmd == "ask":
        from .answer import answer
        from .index import Index
        from .llm import LLM

        if a.llm:
            os.environ["COSORAG_LLM"] = a.llm
        ix = Index.build()
        from .llm import llm_configured

        llm = None if a.no_llm else (LLM() if llm_configured() else None)
        if llm is None and not a.no_llm:
            print("No language model configured (see .env.example): showing the cited passages only.")
        q, docs = " ".join(a.question), set(a.doc) if a.doc else None
        try:
            res = answer(ix, q, llm=llm, docs=docs, rerank=not a.no_rerank)
        except Exception as e:  # noqa: BLE001  (no credentials, no network, model not enabled ...)
            if llm is None:
                raise
            print(f"Model call failed ({type(e).__name__}: {str(e)[:120]}). Showing the cited passages only.")
            res = answer(ix, q, llm=None, docs=docs, rerank=not a.no_rerank)
        _print_answer(res, a.sources or a.no_llm)


if __name__ == "__main__":
    main()
