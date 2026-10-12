"""Ingestion: raw PDFs -> data/processed/chunks.jsonl (committed, so reviewers do not need OCR or the PDFs)."""
from __future__ import annotations

from collections import Counter

from ..chunk import save_chunks
from ..config import DOCS, PROCESSED_DIR, RAW_DIR


def run(only: set[str] | None = None) -> None:
    from . import bsr, gcc, nit

    from ..chunk import load_chunks

    out_path = PROCESSED_DIR / "chunks.jsonl.gz"
    chunks = [c for c in load_chunks(out_path) if c.doc not in only] if only and out_path.exists() else []
    for doc_id, parser in (("gcc", gcc.parse), ("bsr", bsr.parse), ("nit", nit.parse)):
        if only and doc_id not in only:
            continue
        if doc_id == "nit":
            out = parser()            # reads the committed OCR output, see ingest/nit.py
        else:
            pdf = RAW_DIR / DOCS[doc_id]["file"]
            if not pdf.exists():
                raise SystemExit(f"missing {pdf}; run `python scripts/download_corpus.py` first")
            out = parser(pdf)
        print(f"{doc_id}: {len(out)} chunks")
        chunks.extend(out)
    dupes = [k for k, v in Counter(c.id for c in chunks).items() if v > 1]
    assert not dupes, f"duplicate chunk ids: {dupes[:5]}"
    save_chunks(chunks, PROCESSED_DIR / "chunks.jsonl.gz")
    print(f"wrote {len(chunks)} chunks -> {PROCESSED_DIR / 'chunks.jsonl.gz'}")
