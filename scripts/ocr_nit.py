"""Transcribe the image-only Hindi tender notice (one-off, offline ingestion step).

Bake-off on 6 hand-transcribed regions (see DESIGN.md): Claude Sonnet 4.6 vision CER 0.6% and 86/86 numbers,
Gemma 3 27B 4.3% / 86/86, Tesseract hin+eng 12.8% / 51/86, RapidOCR PP-OCRv5 22.8% / 78/86.
So the notice is read by two independent vision models: Claude Sonnet 4.6 (primary text) and open-weight
Gemma 3 27B (second reading). ingest/nit.py flags every number the two readings disagree on.
Output: data/processed/nit_ocr.json (committed, so nobody needs to re-run this).

  uv run python scripts/ocr_nit.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosorag.config import DOCS, PROCESSED_DIR, RAW_DIR  # noqa: E402
from cosorag.llm import LLM  # noqa: E402

PROMPT = ("Transcribe ALL text on this scanned page exactly as printed or handwritten, line by line, in the original "
          "language and script (Hindi in Devanagari, English as is). Keep every number, date, code and punctuation mark. "
          "Write tables one row per line with cells separated by ' | '. Keep the numbering of conditions (1., 2., (अ), ...) "
          "at the start of their lines. Mark handwritten text as [hw: ...]. Do not translate, correct, summarize or add "
          "anything. Output only the transcription.")
READERS = {"primary": ("bedrock:eu.anthropic.claude-sonnet-4-6", "eu-north-1"),
           "second": ("bedrock:google.gemma-3-27b-it", "ap-south-1")}


def main() -> None:
    import pymupdf

    pdf = RAW_DIR / DOCS["nit"]["file"]
    doc = pymupdf.open(pdf)
    pages = [doc[i].get_pixmap(dpi=300).tobytes("png") for i in range(doc.page_count)]

    def read(role: str, i: int) -> tuple[str, int, str, float, str]:
        spec, region = READERS[role]
        t = time.time()
        r = LLM(spec, region=region).chat("You are a precise OCR engine.", PROMPT, images=[pages[i]], max_tokens=6000)
        return role, i + 1, r.text, round(time.time() - t, 1), r.model

    jobs = [(role, i) for role in READERS for i in range(len(pages))]
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda j: read(*j), jobs))
    out = {"source": DOCS["nit"]["file"], "readers": {k: v[0] for k, v in READERS.items()}, "pages": {}}
    for role, page, text, secs, model in results:
        out["pages"].setdefault(str(page), {})[role] = {"text": text, "secs": secs, "model": model}
    # A third independent reading (Llama 4 Maverick, open-weight). Gemma sometimes stops after the letterhead on
    # dense pages; such a short second reading is replaced by this one. A number counts as confirmed if either
    # secondary reading contains it.
    READERS["third"] = ("bedrock:us.meta.llama4-maverick-17b-instruct-v1:0", "us-east-1")
    with ThreadPoolExecutor(6) as ex:
        for role, page, text, secs, model in ex.map(lambda i: read("third", i), range(len(pages))):
            d = out["pages"][str(page)]
            d["third"] = {"text": text, "secs": secs, "model": model}
            if len(d["second"]["text"]) < 0.4 * len(d["primary"]["text"]):
                d["second"] = {**d["third"], "fallback_from": READERS["second"][0]}
    for p, d in sorted(out["pages"].items()):
        print(f"p{p} primary {len(d['primary']['text'])} | second {len(d['second']['text'])} ({d['second']['model']}) | third {len(d['third']['text'])}")

    # Number disagreements between readers are adjudicated by a stronger model on a targeted question
    # ("near <context>, is it 06.00 or 08.00?"), so a 1-vs-1 split is resolved by looking, not by guessing.
    from cosorag.ingest.vote import vote
    from cosorag.llm import parse_json

    judge = LLM("bedrock:eu.anthropic.claude-opus-5-5", region="eu-north-1")
    for p, d in sorted(out["pages"].items()):
        secs = [d["second"]["text"]] + ([d["third"]["text"]] if d["third"]["text"] != d["second"]["text"] else [])
        _, findings = vote(d["primary"]["text"], secs)
        for f in findings:
            q = (f"On this scanned page, find the text near: \"{f['context']}\". One reader read a value there as "
                 f"\"{f['value']}\", another as \"{f['alt']}\". Look carefully at the image: what exactly is printed or "
                 f"handwritten at that spot? Reply with JSON only: {{\"printed\": \"...\", \"confidence\": \"high|low\"}}")
            r = judge.chat("You verify OCR readings against the image.", q, images=[pages[int(p) - 1]], max_tokens=4000)
            try:
                f["adjudicated"] = parse_json(r.text)
            except Exception:  # noqa: BLE001
                f["adjudicated"] = {"printed": "", "confidence": "low", "raw": r.text[:200]}
            print(f"  p{p} {f['value']} vs {f['alt']} -> {f['adjudicated']}")
        d["number_checks"] = findings
    path = PROCESSED_DIR / "nit_ocr.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
