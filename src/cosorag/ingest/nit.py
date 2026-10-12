"""Rajasthan PWD e-tender notice 68/2024-25 (6-page image-only Hindi scan) -> chunks.

Text comes from scripts/ocr_nit.py (two independent vision-model readings, committed in nit_ocr.json):
  * segmentation: letterhead, tables (rows written with ' | '), numbered conditions ("11. ...") -> one chunk each,
    locator "p. 4, condition 11"; a chunk never spans a page
  * every number in a chunk is looked up in the second reading of the same page; misses are stored as
    meta["unconfirmed_numbers"] and shown to the model as low-confidence
  * an English "twin" (gpt-oss-120b translation) is stored with each chunk so English questions find Hindi text;
    the translation must preserve every number of the Hindi original or it is retried, then flagged
  * the amounts are reconciled arithmetically (2% and 0.5% of Rs 286.47 lakh must equal the printed EMDs)
Translations are committed (nit_twins.json) so re-running ingestion needs no model.
"""
from __future__ import annotations

import json
import re

from ..chunk import Chunk
from ..config import PROCESSED_DIR
from ..text import numbers

COND_RE = re.compile(r"^\s*(\d{1,2})\.\s+(?!\|)")
TWIN_PATH = PROCESSED_DIR / "nit_twins.json"
TWIN_SYSTEM = """Translate this passage from a Rajasthan PWD tender notice (Hindi, OCR text) into clear English.
Reasoning: low
Keep EVERY number, amount, date, time, percentage, code and URL exactly as written (do not convert formats, do not add or drop digits). Keep proper nouns. Translate table rows row by row keeping ' | ' separators. Output only the translation."""


def _apply_checks(text: str, checks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Apply adjudicated number corrections (see scripts/ocr_nit.py) at the spot they were found."""
    corrections, unresolved = [], []
    for c in checks:
        adj = (c.get("adjudicated") or {}).get("printed", "")
        sure = (c.get("adjudicated") or {}).get("confidence") == "high"
        if (sure and adj == c["value"]) or c["value"] in c["alt"] or c["alt"] in c["value"]:
            continue    # primary reading confirmed, or "68" vs "68/2024-25" (a tokenisation split, not a disagreement)
        if sure and adj == c["alt"] and c["value"] in text:
            ctx = [t for t in c["context"].split() if t not in (c["value"],)][:3]
            spots = [m.start() for m in re.finditer(re.escape(c["value"]), text)]
            best = max(spots, key=lambda s: sum(t in text[max(0, s - 60): s + 60] for t in ctx))
            text = text[:best] + c["alt"] + text[best + len(c["value"]):]
            corrections.append({"from": c["value"], "to": c["alt"], "alt": c["alt"], "how": "2nd reading + adjudicator"})
        else:
            unresolved.append(c)
    return text, corrections, unresolved


def _segments(text: str, prev_label: str = "") -> list[tuple[str, list[str]]]:
    """Split one page into (label, lines) segments. Text before the first heading that is not a letterhead
    continues the previous page's last condition (condition 17 runs from p. 4 onto p. 5)."""
    segs: list[tuple[str, list[str]]] = []
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    starts_with_office = bool(re.match(r"^\(?\s*(कार्यालय अधिशाषी|वेब साई)", first.strip()))
    cont = prev_label.split(" (cont")[0] if prev_label.startswith("condition") else "continued"
    label, buf = ("letterhead" if starts_with_office else f"{cont} (cont.)"), []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if label == "copy-to list" or "प्रतिलिपि" in s:   # "copy forwarded to": a numbered distribution list, not conditions
            if label != "copy-to list":
                if buf:
                    segs.append((label, buf))
                label, buf = "copy-to list", []
            buf.append(s)
            continue
        is_table = s.count("|") >= 2
        m = COND_RE.match(s)
        if is_table and not label.startswith("table"):
            if buf:
                segs.append((label, buf))
            label, buf = "table", []
        elif not is_table and label.startswith("table"):
            segs.append((label, buf))
            label, buf = ("condition " + m.group(1)) if m else "notes", []
        elif m and not is_table:
            if buf:
                segs.append((label, buf))
            label, buf = "condition " + m.group(1), []
        buf.append(s)
    if buf:
        segs.append((label, buf))
    # split very long segments at line boundaries (~1300 chars) so a chunk stays focused
    out = []
    for lab, lines in segs:
        part, n = [], 0
        for ln in lines:
            if part and n + len(ln) > 1300:
                out.append((lab, part)); part, n = [], 0
            part.append(ln); n += len(ln)
        if part:
            out.append((lab, part))
    return out


def _translate(texts: dict[str, str]) -> dict[str, dict]:
    """Translate missing passages (needs an LLM); returns {key: {"en": ..., "number_mismatch": [...]}}."""
    twins = json.loads(TWIN_PATH.read_text(encoding="utf-8")) if TWIN_PATH.exists() else {}
    todo = {k: v for k, v in texts.items() if k not in twins or twins[k].get("hi") != v}
    if todo:
        from concurrent.futures import ThreadPoolExecutor

        from ..llm import LLM

        llm = LLM("bedrock:openai.gpt-oss-120b-1:0")
        llm._bedrock()

        def one(item):
            k, hi = item
            best = None
            for attempt in range(3):
                en = llm.chat(TWIN_SYSTEM, hi if attempt == 0 else hi + "\n\n(Keep all numbers exactly.)",
                              max_tokens=2500, temperature=0.0 if attempt == 0 else 0.3).text.strip()
                mism = sorted(numbers(hi) ^ numbers(en))
                if best is None or len(mism) < len(best[1]):
                    best = (en, mism)
                if not mism:
                    break
            return k, {"hi": hi, "en": best[0], "number_mismatch": best[1]}

        with ThreadPoolExecutor(6) as ex:
            for k, v in ex.map(one, todo.items()):
                twins[k] = v
        TWIN_PATH.write_text(json.dumps(twins, ensure_ascii=False, indent=1), encoding="utf-8")
    return twins


def reconcile(chunks: list[Chunk]) -> list[str]:
    """Arithmetic checks on the extracted amounts (cheap, catches OCR digit errors)."""
    report = []
    allnums = set().union(*(numbers(c.text) for c in chunks)) if chunks else set()
    cost_lakh = 286.47
    for pct, printed in ((2.0, "572940"), (0.5, "143235")):
        calc = round(cost_lakh * 1e5 * pct / 100)
        ok = str(calc) == printed and printed in allnums
        report.append(f"EMD {pct}% of Rs {cost_lakh} lakh = {calc}: printed {printed} -> {'OK' if ok else 'MISMATCH'}")
    return report


def parse() -> list[Chunk]:
    path = PROCESSED_DIR / "nit_ocr.json"
    if not path.exists():
        print("nit: no OCR output yet (run scripts/ocr_nit.py)")
        return []
    ocr = json.loads(path.read_text(encoding="utf-8"))
    raw = []
    for p, d in sorted(ocr["pages"].items(), key=lambda kv: int(kv[0])):
        primary, corrections, unresolved = _apply_checks(d["primary"]["text"], d.get("number_checks", []))
        counts: dict[str, int] = {}
        for label, lines in _segments(primary, prev_label=raw[-1][1] if raw else ""):
            text = "\n".join(lines)
            n = counts.get(label, 0); counts[label] = n + 1
            key = f"p{p}:{label.replace(' ', '')}:{n}"
            mine = [c for c in corrections if c["alt"] in text]
            unconfirmed = [f"{u['value']} (other reading: {u['alt']})" for u in unresolved if u["value"] in text]
            raw.append((int(p), label, key, text, unconfirmed, "[hw:" in text, mine))
    twins = _translate({r[2]: r[3] for r in raw})
    chunks = []
    for p, label, key, text, unconfirmed, hw, fixed in raw:
        tw = twins[key]
        loc = f"p. {p}, {label}" + (" (cont.)" if key.endswith(tuple(f":{i}" for i in range(1, 9))) else "")
        warn = f"\n[OCR check: low confidence, readings disagree on: {'; '.join(unconfirmed)}]" if unconfirmed else ""
        body = f"{text}\n[English translation] {tw['en']}{warn}"
        chunks.append(Chunk(
            id=f"nit:{key}", doc="nit", pdf_page=p, page_label=str(p), locator=loc, title="e-Tender Notice 68/2024-25",
            text=body, index_text=f"Rajasthan PWD e-tender notice 68/2024-25 (Yatri Niwas, Jaipur) | {loc}\n{text}\n{tw['en']}",
            method="vlm-ocr",
            meta={"unconfirmed_numbers": unconfirmed, "ocr_corrections": fixed, "handwritten": hw,
                  "translation_number_mismatch": tw["number_mismatch"]},
        ))
    for line in reconcile(chunks):
        print("  nit check:", line)
    return chunks
