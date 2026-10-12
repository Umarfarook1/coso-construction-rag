"""CPWD GCC 2020 (born-digital contract) -> clause-aware chunks.

Layout facts this relies on (checked on the PDF, see DESIGN.md):
  * clause headings are bold body lines "Clause 10CC"; the clause title sits bold in the left margin (x < 140)
  * 1pt Type3 strips of "1234567890..." at the top of every page are decoration, as are the running
    header "CLAUSES OF CONTRACT" and the footer (page number + "165 Years of Engineering Excellence")
  * printed page number = PDF page - 2 (front cover and inside cover are unnumbered)
A chunk never spans a page, so every citation lands on one page with a bbox.
"""
from __future__ import annotations

import re

import pymupdf

from ..chunk import Chunk

CLAUSE_RE = re.compile(r"^clause\s*(\d+\s*[A-Z]{0,2}(?:\(\w+\))?)\s*$", re.I)
NOISE_RE = re.compile(r"^(\d{10,}|CLAUSES OF CONTRACT|165 Years of Engineering Excellence)$")
# The bilingual forms carry Hindi in a legacy Kruti Dev font: the text layer holds ASCII gibberish such as
# "fufonk ,oa lafonk" (= "निविदा एवं संविदा"). It is unreadable to any retriever, so those lines are dropped
# (the English version prevails per the book itself). A line needs 2+ hits so possessives like "contractor's" survive.
KRUTI_RE = re.compile(r"[çÙ¼½Øß]|\b(?:fufonk|lafonk|vkSj|dk;Z|ds|dh|gS|esa|rFkk|fd|dks|ls|ij|;g|vFkok|uke|fnukad|foHkkx|ifj;kstuk)\b|[a-z]'k|;k\b")
MAX_CHARS = 1400
PARA_GAP = 15.0   # vertical gap (pt) that starts a new paragraph; body line pitch is ~13pt


def _lines(page: pymupdf.Page):
    """Yield (x0, y0, x1, y1, text, bold) for real text lines, top-to-bottom."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans or spans[0]["size"] < 4:          # drops the 1pt decoration strips
                continue
            text = " ".join(s["text"].strip() for s in spans)
            x0, y0, x1, y1 = l["bbox"]
            if y0 > 722 or NOISE_RE.match(text.strip()):    # footer zone + running header
                continue
            if len(KRUTI_RE.findall(text)) >= 2 or "ç" in text:   # legacy-font Hindi gibberish
                continue
            out.append((x0, y0, x1, y1, re.sub(r"\s+", " ", text).strip(), bool(spans[0]["flags"] & 16)))
    # merge fragments on the same baseline (table cells such as "(i) | Compensation | With maximum rate")
    out.sort(key=lambda r: (round(r[1] / 3), r[0]))
    merged: list[list] = []
    for r in out:
        if merged and abs(merged[-1][1] - r[1]) < 3 and r[0] > merged[-1][0] and (r[0] >= 140) == (merged[-1][0] >= 140):
            m = merged[-1]
            merged[-1] = [m[0], min(m[1], r[1]), max(m[2], r[2]), max(m[3], r[3]), m[4] + " " + r[4], m[5] and r[5]]
        else:
            merged.append(list(r))
    return merged


def parse(pdf_path) -> list[Chunk]:
    doc = pymupdf.open(pdf_path)
    chunks: list[Chunk] = []
    section, clause, title = "Front matter", None, ""
    margin_titles: dict[tuple[int, int], str] = {}

    for pno in range(4, doc.page_count + 1):  # pp. 1-3: cover, inside cover, bilingual index (no citable content)
        page = doc[pno - 1]
        lines = _lines(page)
        in_clauses = 14 <= pno <= 57
        in_schedule = pno >= 104
        # left-margin titles (clause side headings) on clause pages
        body = []
        margin = [l for l in lines if in_clauses and l[0] < 140 and l[5]]
        for l in lines:
            if l in margin:
                continue
            body.append(l)

        buf: list = []

        def flush():
            nonlocal buf
            if not buf:
                return
            text = "\n".join(b[4] for b in buf).strip()
            if len(text) >= 25:
                locator = f"Clause {clause}" if clause and (in_clauses or in_schedule) else section
                if in_schedule and clause:
                    locator = f"Schedule F, Clause {clause}"
                n = sum(1 for c in chunks if c.locator == locator)
                cid = f"gcc:{re.sub(r'[^A-Za-z0-9]+', '-', locator).strip('-')}:{n}"
                head = f"CPWD GCC 2020 | {locator}" + (f" | {title}" if title else "")
                chunks.append(Chunk(
                    id=cid, doc="gcc", pdf_page=pno, page_label=str(pno - 2) if pno > 2 else "",
                    locator=locator, title=title, text=text, index_text=f"{head}\n{text}",
                    bboxes=[[round(min(b[0] for b in buf), 1), round(buf[0][1], 1),
                             round(max(b[2] for b in buf), 1), round(buf[-1][3], 1)]],
                ))
            buf = []

        prev_y = None
        for l in body:
            x0, y0, x1, y1, text, bold = l
            m = CLAUSE_RE.match(text)
            is_heading = bold and len(text) < 80 and not text.endswith((".", ",", ";"))
            if m and bold:
                flush()
                clause = re.sub(r"\s+", "", m.group(1)).upper()
                # the margin title starts within ~30pt below the clause heading
                t = [ml for ml in margin if y0 - 6 <= ml[1] <= y0 + 40]
                if t:
                    # join contiguous margin lines (title wraps over several lines)
                    start = t[0][1]
                    seq = [ml for ml in margin if start - 1 <= ml[1] <= start + 14 * 8]
                    joined, last = [], None
                    for ml in seq:
                        if last is not None and ml[1] - last > 16:
                            break
                        joined.append(ml[4]); last = ml[1]
                    title = " ".join(joined)
                elif in_schedule:
                    title = "Schedule F (contract-specific values)"
                else:
                    title = ""
                prev_y = y1
                continue
            fm = None if in_clauses else re.search(r"\bForm-\s?([IVXLC]+)\b", text)
            if fm:  # labour-register forms (pp. 80-95) carry no bold headings; "Form-XII" opens a section
                flush()
                clause, section = None, f"Form-{fm.group(1)}"
                title = section
                prev_y = y1
                continue
            if section.startswith("Form-") and "Register" in text and section.count(" ") == 0:
                section = f"{section} {text[text.find('Register'):][:60]}"
                title = section
            if is_heading and not in_clauses and not CLAUSE_RE.match(text):
                # a heading opens a new section, unless the buffer holds only a stub (e.g. a form title line)
                stub = sum(len(b[4]) for b in buf) < 200
                if not stub:
                    flush()
                if not in_schedule:
                    clause = None
                if len(text) > 4:
                    section = (f"{section.split(' / ')[-1]} / {text}" if stub and buf else text)[:110]
                title = section
            gap = (y0 - prev_y) if prev_y is not None else 0
            if buf and (gap > PARA_GAP and sum(len(b[4]) for b in buf) > MAX_CHARS * 0.5 or sum(len(b[4]) for b in buf) > MAX_CHARS):
                flush()
            buf.append(l)
            prev_y = y1
        flush()
    return chunks
