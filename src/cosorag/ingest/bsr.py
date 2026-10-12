"""RHB BSR 2024 (824-page schedule of rates) -> one record per rated item, plus group and notes chunks.

The table is a hierarchy printed flat:
    1.8     P & F water closet Seat Covers with brass hinges complete :      (parent, no rate)
    1.8.1   Solid PVC (ISI marked) grade-I Black for EWC.     Each   675.00
    1.8.2   -do- White.                                        Each   731.00  ("-do-" = ditto previous sibling)
so a row is only meaningful with its parent chain resolved. Item numbers restart in every chapter
(G, B, S, E, L, H, R), so ids are chapter-qualified: "S-1.8.2". Rows are segmented by item-number
anchors in the left column rather than by text lines, because unit/rate cells sit 1-2 pt off the
description baseline.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from ..chunk import Chunk

CHAPTERS = {  # chapter letter -> (name, first pdf page, last pdf page)
    "G": ("General Basic Rates (labour and materials)", 11, 56),
    "B": ("Building Work", 59, 237),
    "S": ("Sanitary Work", 240, 394),
    "E": ("Electric Work", 397, 630),
    "L": ("Lift Work", 632, 636),
    "H": ("Horticulture Work", 639, 652),
    "R": ("Road Work", 655, 687),
    "M": ("Miscellaneous Circulars", 689, 822),
}
ITEM_RE = re.compile(r"^\d{1,5}(\.{1,2}\d{1,3}){0,4}(\([a-zA-Z]\)|[a-z])?\.?$")  # 2.31..2 typos, 1.22(a) variants
NOTE_RE = re.compile(r"^\(([A-Za-z]|[ivx]{1,4})\)$")
RATE_RE = re.compile(r"^(Rs\.?)?[\d,]+(\.\d+)?%?$")
FOOTER_RE = re.compile(r"\bRHB\s*BSR\s*-\s*2024\b")
DITTO_RE = re.compile(r"^\(?\s*-+\s*do\s*-+\s*\)?", re.I)
HEADER_WORDS = {"Description", "DESCRIPTION", "PARTICULARS", "Particulars"}


@dataclass
class Item:
    chapter: str
    no: str
    own: str                  # description text printed on this row
    unit: str
    rate: str
    pdf_page: int
    page_label: str
    heading: str
    bbox: list[float]
    is_note: bool = False
    subchapter: str = ""
    resolved: str = ""        # parent chain + ditto resolved
    children: list[str] = field(default_factory=list)


def chapter_of(pno: int) -> str | None:
    for ch, (_, a, b) in CHAPTERS.items():
        if a <= pno <= b:
            return ch
    return None


def _page_label(words, ch: str) -> str:
    top = sorted((w for w in words if w[1] < 60), key=lambda w: w[0])
    lab = re.sub(r"\s", "", "".join(w[4] for w in top))
    m = re.match(rf"^{ch}-?(\d+)$", lab)
    return f"{ch}-{m.group(1)}" if m else ""


def _group_lines(words, tol=3.0):
    lines: list[list] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if lines and abs(lines[-1][0][1] - w[1]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w[0]) for l in lines]


def _is_heading(line_words) -> bool:
    text = " ".join(w[4] for w in line_words)
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= 5 and letters.isupper() and "RHBBSR" not in letters and not any(RATE_RE.match(w[4]) and w[0] > 440 for w in line_words)


def parse_items(pdf_path) -> tuple[list[Item], list[Chunk]]:
    """Return rated/parent items for table chapters and plain text chunks for everything else."""
    doc = pymupdf.open(pdf_path)
    items: list[Item] = []
    text_chunks: list[Chunk] = []
    heading = {ch: "" for ch in CHAPTERS}
    subchapter: dict[str, str] = {}

    for pno in range(1, doc.page_count + 1):
        page = doc[pno - 1]
        ch = chapter_of(pno)
        words = [w for w in page.get_text("words") if w[4].strip()]
        if not words:
            continue
        label = _page_label(words, ch) if ch else ""
        portrait = page.rect.width < page.rect.height
        glines = _group_lines(words)
        for k, l in enumerate(glines):
            t = " ".join(w[4] for w in l)
            m = re.search(r"CHAPTER\s*:?\s*([A-Z]-\d+)\s*(.*)", t)
            if m:
                name = m.group(2).strip() or (" ".join(w[4] for w in glines[k + 1]) if k + 1 < len(glines) else "")
                subchapter[ch or "?"] = f"{m.group(1)} {name.title()}".strip()
                heading[ch or "?"] = ""
        units = [w for w in words if w[4].lower() == "unit"]
        hdr = [w for w in words if w[4] in HEADER_WORDS and any(abs(u[1] - w[1]) < 12 for u in units)]
        if not ch or ch == "M" or not portrait or not hdr:
            text_chunks.extend(_text_chunks(page, pno, ch, label))
            continue

        y_top = max(w[3] for w in words if abs(w[1] - hdr[0][1]) <= 12) + 2
        above = [w for w in words if 60 < w[3] < hdr[0][1] - 10]
        if len(above) > 8:   # chapter notes printed above the table header
            text_chunks.append(_notes_chunk(above, pno, ch, label, subchapter.get(ch, "")))
        unit_hdr = [w for w in units if abs(w[1] - hdr[0][1]) < 12]
        rate_hdr = [w for w in words if w[4].lower().startswith("rate") and abs(w[1] - hdr[0][1]) < 14]
        unit_x = (unit_hdr[0][0] - 14) if unit_hdr else 400
        rate_x = (rate_hdr[0][0] - 14) if rate_hdr else 455
        desc_x = hdr[0][0] - 160  # description column starts well left of its centred header
        body = [w for w in words if y_top < w[1] < page.rect.height - 30]
        lines = _group_lines(body)
        left_edge = min((l[0][0] for l in lines), default=80)
        # the item-number column is right-aligned on some pages, so a short "5.3" sits further right than "5.2.10":
        # accept a dotted number that ends before the description column (the most common line start)
        starts = Counter(round(l[0][0]) for l in lines if not (ITEM_RE.match(l[0][4]) or NOTE_RE.match(l[0][4])))
        desc_x = starts.most_common(1)[0][0] if starts else 120

        anchors = []
        for l in lines:
            w0 = l[0]
            if (ITEM_RE.match(w0[4]) or NOTE_RE.match(w0[4])) and w0[0] < unit_x - 120 and (
                    w0[0] <= left_edge + 20 or ("." in w0[4] and w0[2] <= desc_x - 1)):
                anchors.append(w0)
        if not anchors:
            # continuation page with no item numbers (long notes): keep as text
            text_chunks.extend(_text_chunks(page, pno, ch, label))
            continue

        bounds = [a[1] - 2.5 for a in anchors] + [page.rect.height]
        # text above the first anchor continues the previous page's last item (or is a heading/note)
        lead = [l for l in lines if l[0][1] < bounds[0]]
        if lead:
            lead_heads = [l for l in lead if _is_heading(l)]
            rest = [w for l in lead if not _is_heading(l) for w in l]
            if lead_heads:
                heading[ch] = " ".join(w[4] for w in lead_heads[-1])
            if rest and items and items[-1].chapter == ch:
                prev = items[-1]
                add_desc = " ".join(w[4] for w in rest if w[0] < unit_x)
                add_unit = " ".join(w[4] for w in rest if unit_x <= w[0] < rate_x)
                add_rate = " ".join(w[4] for w in rest if w[0] >= rate_x and RATE_RE.match(w[4].replace("Rs.", "")))
                prev.own = (prev.own + " " + add_desc).strip()
                prev.unit = prev.unit or add_unit
                prev.rate = prev.rate or add_rate

        for i, a in enumerate(anchors):
            blk = [w for w in body if bounds[i] <= w[1] < bounds[i + 1] and w is not a]
            blines = _group_lines(blk)
            desc, unit, rate, seen_rate, next_heading = [], [], [], False, ""
            for l in blines:
                if seen_rate and _is_heading(l):
                    next_heading = " ".join(w[4] for w in l)
                    continue
                for w in l:
                    if w[0] >= rate_x and RATE_RE.match(w[4].replace("Rs.", "")):
                        rate.append(w[4]); seen_rate = True
                    elif unit_x <= w[0] < rate_x and not desc_x > w[0]:
                        unit.append(w[4])
                    else:
                        desc.append(w[4])
            no = re.sub(r"\.\.+", ".", a[4].rstrip("."))
            is_note = bool(NOTE_RE.match(a[4]))
            ys = [w[3] for w in blk] or [a[3]]
            items.append(Item(
                subchapter=subchapter.get(ch, ""), chapter=ch, no=no, own=re.sub(r"\s*\bnote\s*:?\s*-?\s*$", "", FOOTER_RE.sub("", re.sub(r"\s+", " ", " ".join(desc))).strip(), flags=re.I),
                unit=" ".join(unit).strip(), rate=" ".join(rate).strip(), pdf_page=pno, page_label=label,
                heading=heading[ch], bbox=[round(a[0], 1), round(a[1], 1), round(page.rect.width - 40, 1), round(max(ys), 1)],
                is_note=is_note,
            ))
            if next_heading:
                heading[ch] = next_heading
    _resolve(items)
    return items, text_chunks


def _resolve(items: list[Item]) -> None:
    """Attach parent descriptions and expand '-do-' (ditto) from the previous sibling."""
    by_key: dict[tuple[str, str], Item] = {}
    last_note_parent: dict[str, str] = {}
    prev_sibling: dict[tuple[str, str], Item] = {}
    for it in items:
        if it.is_note:
            it.no = f"{last_note_parent.get(it.chapter, '0')} Note {it.no}"
            it.resolved = it.own
            continue
        base_no = re.sub(r"\(?[a-zA-Z]\)?$", "", it.no)  # 1.22(a) is a variant of 1.22
        parts = it.no.split(".") if base_no == it.no else base_no.split(".") + [""]
        parent_no = ".".join(parts[:-1]) if base_no == it.no else base_no
        parent = by_key.get((it.chapter, parent_no))
        own = it.own
        it.ditto_base = own  # type: ignore[attr-defined]
        if DITTO_RE.match(own):
            sib = prev_sibling.get((it.chapter, parent_no))
            base = getattr(sib, "ditto_base", sib.own) if sib is not None else ""
            it.ditto_base = base  # type: ignore[attr-defined]
            own = f"{base} [-do-, i.e. same as above but: {DITTO_RE.sub('', own).strip()}]"
        it.own_resolved = own  # type: ignore[attr-defined]
        chain = []
        p = parent
        while p is not None:
            chain.append(p.own)
            p = by_key.get((p.chapter, ".".join(p.no.split(".")[:-1]))) if "." in p.no else None
        it.resolved = " > ".join(list(reversed(chain)) + [own])
        it.chain = " > ".join(reversed(chain))  # type: ignore[attr-defined]
        if parent is not None:
            parent.children.append(it.no)
        by_key[(it.chapter, it.no)] = it
        prev_sibling[(it.chapter, parent_no)] = it
        last_note_parent[it.chapter] = it.no


def _notes_chunk(words, pno: int, ch: str, label: str, sub: str) -> Chunk:
    text = "\n".join(" ".join(w[4] for w in l) for l in _group_lines(words))
    name = CHAPTERS[ch][0]
    loc = f"Chapter {sub or ch} notes"
    return Chunk(
        id=f"bsr:notes:{pno}", doc="bsr", pdf_page=pno, page_label=label or str(pno), locator=loc,
        title=f"{name} / {sub}", text=text, index_text=f"RHB BSR 2024 | {name} | {sub} | chapter notes\n{text}",
        bboxes=[[round(min(w[0] for w in words), 1), round(min(w[1] for w in words), 1),
                 round(max(w[2] for w in words), 1), round(max(w[3] for w in words), 1)]],
        meta={"chapter": ch, "kind": "notes"},
    )


def _text_chunks(page, pno: int, ch: str | None, label: str) -> list[Chunk]:
    text = page.get_text().strip()
    if len(text) < 40:
        return []
    paras = re.split(r"\n\s*\n", text)
    out, buf = [], ""
    for p in paras + [""]:
        if p and len(buf) + len(p) < 1400:
            buf += ("\n" if buf else "") + p
            continue
        if buf.strip():
            name = CHAPTERS[ch][0] if ch else "Front matter"
            loc = f"{name}, page {label or pno}"
            out.append(Chunk(
                id=f"bsr:page{pno}:{len(out)}", doc="bsr", pdf_page=pno, page_label=label or str(pno),
                locator=loc, title=name, text=re.sub(r"[ \t]+", " ", buf.strip()),
                index_text=f"RHB BSR 2024 | {name} | page {label or pno}\n{buf.strip()}",
            ))
        buf = p
    return out


def to_chunks(items: list[Item], text_chunks: list[Chunk]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    by_key = {(it.chapter, it.no): it for it in items}
    for it in items:
        if not it.rate and not it.children and not it.is_note:
            continue  # stray anchor (e.g. a numbered sentence) with nothing to cite
        base = f"bsr:{it.chapter}:{it.no}"
        n = seen.get(base, 0); seen[base] = n + 1
        cid = base if n == 0 else f"{base}#{n}"
        chap = CHAPTERS[it.chapter][0] + (f" / {it.subchapter}" if it.subchapter else "")
        loc = f"Item {it.chapter}-{it.no}"
        # The row's own text, unit and rate come FIRST and the parent chain after: encoders truncate at 256 tokens
        # and 22% of rows sit under a parent long enough to push "15 mm ... Rs 787" out of the window.
        own = getattr(it, "own_resolved", it.own)
        chain = getattr(it, "chain", "")
        part_of = f"\nPart of: {chain}" if chain else ""
        if it.rate:
            rate = it.rate if it.rate.endswith("%") else "Rs. " + it.rate
            body = f"{own}\nUnit: {it.unit or '-'} | Rate: {rate}{part_of}"
        else:
            kids = [by_key.get((it.chapter, k)) for k in it.children]
            lines = [f"  {k.chapter}-{k.no}: {getattr(k, 'own_resolved', k.own)} | {k.unit} | {k.rate}" for k in kids if k]
            body = f"{own}{part_of}\nSub-items:\n" + "\n".join(lines)
        head = f"RHB BSR 2024 | {loc}"
        ctx = f"{chap}" + (f" | {it.heading.title()}" if it.heading else "")
        chunks.append(Chunk(
            id=cid, doc="bsr", pdf_page=it.pdf_page, page_label=it.page_label or str(it.pdf_page),
            locator=loc, title=f"{chap}" + (f" / {it.heading.title()}" if it.heading else ""),
            text=f"{loc}: {body}",
            index_text=f"{head}: {body}\n{ctx}",
            bboxes=[it.bbox],
            meta={"chapter": it.chapter, "item_no": it.no, "unit": it.unit, "rate": it.rate, "kind": "note" if it.is_note else ("item" if it.rate else "group")},
        ))
    return chunks + text_chunks


def parse(pdf_path) -> list[Chunk]:
    items, text_chunks = parse_items(pdf_path)
    return to_chunks(items, text_chunks)
