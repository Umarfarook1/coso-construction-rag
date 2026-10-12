"""Independent reference parser for the schedule of rates (column edges from the header box), written separately
during the table-extraction bake-off. Used only to cross-check src/cosorag/ingest/bsr.py.
"""
"""BSR table extraction bake-off. usage: bsr_tables.py <bsr.pdf> <gt.json> [dump_dir]
Each method returns rows [item_no, description, unit, rate]. A shared post-step merges continuation
rows (empty item_no) into the previous row, so every method gets the same multi-line help.
A GT row counts as correct when item_no, unit and rate match exactly AND the description starts with
the GT first words and ends with the GT last word (proves the multi-line description was joined and
nothing from the next row leaked in)."""
import sys, json, re, time, pathlib
import pymupdf
try:
    import pdfplumber  # only for the bake-off comparison, not needed here
except ImportError:
    pdfplumber = None

ITEM = re.compile(r"^\d+(\.\d+)+$")
RATE = re.compile(r"^\d+\.\d{2}$")
UNITS = {"each", "rm", "rmt", "kg", "sqm", "cum", "m", "mtr", "no", "nos", "sq.m", "cu.m", "per", "%"}


def clean(c):
    return re.sub(r"\s+", " ", (c or "")).strip()


def to4(row):
    """map an arbitrary row of cells to [item, desc, unit, rate] by content pattern."""
    cells = [clean(c) for c in row]
    if len(cells) == 4:
        return cells
    item = unit = rate = ""
    rest = []
    for c in cells:
        if not item and ITEM.match(c):
            item = c
        elif not rate and RATE.match(c):
            rate = c
        elif not unit and c.lower() in UNITS:
            unit = c
        elif c:
            rest.append(c)
    return [item, " ".join(rest), unit, rate]


def merge(rows):
    out = []
    for r in rows:
        r = [clean(x) for x in r]
        if r[1] == "Description" or r[0].startswith("Item"):
            continue
        if r[0] == "§":
            out.append(["", r[1], r[2], r[3]])  # heading row, kept separate
        elif not r[0] and out:
            out[-1][1] = (out[-1][1] + " " + r[1]).strip()
            out[-1][2] = out[-1][2] or r[2]
            out[-1][3] = out[-1][3] or r[3]
        elif any(r):
            out.append(r)
    return out


def m_fitz(page, **kw):
    rows = []
    for t in page.find_tables(**kw).tables:
        rows += [to4(r) for r in t.extract()]
    return rows


def header_edges(page):
    """x positions of the vertical rules of the boxed header row (Item No | Description | Unit | Rate)."""
    xs = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l" and abs(it[1].x - it[2].x) < 1 and abs(it[1].y - it[2].y) > 15:
                xs.append(it[1].x)
            elif it[0] == "re" and it[1].width < 2 and it[1].height > 15:
                xs.append(it[1].x0)
    xs.sort()
    edges = []
    for x in xs:
        if not edges or x - edges[-1] > 3:
            edges.append(x)
    return edges


def m_header_cols(page):
    words = page.get_text("words")
    edges = header_edges(page)
    if len(edges) < 5:
        return []
    hb = [w[3] for w in words if w[4] in ("Item", "Description", "Unit", "(Rs.)", "Rate") and w[1] < page.rect.height / 3]
    if not hb:
        return []
    hdr_bottom = max(hb)
    lines = {}
    for w in words:
        if w[1] <= hdr_bottom or w[1] > page.rect.height - 45:  # skip header + running footer
            continue
        lines.setdefault(round(w[3] / 4), []).append(w)
    rows, prev_y, lh = [], None, None
    for k in sorted(lines):
        ws = sorted(lines[k], key=lambda w: w[0])
        text = " ".join(w[4] for w in ws)
        if re.fullmatch(r"RHB BSR - \d{4}|S ?- ?\d+", text):  # running footer / page label
            continue
        cols = ["", "", "", ""]
        for w in ws:
            xc = (w[0] + w[2]) / 2
            ci = max(0, min(3, sum(xc > e for e in edges[1:4])))
            cols[ci] = (cols[ci] + " " + w[4]).strip()
        y0, y1 = min(w[1] for w in ws), max(w[3] for w in ws)
        if HEADING_GAP and prev_y is not None and not cols[0] and (y0 - prev_y) > 0.9 * (lh or (y1 - y0)):
            cols[0] = "§"  # vertical gap before an un-numbered line: heading/note, not a continuation
        prev_y, lh = y1, (y1 - y0)
        rows.append(cols)
    return rows


HEADING_GAP = True


def with_parents(rows):
    """item 5.3.1.1 inherits text of 5.3.1 and 5.3 (same page here; across pages in the real pipeline)."""
    desc = {r[0]: r[1] for r in rows if r[0]}
    out = []
    for r in rows:
        if r[0] and r[0] != "§" and r[3]:
            parts = r[0].split(".")
            chain = [desc.get(".".join(parts[:i]), "") for i in range(2, len(parts))]
            out.append(r + [" > ".join([c for c in chain if c] + [r[1]])])
    return out


def m_plumber(pp, settings=None):
    rows = []
    for t in (pp.extract_tables(settings) if settings else pp.extract_tables()):
        rows += [to4(r) for r in t]
    return rows


def m_fitz_explicit(page):
    """find_tables with the header rules as explicit vertical lines + text-based rows."""
    edges = header_edges(page)
    if len(edges) < 5:
        return []
    clip = pymupdf.Rect(edges[0], 0, edges[-1], page.rect.height - 45)
    return m_fitz(page, vertical_strategy="explicit", vertical_lines=edges[:5],
                  horizontal_strategy="text", clip=clip)


def score(rows, gt):
    got = {}
    for r in rows:
        if r[0]:
            got.setdefault(r[0], r)
    ok, bad = 0, []
    for item, unit, rate, first, last in gt:
        r = got.get(item)
        d = clean(r[1]) if r else ""
        good = bool(r) and r[2] == unit and r[3] == rate and d.startswith(first) and d.endswith(last)
        ok += good
        if not good:
            bad.append(item)
    return ok, bad


if __name__ == "__main__":
    doc = pymupdf.open(sys.argv[1])
    pl = pdfplumber.open(sys.argv[1])
    gt = json.load(open(sys.argv[2]))
    dump = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
    methods = {
        "pymupdf find_tables() default (lines)": lambda p, q: m_fitz(p),
        "pymupdf find_tables(strategy='text')": lambda p, q: m_fitz(p, strategy="text"),
        "pymupdf find_tables(explicit header x-rules, text rows)": lambda p, q: m_fitz_explicit(p),
        "pdfplumber extract_tables() default (lines)": lambda p, q: m_plumber(q),
        "pdfplumber text/text strategy": lambda p, q: m_plumber(q, {"vertical_strategy": "text", "horizontal_strategy": "text"}),
        "header-rule column binning of pymupdf words": lambda p, q: m_header_cols(p),
    }
    total = sum(len(v) for v in gt.values())
    for name, fn in methods.items():
        ok_all, detail, t0 = 0, {}, time.perf_counter()
        for pno, g in gt.items():
            try:
                rows = merge(fn(doc[int(pno) - 1], pl.pages[int(pno) - 1]))
            except Exception as e:  # record and move on
                rows = []
                detail[pno] = f"ERR {e!r:.80}"
            ok, bad = score(rows, g)
            ok_all += ok
            detail.setdefault(pno, f"{ok}/{len(g)} miss={bad[:5]}")
            if dump:
                (dump / f"{re.sub(r'[^a-z]+', '_', name.lower())[:40]}_p{pno}.json").write_text(
                    json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
        dt = time.perf_counter() - t0
        print(f"{name} | {ok_all}/{total} ({ok_all / total:.0%}) | {dt / len(gt):.2f}s/page | {detail}")
    # parent inheritance demo for the best method
    for pno in gt:
        rows = with_parents(merge(m_header_cols(doc[int(pno) - 1])))
        sub = [r for r in rows if r[0].count(".") >= 2]
        print(f"p{pno}: {sum(' > ' in r[4] for r in sub)}/{len(sub)} sub-items got same-page parent text; e.g. {rows[min(7, len(rows) - 1)][0]}: {rows[min(7, len(rows) - 1)][4][:160]}")
