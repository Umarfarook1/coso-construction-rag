"""Grounded answer generation: quote-then-answer over numbered sources, then deterministic verification.

The model must (1) copy verbatim quotes from the sources, (2) answer only from those quotes with [S#] markers,
(3) say it cannot answer when the sources do not contain the answer. We then check, without any LLM:
  * every cited source exists and every quote really occurs in the source it is attributed to
  * every number in the answer occurs in a cited source (or in the question); others are flagged as unverified
A failed check triggers one retry; if it still fails the answer is shown with an explicit warning.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .chunk import Chunk
from .index import Index
from .llm import LLM, parse_json
from .retrieve import Retrieval, retrieve
from .text import normalize, numbers, tokenize

SYSTEM = """You answer questions about Indian construction documents: a construction contract (CPWD GCC 2020), a schedule of rates (RHB BSR 2024) and a Hindi tender notice (Rajasthan PWD NIT 68/2024-25, OCR'd from a scan).
Reasoning: medium

Rules:
1. Use ONLY the numbered sources. Never use outside knowledge, even if you know the answer.
2. First copy 1-4 short verbatim quotes (exact substrings, original language, at most ~40 words each) that contain the facts you need.
3. Then write the answer in English (or in Hindi if the question is in Hindi), citing every sentence with [S#]. Copy numbers, units, item codes and clause numbers exactly as in the source. Rates are in Indian Rupees.
4. Schedule of rates items are hierarchical: a child row (e.g. 2.33.1) inherits its parent's description; the source text already joins them with ">". "-do-" means "same as the previous item". Chapter G holds BASIC rates of labour and materials (cost of the material or labour alone); other chapters hold complete item rates (providing and fixing, incl. labour). Say which kind a rate is. If the question asks for a size, material or variant that no source lists exactly, say that it is not listed and give the closest listed item clearly labelled as different, instead of presenting it as the answer.
5. If a question names an item or clause that the lookup note says does not exist, say so plainly. If an item number exists in several chapters, list each chapter's item instead of picking one.
6. If the sources do not contain the answer, set "answerable": false and say what the sources do cover. A partial answer is fine if you say what is missing. Do not guess.
7. When quoting Hindi, keep the Hindi and give the English meaning. Text marked "[OCR check: low confidence ...]" may be misread: mention the uncertainty if you use that number.
8. The three documents are independent. Questions about "this tender", the notice, bids or bidders are governed by the tender notice; do not answer them from the CPWD contract (a different, central-government contract). If the notice is silent, say so, and only then mention what the CPWD contract says, labelled as such.
9. Amounts: 1 crore = 100 lakh. Convert to the same unit before comparing with slabs or limits. Show any arithmetic you do.

Return ONLY a JSON object:
{"answerable": true|false, "quotes": [{"source": "S1", "quote": "..."}], "answer": "... [S1]", "missing": "what the sources do not say (empty if nothing)"}"""


@dataclass
class Answer:
    question: str
    answer: str
    answerable: bool
    citations: list[dict]               # [{"n": 1, "chunk_id": ..., "cite": ..., "quote": ...}]
    sources: list[Chunk]
    checks: dict = field(default_factory=dict)
    retrieval: Retrieval | None = None
    timings: dict = field(default_factory=dict)
    model: str = ""
    raw: str = ""
    gate: str = ""


def format_sources(chunks: list[Chunk], max_chars: int = 1800) -> str:
    parts = []
    for n, c in enumerate(chunks, 1):
        method = " | text from OCR of a scanned page" if c.method != "native" else ""
        parts.append(f"[S{n}] {c.cite()}{method}\n{c.text[:max_chars]}")
    return "\n\n".join(parts)


def expand_neighbours(index: Index, chunks: list[Chunk], top: int = 3, cap: int = 11) -> list[Chunk]:
    """A clause can run over several page-bounded chunks (Clause 2 spans pp. 16-17): pull the next and previous
    chunk of the same clause for the top hits so a proviso on the next page is not lost."""
    out = list(chunks)
    have = {c.id for c in out}
    for c in chunks[:top]:
        if c.doc != "gcc" or ":" not in c.id:
            continue
        stem, n = c.id.rsplit(":", 1)
        for nb in (f"{stem}:{int(n) + 1}", f"{stem}:{int(n) - 1}"):
            if nb in index.by_id and nb not in have and len(out) < cap:
                out.insert(out.index(c) + 1, index.chunks[index.by_id[nb]])
                have.add(nb)
    return out


def _lookup_note(r: Retrieval) -> str:
    notes = []
    for lk in r.id_lookups:
        if not lk["found"]:
            notes.append(f"Lookup note: {lk['id']} does NOT exist in the indexed documents.")
        elif lk.get("ambiguous"):
            notes.append(f"Lookup note: {lk['id']} exists in several chapters; the matching items are included as sources.")
    return "\n".join(notes)


def _quote_ok(quote: str, src: str) -> bool:
    q, s = normalize(quote), normalize(src)
    q = re.sub(r"\s+", " ", q).strip(" .\"'")
    s = re.sub(r"\s+", " ", s)
    if q and q in s:
        return True
    qt, st = tokenize(quote), set(tokenize(src))
    return bool(qt) and sum(t in st for t in qt) / len(qt) >= 0.85


# [S1], (S3), [S1, S2], [S1][S2]: the model does not always use the exact marker format
CITE_RE = re.compile(r"[\[(]\s*(S\d+(?:\s*[,;&]\s*S?\d+)*)\s*[\])]")


def _derivable(n: str, pool: set[str], limit: int = 60) -> bool:
    """True if n is one arithmetic step (+ - x / or a percentage) from two numbers in the sources/question.
    "Rs 320 more per metre" (2509 - 2189) is a computed, checkable claim, not a hallucination."""
    try:
        target = float(n)
    except ValueError:
        return False
    vals = sorted({float(x) for x in pool if re.fullmatch(r"\d+(\.\d+)?", x)})[:limit]
    tol = max(0.011, abs(target) * 1e-4)
    for a in vals:
        for b in vals:
            for r in (a + b, a - b, a * b, a * b / 100, a * (1 + b / 100), a * (1 - b / 100), (a / b if b else None)):
                if r is not None and abs(r - target) <= tol:
                    return True
    return False


def verify(question: str, data: dict, sources: list[Chunk]) -> dict:
    answer = data.get("answer", "") or ""
    cited = sorted({int(n) for grp in CITE_RE.findall(answer) for n in re.findall(r"\d+", grp)})
    bad_cites = [n for n in cited if not 1 <= n <= len(sources)]
    quotes = data.get("quotes", []) or []
    bad_quotes = []
    for q in quotes:
        m = re.match(r"S?(\d+)", str(q.get("source", "")))
        n = int(m.group(1)) if m else 0
        if not (1 <= n <= len(sources)) or not _quote_ok(str(q.get("quote", "")), sources[n - 1].text):
            bad_quotes.append(q)
    cited_text = " ".join(sources[n - 1].text for n in cited if 1 <= n <= len(sources))
    allowed = numbers(cited_text) | numbers(question)
    ans_wo_cites = CITE_RE.sub(" ", answer)
    loose = sorted(n for n in numbers(ans_wo_cites) if n not in allowed and len(n.replace(".", "")) >= 2)
    derived = [n for n in loose if _derivable(n, allowed)]
    unsupported = [n for n in loose if n not in derived]
    answerable = bool(data.get("answerable", True))
    return {
        "cited": cited,
        "derived_numbers": derived,
        "bad_citations": bad_cites,
        "bad_quotes": bad_quotes,
        "unsupported_numbers": unsupported,
        "uncited_answer": answerable and not cited,
        "ok": not bad_cites and not bad_quotes and not unsupported and not (answerable and not cited),
    }


@dataclass
class Prepared:
    """Search and rerank results for one question, before any model writes: lets the web view run the input
    guardrail and the search at the same time."""
    question: str
    retrieval: Retrieval
    sources: list[Chunk]
    timings: dict
    gate: str = ""


def prepare(index: Index, question: str, k: int = 8, mode: str = "hybrid", docs: set[str] | None = None,
            rerank: bool = True, reranker: str | None = None) -> Prepared:
    from . import rerank as rr

    t0 = time.time()
    xenc = (reranker or rr.default_reranker()) if rerank else None
    r = retrieve(index, question, k=20 if xenc else k, mode=mode, docs=docs, pool_all_docs=bool(xenc))
    t1 = time.time()
    timings = {"retrieve_s": round(t1 - t0, 3)}
    gate = ""
    if xenc:
        pinned = [h for h in r.hits if "id" in h.sources]
        rest = [h for h in r.hits if "id" not in h.sources]
        probs = rr.score(xenc, question, [index.chunks[h.idx].index_text for h in rest])
        for h, pr in zip(rest, probs):
            h.sources["xenc"] = round(pr, 3)
        r.hits = (pinned + rr.fused_order(rest, probs))[:k]
        thr = rr.gate_threshold(xenc)
        top = max(probs, default=0.0)
        # The gate is a review flag, not an instruction to the generator: a threshold chosen on short synthetic
        # queries put 45% of real (longer, multi-part) answerable questions below it, and telling the model
        # "no source answers" turned them into refusals. Low scores are shown and logged for review instead.
        if thr is not None:
            gate = f"top passage score {top:.2f} (threshold {thr:.2f})"
            if top < thr and not pinned:
                gate += ": low confidence, check the cited passage"
        timings["rerank_s"] = round(time.time() - t1, 3)
    return Prepared(question, r, expand_neighbours(index, [index.chunks[h.idx] for h in r.hits]), timings, gate)


def answer(index: Index, question: str, llm: LLM | None = None, k: int = 8, mode: str = "hybrid",
           docs: set[str] | None = None, retries: int = 1, rerank: bool = True, reranker: str | None = None,
           prepared: Prepared | None = None) -> Answer:
    p = prepared
    if p is None or p.question != question:
        p = prepare(index, question, k=k, mode=mode, docs=docs, rerank=rerank, reranker=reranker)
    r, sources, timings, gate = p.retrieval, p.sources, dict(p.timings), p.gate
    t1 = time.time()
    if llm is None:  # retrieval-only mode: no model, show the best passages
        return Answer(question, "(retrieval-only mode: no LLM configured; top passages below)", True,
                      [{"n": i + 1, "chunk_id": c.id, "cite": c.label(), "quote": c.text[:300], "doc": c.doc,
                        "pdf_page": c.pdf_page, "locator": c.locator, "bboxes": c.bboxes} for i, c in enumerate(sources[:3])],
                      sources, {}, r, timings, gate=gate)

    user = f"{_lookup_note(r)}\n\n\nSources:\n{format_sources(sources)}\n\nQuestion: {question}".strip()
    data, checks, raw, reply, parsed = {}, {}, "", None, True
    for attempt in range(retries + 1):
        if attempt == 0:
            prompt = user
        elif not parsed:
            prompt = user + "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object."
        else:
            fails = ("bad_citations", "bad_quotes", "unsupported_numbers", "uncited_answer")
            prompt = (user + "\n\nYour previous reply failed verification: "
                      + "; ".join(f"{k}={checks[k]}" for k in fails if checks.get(k))
                      + ". Quote exactly, cite every sentence, and use only numbers present in the cited sources.")
        reply = llm.chat(SYSTEM, prompt, max_tokens=3000 if attempt == 0 else 6000)
        raw = reply.text
        try:
            data, parsed = parse_json(raw), True
        except Exception:  # noqa: BLE001  (unparseable or truncated reply: retry, then show it with a warning)
            data, parsed = {"answerable": False, "answer": "The model's reply could not be parsed; no verified answer.",
                            "quotes": []}, False
        checks = verify(question, data, sources)
        checks["parse_error"] = not parsed
        if parsed and (checks["ok"] or not data.get("answerable", True)):
            break
    timings["generate_s"] = round(time.time() - t1, 3)
    if reply is not None:
        timings.update(input_tokens=reply.input_tokens, output_tokens=reply.output_tokens, cached=reply.cached)
    cits = []
    for n in checks.get("cited", []):
        if 1 <= n <= len(sources):
            c = sources[n - 1]
            q = next((x.get("quote", "") for x in data.get("quotes", []) if re.match(rf"S?{n}$", str(x.get("source", "")))), "")
            cits.append({"n": n, "chunk_id": c.id, "cite": c.label(), "quote": q, "doc": c.doc, "pdf_page": c.pdf_page,
                         "locator": c.locator, "bboxes": c.bboxes})
    text = data.get("answer", "") or ""
    if not data.get("answerable", True) and data.get("missing") and data["missing"] not in text:
        text = f"{text} {data['missing']}".strip()
    return Answer(question, text, bool(data.get("answerable", True)), cits, sources, checks, r, timings, llm.name, raw, gate)
