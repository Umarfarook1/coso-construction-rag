"""Hybrid retrieval: exact identifier path first, then reciprocal-rank fusion of BM25 and dense.

Why an identifier path: "item 2.33.1" and "2.33.6" differ by one digit and a few words, and "clause 10C"
vs "10CC" by one letter. Embeddings blur exactly these differences, so explicit ids are resolved by lookup
and never left to similarity. If a user names an id that does not exist, we can say so for certain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Index

# suffix attached ("10CC") or a known spaced suffix ("10 CC"); "Clause 2 of" must stay Clause 2
CLAUSE_Q = re.compile(r"\bclause\s*(\d+(?:[a-z]{1,2}|\s+(?:a|b|c|d|ca|cc))?)\b", re.I)
ITEM_Q = re.compile(r"\b(item\s*(?:no\.?)?\s*)?(?:([GBSELHR])\s*-\s*)?(\d{1,2}(?:\.\d{1,3}){1,3})\b(?!\s*(?:mm|cm|m\b|kw|hp|tr|%|per|lakh|sq|cum|kg|x|ton|ltr|litre|rs))", re.I)
CODE_Q = re.compile(r"\b(?:code|item)\s*(?:no\.?)?\s*(\d{3,5})\b", re.I)
CHAPTER_WORDS = {"sanitary": "S", "building": "B", "electric": "E", "electrical": "E", "lift": "L",
                 "horticulture": "H", "road": "R", "general": "G", "labour": "G", "material": "G"}


@dataclass
class Hit:
    idx: int
    score: float
    sources: dict = field(default_factory=dict)   # retriever -> rank (1-based)


@dataclass
class Retrieval:
    hits: list[Hit]
    id_lookups: list[dict]        # [{"id": "S-2.33.1", "found": True, "chunk_ids": [...]}, ...]


def find_identifiers(index: Index, query: str) -> list[dict]:
    """Resolve explicit clause / item ids in the query to chunks."""
    out = []
    q = query.lower()
    for m in CLAUSE_Q.finditer(query):
        cid = re.sub(r"\s+", "", m.group(1)).upper()
        ids = [c.id for c in index.chunks if c.doc == "gcc" and c.locator in (f"Clause {cid}", f"Schedule F, Clause {cid}")]
        out.append({"id": f"Clause {cid}", "found": bool(ids), "chunk_ids": ids})
    chapter_hint = next((v for k, v in CHAPTER_WORDS.items() if k in q), None)
    for m in ITEM_Q.finditer(query):
        if CLAUSE_Q.search(query[max(0, m.start() - 10): m.end()]):
            continue  # "clause 5.2" is a sub-clause, not a BSR item
        prefix, letter, no = m.group(1), m.group(2), m.group(3)
        if not prefix and not letter and no.count(".") < 2:
            continue  # a bare "2.2" or "12.5" is a measurement, not an item number
        chap = (letter or chapter_hint or "").upper()
        cands = [c.id for c in index.chunks if c.doc == "bsr" and c.meta.get("item_no") == no
                 and (not chap or c.meta.get("chapter") == chap)]
        label = f"{chap + '-' if chap else ''}{no}"
        out.append({"id": f"Item {label}", "found": bool(cands), "chunk_ids": cands,
                    "ambiguous": not chap and len({index.chunks[index.by_id[c]].meta.get('chapter') for c in cands}) > 1})
    for m in CODE_Q.finditer(query):
        cands = [c.id for c in index.chunks if c.doc == "bsr" and c.meta.get("chapter") == "G" and c.meta.get("item_no") == m.group(1)]
        out.append({"id": f"Item G-{m.group(1)}", "found": bool(cands), "chunk_ids": cands})
    return out


DOC_CUES = {
    "nit": re.compile(r"\b(tenders?|bids?|bidders?|bidding|notices?|nit|emd|earnest|risl|e-?gras|challan|pre-?bid|yatri|"
                      r"ubn|eproc|processing fee|additional performance)\b|निविदा|संवेदक|धरोहर|अमानत|ई-ग्रास|टेण्डर|टेंडर", re.I),
    "bsr": re.compile(r"\b(rates?|prices?|costs?|sqm|cum|per (?:metre|meter|sqm|cum|point)|bsr|rhb|schedule of rates|"
                      r"items? [a-z]-?\d)\b|दर", re.I),
    "gcc": re.compile(r"\b(clauses?|contracts?|engineer.in.charge|eic|arbitrat\w*|schedule f|extension of time|gcc|cpwd|"
                      r"compensation|security deposit|performance guarantee|escalation|deviations?|integrity pact|"
                      r"mobili[sz]ation advance)\b|अनुबंध|संविदा", re.I),
}


def doc_hints(query: str) -> set[str]:
    """Which documents the query talks about. Used to give a 32-chunk tender notice a fair chance against a
    9,600-row schedule of rates: each hinted document adds its own ranking to the fusion."""
    return {d for d, rx in DOC_CUES.items() if rx.search(query)}


def rrf(rankings: dict[str, list[tuple[int, float]]], k: int = 60) -> list[Hit]:
    fused: dict[int, Hit] = {}
    for name, ranking in rankings.items():
        for rank, (idx, _) in enumerate(ranking, start=1):
            h = fused.setdefault(idx, Hit(idx, 0.0))
            h.score += 1.0 / (k + rank)
            h.sources[name] = rank
    return sorted(fused.values(), key=lambda h: -h.score)


def retrieve(index: Index, query: str, k: int = 8, mode: str = "hybrid", docs: set[str] | None = None,
             depth: int = 50, pool_all_docs: bool = False) -> Retrieval:
    """mode: hybrid | bm25 | dense. The identifier path runs in every mode except 'dense-only' ablations."""
    rankings = {}
    if mode in ("hybrid", "bm25"):
        rankings["bm25"] = index.bm25_search(query, depth, docs)
    if mode in ("hybrid", "dense"):
        rankings["dense"] = index.dense_search(query, depth, docs)
    if mode == "hybrid" and not docs:
        for d in doc_hints(query):   # per-document lists, so a small document is not drowned by a large one
            rankings[f"bm25:{d}"] = index.bm25_search(query, 10, {d})
            rankings[f"dense:{d}"] = index.dense_search(query, 10, {d})
    hits = rrf(rankings)
    lookups = find_identifiers(index, query) if mode == "hybrid" else []
    pinned = []
    for lk in lookups:
        for cid in lk["chunk_ids"][:4]:
            i = index.by_id[cid]
            pinned.append(Hit(i, 1.0, {"id": 1}))
    seen, final = set(), []
    for h in pinned + hits:
        if h.idx in seen:
            continue
        seen.add(h.idx)
        final.append(h)
        if len(final) >= k:
            break
    if pool_all_docs and mode == "hybrid":
        # candidate pool for the reranker: every document's own top-3 joins, so the cross-encoder (not the
        # corpus size) decides between a contract clause, a rate row and a tender condition
        for d in ("gcc", "bsr", "nit"):
            if docs and d not in docs:
                continue
            for h in rrf({"b": index.bm25_search(query, 10, {d}), "d": index.dense_search(query, 10, {d})})[:3]:
                if h.idx not in seen:
                    seen.add(h.idx)
                    final.append(h)
    return Retrieval(final, lookups)
