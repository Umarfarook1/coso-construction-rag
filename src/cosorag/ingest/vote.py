"""Number-level majority vote across independent OCR readings of the same page.

Why: even the best reader (0.6% CER) misread "06.00" as "08.00" and "151" as "161" on this notice. Page-level
set comparison misses such errors when the wrong number also occurs elsewhere on the page, so numbers are
aligned by their neighbouring words instead:
  * 2 of 2 secondary readings agree on a different value at the same spot -> corrected (logged)
  * only one secondary reading disagrees -> kept, flagged low-confidence with the alternative
"""
from __future__ import annotations

import re
import unicodedata

NUM = re.compile(r"^\d+(?:[.:/-]\d+)*$")
MARKS = re.compile(r"[ऀ-ः़्॑-॔]")   # anusvara, visarga, nukta, virama: spelling noise


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    text = re.sub(r"[‐-―−]", "-", text)
    out = []
    for t in re.split(r"[\s|()\[\],;।]+", text):
        t = t.strip("-/–:.")
        if t:
            out.append(t)
    return out


def _ctx(tok: str) -> str:
    return MARKS.sub("", tok).lower()


def _index(tokens: list[str]) -> dict[tuple, list[str]]:
    idx: dict[tuple, list[str]] = {}
    for i, t in enumerate(tokens):
        if NUM.match(t):
            for key in (("L",) + tuple(_ctx(x) for x in tokens[max(0, i - 2):i]),
                        ("R",) + tuple(_ctx(x) for x in tokens[i + 1:i + 3])):
                if len(key) == 3:
                    idx.setdefault(key, []).append(t)
    return idx


def vote(primary: str, secondaries: list[str]) -> tuple[str, list[dict]]:
    """Return (corrected primary text, findings). Findings: {"value", "alt", "action": corrected|flagged}."""
    toks = _tokens(primary)
    sec_idx = [_index(_tokens(s)) for s in secondaries if s]
    findings, fixes = [], {}
    for i, t in enumerate(toks):
        if not NUM.match(t) or len(t.replace(".", "")) < 2:
            continue
        left = ("L",) + tuple(_ctx(x) for x in toks[max(0, i - 2):i])
        right = ("R",) + tuple(_ctx(x) for x in toks[i + 1:i + 3])
        alts = []
        for idx in sec_idx:
            cand = (idx.get(left) or []) + (idx.get(right) or [])
            if not cand:
                alts.append(None)          # this reader has nothing at this spot: abstain
            elif t in cand:
                alts.append(t)
            else:
                alts.append(cand[0])
        votes = [a for a in alts if a is not None]
        if votes and all(a == votes[0] != t for a in votes) and len(votes) >= 2:
            findings.append({"value": t, "alt": votes[0], "action": "corrected", "context": " ".join(toks[max(0, i - 3):i + 2])})
            fixes[i] = votes[0]
        elif any(a not in (None, t) for a in votes):
            alt = next(a for a in votes if a not in (None, t))
            findings.append({"value": t, "alt": alt, "action": "flagged", "context": " ".join(toks[max(0, i - 3):i + 2])})
    if not fixes:
        return primary, findings
    # apply corrections at the right occurrence: replace the n-th occurrence of the token in the original text
    out = primary
    for i, new in sorted(fixes.items(), reverse=True):
        old = toks[i]
        nth = sum(1 for j in range(i) if toks[j] == old)
        pos = -1
        for _ in range(nth + 1):
            pos = out.find(old, pos + 1)
        if pos >= 0:
            out = out[:pos] + new + out[pos + len(old):]
    return out, findings
