"""Normalization and tokenization shared by indexing, querying and answer verification."""
from __future__ import annotations

import re
import unicodedata

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# nukta forms -> base consonant, chandrabindu -> anusvara, drop ZWJ/ZWNJ
DEV_FOLD = str.maketrans({"़": None, "ँ": "ं", "‌": None, "‍": None})

TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)+"            # item codes and decimals kept whole: 2.33.1, 286.47
    r"|[a-z]+(?:\d+[a-z]*)+"    # alphanumeric ids kept whole: "pwd2425wsob17397", "is456"
    r"|\d+[a-z]+"               # 10cc, 2a
    r"|[ऀ-ॿ]+"        # a Devanagari word incl. matras and virama
    r"|[a-z]+|\d+"
)
STOP = set(
    "a an the of to in on for and or is are was be by with as at from that this which what how much "
    "does do shall will per any all its it under any".split()
) | set("का के की है में और से को पर एवं तथा या हेतु द्वारा किया गया जो यह".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(DEV_DIGITS).translate(DEV_FOLD)
    text = re.sub(r"[‐-―−]", "-", text)   # non-breaking hyphens / dashes that LLMs emit
    # Indian and western digit grouping -> plain digits: 5,72,940 -> 572940 ; 1,25,000 -> 125000
    text = re.sub(r"(?<=\d),(?=\d{2,3}\b)", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("₹", " rs ")
    return text.lower()


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(normalize(text))
    # glue "10 cc" -> "10cc" style clause ids written with a space
    out: list[str] = []
    for t in toks:
        if out and out[-1].isdigit() and t in {"a", "b", "c", "d", "ca", "cc"}:
            out[-1] = out[-1] + t
            continue
        if t not in STOP and len(t) > 0:
            out.append(t)
    return out


NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    """Numbers in a text, canonicalised so '1188.00' == '1188' and '5,72,940' == '572940'."""
    out = set()
    for n in NUM_RE.findall(normalize(text)):
        ip, _, fp = n.partition(".")
        fp = fp.rstrip("0")
        out.add((ip.lstrip("0") or "0") + ("." + fp if fp else ""))
    return out
