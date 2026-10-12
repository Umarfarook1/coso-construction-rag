"""Is a PDF page's text layer usable, or should the page be re-OCR'd?

"Has a text layer" is not "has usable text". In this corpus:
  * the PMGSY-III summary's Hindi layer drops conjuncts (wkhtmltopdf): "प्रधानमंत्री" -> "धानमंी", so many
    Devanagari words start with a dependent vowel sign, which no valid word does
  * the IS-code scans carry an OCR layer that lost its spaces: "ThisIndianStandard", "tofillallthevoidswhich"
  * the CPWD contract's bilingual pages hold Hindi in a legacy Kruti Dev font: ASCII gibberish ("fufonk ,oa")
Three cheap signals catch all three without a dictionary or a model.
"""
from __future__ import annotations

import re

DEV_WORD = re.compile(r"[ऀ-ॿ]+")
VOWEL_SIGN = re.compile(r"^[ा-ौॢॣ]")          # matra at word start = broken shaping
ODD_CHAR = re.compile(r"[-￰-￿\x00-\x08\x0b\x0c\x0e-\x1f]")
GLUED = re.compile(r"^[A-Za-z]{18,}$")                             # very long letter-only tokens = lost spaces
KRUTI = re.compile(r"\b(?:fufonk|lafonk|vkSj|dk;Z|ds|dh|gS|esa|rFkk)\b|[a-z]'k")


def quality(text: str) -> dict:
    words = text.split()
    dev = DEV_WORD.findall(text)
    latin = [w for w in words if re.fullmatch(r"[A-Za-z]+", w)]
    q = {
        "chars": len(text.strip()),
        "dev_matra_start": round(sum(bool(VOWEL_SIGN.match(w)) for w in dev) / len(dev), 3) if dev else 0.0,
        "odd_chars_per_1k": round(1000 * len(ODD_CHAR.findall(text)) / max(1, len(text)), 1),
        "glued_words": round(sum(bool(GLUED.match(w)) for w in latin) / len(latin), 3) if latin else 0.0,
        "kruti_hits_per_1k_words": round(1000 * len(KRUTI.findall(text)) / max(1, len(words)), 1),
    }
    reasons = []
    if q["chars"] < 30:
        reasons.append("no text layer")
    if q["dev_matra_start"] > 0.05:
        reasons.append("broken Devanagari shaping")
    if q["odd_chars_per_1k"] > 5:
        reasons.append("private-use / control characters")
    if q["glued_words"] > 0.03:
        reasons.append("words glued together (bad OCR layer)")
    if q["kruti_hits_per_1k_words"] > 20:
        reasons.append("legacy Kruti Dev font")
    q["route"] = "ocr" if reasons else "native"
    q["reasons"] = reasons
    return q
