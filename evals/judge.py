"""Answer correctness: a deterministic fact matcher (strict lower bound) and an LLM judge (primary).

The judge is Claude Sonnet 4.6, a different model family from the generator (gpt-oss-120b), which reduces
self-preference bias. It sees the question, the gold answer and key facts, and the system answer; it does not
see the sources, so it grades correctness against the gold, not plausibility. Verdicts are cached.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosorag.llm import LLM, parse_json  # noqa: E402
from cosorag.text import normalize, numbers  # noqa: E402

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
UNITS = {
    "sqm": ["sqm", "sq m", "sq.m", "square metre", "square meter", "m2", "m²", "sq. m", "per square"],
    "mtr": ["mtr", "metre", "meter", "per m ", "rmt", "running metre", "running meter", "rm"],
    "rmt": ["rmt", "running metre", "running meter", "per metre", "per meter", "rm"],
    "running m": ["running m", "running metre", "running meter", "per metre", "per meter"],
    "cum": ["cum", "cubic metre", "cubic meter", "m3", "m³", "cu m"],
    "each": ["each", "per piece", "per unit", "per item", "a piece", "per number", "per no"],
    "p. point": ["point", "per point"],
    "per hect.": ["hectare", "per hect", "ha"],
    "per day": ["per day", "a day", "daily"],
}
DATE_RE = re.compile(r"(\d{1,2})[.\-/ ](\d{1,2}|[a-z]{3,9})[.\-/ ,]+(\d{2,4})")


def _dates(text: str) -> set[tuple[int, int, int]]:
    out = set()
    for d, m, y in DATE_RE.findall(normalize(text)):
        mm = int(m) if m.isdigit() else MONTHS.get(m[:3], 0)
        yy = int(y) + (2000 if len(y) == 2 else 0)
        if mm:
            out.add((int(d), mm, yy))
    return out


def fact_found(fact: str, answer: str) -> bool:
    a = re.sub(r"[-‐-―]", " ", normalize(answer))
    f = re.sub(r"[-‐-―]", " ", normalize(fact)).strip(" .")
    if DATE_RE.search(normalize(fact)):
        return bool(_dates(fact) & _dates(answer)) or f in a
    key = f.rstrip(".")
    if key in UNITS:
        return any(u in a for u in UNITS[key]) or key in a
    nums = numbers(fact)
    if nums:
        return nums <= numbers(answer)
    words = [w.rstrip("s") for w in re.findall(r"\w+", f)]
    return all(w in a for w in words)


JUDGE_SYSTEM = """You grade answers from a document question-answering system against a gold answer written by a domain expert.
Rules:
- "correct": the answer states the gold facts (same numbers, units, conditions; formats may differ) and adds nothing that contradicts the gold. Extra correct detail is fine.
- "partial": some gold facts are right but a key fact is missing or one part is wrong.
- "incorrect": the key fact is wrong or missing, or the answer refuses although the gold has an answer.
- If the gold says the documents do NOT answer the question (answerable=false), the answer is "correct" only if it declines or clearly says the information is not in the documents / the premise is wrong; giving a confident specific answer is "incorrect".
Return ONLY JSON: {"verdict": "correct|partial|incorrect", "reason": "one sentence"}"""


def judge(question: str, gold: dict, answer: str, llm: LLM | None = None) -> dict:
    llm = llm or LLM("bedrock:eu.anthropic.claude-sonnet-4-6", region="eu-north-1")
    user = (f"Question: {question}\nGold answerable: {gold['answerable']}\nGold answer: {gold['answer']}\n"
            f"Gold key facts: {gold.get('key_facts', [])}\n\nSystem answer:\n{answer}")
    r = llm.chat(JUDGE_SYSTEM, user, max_tokens=400)
    try:
        d = parse_json(r.text)
    except Exception:  # noqa: BLE001
        d = {"verdict": "incorrect", "reason": f"unparseable judge reply: {r.text[:80]}"}
    d["score"] = {"correct": 1.0, "partial": 0.5}.get(d.get("verdict"), 0.0)
    return d
