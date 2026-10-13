"""Follow-up questions in a conversation ("and for 20 mm?") are rewritten into a standalone question before
retrieval, so the retrievers, the reranker and the verifier see exactly what they were built and evaluated for.
The rewrite is shown to the reader; a first question is never rewritten.
"""
from __future__ import annotations

from .llm import LLM, parse_json

CONDENSE = """You rewrite the last message of a conversation about Indian construction documents (a CPWD contract, the RHB schedule of rates 2024 and a Hindi Rajasthan PWD tender notice) into ONE standalone question that can be understood without the conversation.
Rules:
- Resolve references such as "it", "that item", "the same for 20 mm", "and in Hindi?" using the conversation.
- Keep every item number, clause number, size, unit and amount exactly as written.
- Keep the language of the last message (English, Hindi or Hinglish).
- If the last message is already standalone, return it unchanged. Never answer it.
Return ONLY a JSON object: {"question": "..."}"""


def standalone(llm: LLM, history: list[dict], question: str) -> str:
    if not history:
        return question
    turns = "\n".join(f"Q: {h['question']}\nA: {h['answer'][:600]}" for h in history[-3:])
    reply = llm.chat(CONDENSE, f"Conversation:\n{turns}\n\nLast message: {question}", max_tokens=1000)
    try:
        q = str(parse_json(reply.text).get("question", "")).strip()
    except Exception:  # noqa: BLE001  (an unparseable rewrite must not block the question)
        return question
    return q[:500] if q else question
