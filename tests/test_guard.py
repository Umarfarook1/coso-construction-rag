"""The input guardrail (cosorag.guard): fixed replies for greetings, a model-free path for plain greetings, and
every unusable router reply treated as a document question. A stub stands in for the router model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosorag.guard import REPLIES, route, rules  # noqa: E402
from cosorag.llm import LLMReply  # noqa: E402


class Stub:
    name = "stub:router"

    def __init__(self, text: str):
        self.text, self.calls = text, 0

    def chat(self, system, user, **kw):
        self.calls += 1
        return LLMReply(self.text, self.name)


def test_plain_greetings_need_no_model():
    assert rules("hi").sub == "hello" and rules("  Thank you! ").sub == "thanks" and rules("धन्यवाद").lang == "hi"
    assert rules("hi, what is the EMD?") is None and rules("EMD?") is None
    stub = Stub('{"route": "docs"}')
    r = route(stub, "Hello 👋")
    assert r.kind == "greeting" and stub.calls == 0 and r.reply().startswith("Hello.")


def test_router_decides_and_replies_are_fixed_text():
    r = route(Stub('{"route": "off_topic", "lang": "en"}'), "write a poem about the moon")
    assert r.kind == "off_topic" and r.reply() == REPLIES["en"]["off_topic"]
    r = route(Stub('{"route": "about", "lang": "en"}'), "what model are you?")
    assert "Claude Sonnet 5.5" in r.reply("Claude Sonnet 5.5")
    r = route(Stub('{"route": "about", "lang": "en"}'), "आप क्या कर सकते हैं?")
    assert r.lang == "hi"                                        # Devanagari wins over the model's label


def test_unusable_router_reply_means_document_question():
    for text in ("not json", '{"route": "weather"}', ""):
        r = route(Stub(text), "rate of cement")
        assert r.kind == "docs" and r.by == "fallback" and r.question == "rate of cement"
    assert route(None, "rate of cement").kind == "docs"          # no model configured: search as before


def test_follow_up_is_rewritten_in_the_same_call():
    stub = Stub('{"route": "docs", "lang": "en", "question": "Rate of a 20 mm CP brass pillar cock?"}')
    hist = [{"question": "Rate of a 15 mm CP brass pillar cock?", "answer": "Rs 787.00 per Each [S1]."}]
    r = route(stub, "and for 20 mm?", hist)
    assert r.kind == "docs" and r.question == "Rate of a 20 mm CP brass pillar cock?" and stub.calls == 1
