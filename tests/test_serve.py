"""Web view guards for a public deployment: access code, input limits, model allow-list, static assets.
No models are loaded and no model is called."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _client(monkeypatch, code: str = "right-code"):
    from fastapi.testclient import TestClient

    from cosorag.serve import create_app

    monkeypatch.setenv("COSORAG_ACCESS_CODE", code)
    monkeypatch.delenv("COSORAG_LLM", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test")   # makes the Bedrock model list available
    return TestClient(create_app())


def test_access_code_and_length_are_checked_before_any_work(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/").status_code == 200
    assert c.post("/ask", json={"question": "EMD?"}).status_code == 401
    assert c.post("/ask", json={"question": "EMD?"}, headers={"x-access-code": "wrong"}).status_code == 401
    assert c.get("/page/bsr:S:2.33.1").status_code == 401
    assert c.post("/warm").status_code == 401
    assert c.post("/ask", json={"question": "x" * 501}, headers={"x-access-code": "right-code"}).status_code == 422


def test_model_list_and_allow_list(monkeypatch):
    from cosorag.models import DEFAULT, MODELS

    c = _client(monkeypatch)
    r = c.get("/models").json()
    assert r["access_required"] is True and r["default"] == DEFAULT
    assert [m["slug"] for m in r["models"]] == list(MODELS)
    assert all("model_id" not in m for m in r["models"])           # provider ids stay on the server
    ok = {"x-access-code": "right-code"}
    assert c.post("/ask", json={"question": "EMD?", "model": "gpt-5-turbo"}, headers=ok).status_code == 422


def test_history_is_bounded(monkeypatch):
    c = _client(monkeypatch)
    ok = {"x-access-code": "right-code"}
    turns = [{"question": "q", "answer": "a"}] * 7
    assert c.post("/ask", json={"question": "and 20 mm?", "history": turns}, headers=ok).status_code == 422
    long = [{"question": "q" * 501, "answer": "a"}]
    assert c.post("/ask", json={"question": "and 20 mm?", "history": long}, headers=ok).status_code == 422


def test_assets_are_served_by_name_only(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/assets/plex-sans.woff2")
    assert r.status_code == 200 and r.headers["content-type"] == "font/woff2"
    assert "immutable" in r.headers["cache-control"]
    assert c.get("/assets/..%2Fserve.py").status_code == 404
    assert c.get("/assets/index.html").status_code == 404


def test_condense_keeps_first_question_and_falls_back(monkeypatch):
    from cosorag.chat import standalone

    class Broken:
        def chat(self, *a, **k):
            class R:
                text = "not json"
            return R()

    assert standalone(Broken(), [], "Rate of a 15 mm pillar cock?") == "Rate of a 15 mm pillar cock?"
    assert standalone(Broken(), [{"question": "q", "answer": "a"}], "and 20 mm?") == "and 20 mm?"


def test_greeting_is_answered_without_a_search(monkeypatch):
    c = _client(monkeypatch)
    ok = {"x-access-code": "right-code"}
    for msg, lang in (("hi", "en"), ("Thanks!", "en"), ("नमस्ते", "hi")):
        r = c.post("/ask", json={"question": msg}, headers=ok).json()
        assert r["route"] == "greeting" and r["route_by"] == "rules" and r["lang"] == lang
        assert r["citations"] == [] and r["passages"] == [] and r["answer"]
