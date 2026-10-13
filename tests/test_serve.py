"""Web view guards for a public deployment: access code and question length (no models are loaded)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_access_code_and_length_are_checked_before_any_work(monkeypatch):
    from fastapi.testclient import TestClient

    from cosorag.serve import create_app

    monkeypatch.setenv("COSORAG_ACCESS_CODE", "right-code")
    c = TestClient(create_app())
    assert c.get("/").status_code == 200
    assert c.post("/ask", json={"question": "EMD?"}).status_code == 401
    assert c.post("/ask", json={"question": "EMD?"}, headers={"x-access-code": "wrong"}).status_code == 401
    assert c.get("/page/bsr:S:2.33.1").status_code == 401
    assert c.post("/ask", json={"question": "x" * 501}, headers={"x-access-code": "right-code"}).status_code == 422
