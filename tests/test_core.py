"""Fast tests (no PDFs, no model, no network): they run on the committed processed chunks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosorag.answer import verify  # noqa: E402
from cosorag.chunk import Chunk, load_chunks  # noqa: E402
from cosorag.config import PROCESSED_DIR  # noqa: E402
from cosorag.text import numbers, tokenize  # noqa: E402


@pytest.fixture(scope="module")
def index():
    from cosorag.index import Index

    return Index(load_chunks(PROCESSED_DIR / "chunks.jsonl.gz"), None, None)


def by_id(index, cid):
    return index.chunks[index.by_id[cid]]


def test_tokenizer_keeps_identifiers_whole():
    assert "2.33.1" in tokenize("rate of item S-2.33.1")
    assert "10cc" in tokenize("Clause 10 CC price variation")
    assert "pwd2425wsob17397" in tokenize("UBN PWD2425WSOB17397")
    assert "572940" in tokenize("धरोहर राशि ५,७२,९४०")          # Devanagari digits + Indian grouping


def test_numbers_are_canonical():
    assert numbers("Rs. 1188.00, 5,72,940 and 0.50%") == {"1188", "572940", "0.5"}


def test_bsr_child_row_inherits_parent_and_page_labels(index):
    c = by_id(index, "bsr:S:2.33.1")
    assert "Pillar Cock 15 mm" in c.text and "CP Brass fittings" in c.text   # child + parent description
    assert c.meta["rate"] == "787.00" and c.meta["unit"] == "Each"
    assert (c.pdf_page, c.page_label) == (260, "S-21")
    assert c.cite().endswith("Item S-2.33.1, p. S-21 (PDF p. 260)")


def test_bsr_ditto_rows_are_resolved(index):
    c = by_id(index, "bsr:S:1.8.2")
    assert "Solid PVC" in c.text and "White" in c.text and "-do-, i.e." in c.text


def test_gcc_clause_title_comes_from_margin(index):
    c = by_id(index, "gcc:Clause-2:0")
    assert c.title == "Compensation for Delay" and c.pdf_page == 16 and c.page_label == "14"


def test_identifier_lookup(index):
    from cosorag.retrieve import find_identifiers

    found = {lk["id"]: lk for lk in find_identifiers(index, "What is item S-2.33.1? And S-2.33.40? And clause 10CC, clause 47?")}
    assert found["Item S-2.33.1"]["found"] and not found["Item S-2.33.40"]["found"]
    assert found["Clause 10CC"]["found"] and not found["Clause 47"]["found"]
    amb = find_identifiers(index, "What is the rate for item 1.1.1?")[0]
    assert amb["found"] and amb["ambiguous"]


def test_measurements_and_clause_words_are_not_ids(index):
    from cosorag.retrieve import find_identifiers

    # "2.2 kW", "1.6 mm", "12.5 per cent" are measurements, not BSR item numbers
    assert find_identifiers(index, "Rate of a 2.2 kW pump with 1.6 mm sheet and 12.5 per cent extra?") == []
    # "Clause 2 of the contract" is Clause 2, not "Clause 2OF"
    ids = [lk["id"] for lk in find_identifiers(index, "What does Clause 2 of the contract say? And clause 10 CC?")]
    assert ids == ["Clause 2", "Clause 10CC"]


def _src(text):
    return Chunk(id="x", doc="bsr", pdf_page=1, page_label="1", locator="Item S-1", text=text)


def test_verify_accepts_grounded_answer():
    src = [_src("Item S-2.33.1: Pillar Cock 15 mm nominal size | Unit: Each | Rate: Rs. 787.00")]
    data = {"answerable": True, "quotes": [{"source": "S1", "quote": "Pillar Cock 15 mm nominal size"}],
            "answer": "The rate is Rs. 787 per piece [S1]."}
    assert verify("rate of 15 mm pillar cock?", data, src)["ok"]


def test_verify_flags_number_not_in_sources_and_fake_quote():
    src = [_src("Item S-2.33.1: Pillar Cock 15 mm nominal size | Unit: Each | Rate: Rs. 787.00")]
    data = {"answerable": True, "quotes": [{"source": "S1", "quote": "Pillar Cock 20 mm costs 900"}],
            "answer": "The rate is Rs. 900 [S1]."}
    chk = verify("rate of pillar cock?", data, src)
    assert not chk["ok"] and chk["unsupported_numbers"] == ["900"] and len(chk["bad_quotes"]) == 1
