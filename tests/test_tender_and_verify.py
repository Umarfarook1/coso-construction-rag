"""Tender OCR safeguards, arithmetic-aware verification and document routing (no PDFs, no models)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosorag.answer import _derivable  # noqa: E402
from cosorag.chunk import load_chunks  # noqa: E402
from cosorag.config import PROCESSED_DIR  # noqa: E402
from cosorag.ingest.vote import vote  # noqa: E402
from cosorag.retrieve import doc_hints  # noqa: E402


def test_vote_corrects_when_both_other_readers_agree():
    primary = "आवेदन 15.04.2025 को सांय 08.00 बजे तक"
    fixed, findings = vote(primary, ["आवेदन 15.04.2025 को सायं 06.00 बजे तक", "आवेदन 15.04.2025 को सांय 06.00 बजे तक"])
    assert "06.00" in fixed and "08.00" not in fixed
    assert findings[0]["action"] == "corrected"


def test_vote_only_flags_a_one_against_one_split():
    fixed, findings = vote("तक | 1500 रूपये | 161 लाख से", ["तक | 1500/- रूपये | 151 लाख से"])
    assert "161" in fixed and findings[0]["action"] == "flagged" and findings[0]["alt"] == "151"


def test_committed_tender_chunks_carry_the_adjudicated_numbers():
    nit = {c.locator: c for c in load_chunks(PROCESSED_DIR / "chunks.jsonl.gz") if c.doc == "nit"}
    cond3 = nit["p. 3, condition 3"]
    assert "06.00" in cond3.text and "08.00" not in cond3.text
    assert {"from": "08.00", "to": "06.00"}.items() <= cond3.meta["ocr_corrections"][0].items()
    assert "151" in nit["p. 4, table"].text
    assert "572940" in nit["p. 3, table"].text and "143235" in nit["p. 3, table"].text
    assert "p. 5, condition 17 (cont.)" in nit          # condition 17 runs over the page break


def test_computed_numbers_are_derivable_but_invented_ones_are_not():
    assert _derivable("320", {"2509", "2189"})            # difference of two cited rates
    assert _derivable("40351.5", {"35868", "12.5"})       # +12.5% note applied to a rate
    assert not _derivable("900", {"787", "15"})


def test_text_layer_quality_routes_broken_layers_to_ocr():
    from cosorag.textlayer import quality

    clean = "प्रधानमंत्री ग्राम सड़क योजना के तीसरे चरण को मंजूरी दी गई है। इसके अंतर्गत सड़कें बनाई जाएंगी।"
    broken = "धानमंी 	ाम सड़क योजना क ि ाम ीण ंिमंडल ￰थक सिम￸त"    # the PMGSY summary's actual layer
    assert quality(clean)["route"] == "native"
    assert quality(broken)["route"] == "ocr"
    glued = "ThisIndianStandardwasadoptedbytheBureau of Indian Standards tofillallthevoidswhichmaybeeither " * 3
    assert "words glued together (bad OCR layer)" in quality(glued)["reasons"]
    kruti = "fufonk ,oa lafonk ds fy, vkSj dk;Z esa çfr'kr nj " * 5
    assert quality(kruti)["route"] == "ocr"


def test_document_hints():
    assert doc_hints("What is the EMD for this tender?") == {"nit"}
    assert "nit" in doc_hints("इस निविदा का UBN नंबर क्या है?")
    assert "bsr" in doc_hints("rate per sqm for paving blocks")
    assert "gcc" in doc_hints("Who decides compensation under the contract?")
