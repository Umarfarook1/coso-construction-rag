"""Deterministic page-block splits, fixed before any training data or gold question was written.

Synthetic fine-tuning queries are generated only from TRAIN pages of the GCC and the BSR; gold questions on
TEST pages therefore measure generalisation to text the adapted model never saw. Blocks (not single pages)
keep a clause or a parent item and its children on one side of the split.
The 6-page Hindi tender is too small to hold out: it is used whole for training (transductive, like a real
deployment that adapts to its own corpus) and its gold questions are flagged as such in the report.
"""
from __future__ import annotations

BLOCK = {"gcc": 4, "bsr": 8}


def page_split(doc: str, pdf_page: int) -> str:
    if doc not in BLOCK:
        return "train"
    b = (pdf_page - 1) // BLOCK[doc]
    return {2: "test", 4: "dev"}.get(b % 5, "train")
