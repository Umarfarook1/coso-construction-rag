"""Paths and env loading. Everything is relative to the repo root so a fresh clone works anywhere."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "RAG-Interview-Test-Corpus"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"
INDEX_DIR = ROOT / "data" / "index"
MODELS_DIR = ROOT / "models"

# The three documents this system answers over (see README "Why these documents").
DOCS = {
    "gcc": {
        "file": "CPWD-GCC-2020_construction-contract_EN.pdf",
        "title": "CPWD General Conditions of Contract 2020",
        "kind": "contract (born-digital, English)",
    },
    "bsr": {
        "file": "RHB-BSR-2024_schedule-of-rates_EN.pdf",
        "title": "RHB Basic Schedule of Rates 2024",
        "kind": "schedule of rates (table-heavy, English)",
    },
    "nit": {
        "file": "RajasthanPWD-eTender-notice_HINDI_image-only-scan.pdf",
        "title": "Rajasthan PWD e-Tender Notice 68/2024-25",
        "kind": "tender notice (image-only scan, Hindi)",
    },
}


def load_env(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency); real env vars win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
