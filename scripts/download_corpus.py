"""Download the COSO test corpus (public Google Drive folder) into data/raw/. Only needed to re-run ingestion
or to view cited pages; answering questions works from the committed data/processed/ files.

  uv run --with gdown python scripts/download_corpus.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FOLDER = "https://drive.google.com/drive/folders/1FcKLOYtcSPljYe4GCrmAmp-M4McD_OID"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
NEEDED = ["CPWD-GCC-2020_construction-contract_EN.pdf", "RHB-BSR-2024_schedule-of-rates_EN.pdf",
          "RajasthanPWD-eTender-notice_HINDI_image-only-scan.pdf"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    corpus = RAW / "RAG-Interview-Test-Corpus"
    if not all((corpus / f).exists() for f in NEEDED):
        subprocess.run([sys.executable, "-m", "gdown", "--folder", FOLDER, "-O", str(RAW)], check=True)
    missing = [f for f in NEEDED if not (corpus / f).exists()]
    print("missing:", missing) if missing else print(f"corpus ready in {corpus}")


if __name__ == "__main__":
    main()
