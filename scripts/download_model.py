"""Download the fine-tuned reranker (release asset) into models/xenc-coso.
Without it everything still runs: retrieval falls back to BM25 + dense fusion and the grounding gate is off.
  uv run python scripts/download_model.py
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/Umarfarook1/coso-construction-rag/releases/download/v1.0/xenc-coso.zip"
DEST = Path(__file__).resolve().parents[1] / "models" / "xenc-coso"


def main() -> None:
    if (DEST / "config.json").exists():
        print(f"already present: {DEST}")
        return
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL) as r:
        data = r.read()
    DEST.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(DEST)
    print(f"reranker ready in {DEST}")


if __name__ == "__main__":
    main()
