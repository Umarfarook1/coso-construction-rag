"""Assemble the Vercel project in dist/coso-rag-web: this folder, the package, the processed chunks, the
committed vectors and the three source PDFs (about 40 MB, under the Hobby upload limit of 100 MB). The models
are fetched on Vercel's build machine by build.py.
  uv run python deploy/vercel/stage.py && cd dist/coso-rag-web && vercel deploy --prod
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "dist" / "coso-rag-web"
sys.path.insert(0, str(ROOT / "src"))

from cosorag.config import DOCS, RAW_DIR  # noqa: E402


def main() -> None:
    if OUT.exists():
        keep = OUT / ".vercel"   # project link from an earlier deploy
        saved = shutil.copytree(keep, ROOT / "dist" / ".vercel-link", dirs_exist_ok=True) if keep.exists() else None
        shutil.rmtree(OUT)
        if saved:
            shutil.move(str(saved), str(keep))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("app.py", "build.py", "pyproject.toml", "vercel.json"):
        shutil.copy(HERE / f, OUT / f)
    # models are fetched on the build machine; a local copy (from testing build.py) must never be uploaded
    (OUT / ".vercelignore").write_text("models/\n.venv/\n__pycache__/\n.env*\n")   # .env.local: CLI token
    shutil.copytree(ROOT / "src" / "cosorag", OUT / "src" / "cosorag",
                    ignore=shutil.ignore_patterns("__pycache__", "ingest", "cli.py", "textlayer.py"))
    (OUT / "data" / "processed").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "processed" / "chunks.jsonl.gz", OUT / "data" / "processed")
    shutil.copytree(ROOT / "data" / "index", OUT / "data" / "index")
    raw = OUT / "data" / "raw" / RAW_DIR.name
    raw.mkdir(parents=True)
    for d in DOCS.values():
        src = RAW_DIR / d["file"]
        if not src.exists():
            raise SystemExit(f"missing {src}: run scripts/download_corpus.py first")
        shutil.copy(src, raw)
    subprocess.run(["uv", "lock"], cwd=OUT, check=True)   # resolves Linux CPU wheels for the build machine
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"staged {OUT} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
