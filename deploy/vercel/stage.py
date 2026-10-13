"""Assemble the Vercel project in dist/coso-rag-web: the entrypoint, the package, the processed chunks, the
precomputed Cohere vectors and the three source PDFs (about 50 MB, under the Hobby upload limit of 100 MB).
No model weights: every model is a Bedrock API call.
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
VECTORS = "vectors__bedrock__cohere.embed-multilingual-v3"
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
    for f in ("app.py", "pyproject.toml", "vercel.json"):
        shutil.copy(HERE / f, OUT / f)
    (OUT / ".vercelignore").write_text(".venv/\n__pycache__/\n.env*\n")   # .env.local holds the CLI token
    shutil.copytree(ROOT / "src" / "cosorag", OUT / "src" / "cosorag",
                    ignore=shutil.ignore_patterns("__pycache__", "ingest", "cli.py", "textlayer.py"))
    if not (OUT / "src" / "cosorag" / "web" / "assets" / "metrics.json").exists():
        raise SystemExit("missing web/assets/metrics.json: run evals/all_metrics.py first")
    (OUT / "data" / "processed").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "processed" / "chunks.jsonl.gz", OUT / "data" / "processed")
    (OUT / "data" / "index").mkdir(parents=True)
    for ext in (".npy", ".json"):
        src = ROOT / "data" / "index" / (VECTORS + ext)
        if not src.exists():
            raise SystemExit(f"missing {src}: build the Cohere vectors first (Index.build(model_name='bedrock:cohere.embed-multilingual-v3'))")
        shutil.copy(src, OUT / "data" / "index")
    raw = OUT / "data" / "raw" / RAW_DIR.name
    raw.mkdir(parents=True)
    for d in DOCS.values():
        src = RAW_DIR / d["file"]
        if not src.exists():
            raise SystemExit(f"missing {src}: run scripts/download_corpus.py first")
        shutil.copy(src, raw)
    subprocess.run(["uv", "lock"], cwd=OUT, check=True)   # resolves Linux wheels for the build machine
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"staged {OUT} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
