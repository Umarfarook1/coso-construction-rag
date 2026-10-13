"""Vercel entrypoint for the web view. Assembled into a deployable folder by deploy/vercel/stage.py.

The function's filesystem is read-only except /tmp, and the models were fetched on the build machine by
build.py (they are too large to upload with the source), so everything below points at local files.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("COSORAG_CACHE_DIR", "/tmp/cosorag/cache")
os.environ.setdefault("COSORAG_VIEWS_DIR", "/tmp/cosorag/views")
os.environ.setdefault("COSORAG_EMBEDDER_DIR", str(HERE / "models" / "multilingual-e5-small"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/tmp/hf")
sys.path.insert(0, str(HERE / "src"))

from cosorag.serve import create_app  # noqa: E402

app = create_app()
