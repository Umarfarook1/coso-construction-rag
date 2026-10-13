"""Vercel build step (runs after the dependencies are installed, before the function is bundled).

Fetches the two models, which are too large to upload with the source on the Hobby plan:
  * the fine-tuned reranker from this repository's GitHub release (fp16, about 220 MB)
  * multilingual-e5-small at the exact revision the committed vectors were computed with
"""
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
XENC_URL = "https://github.com/Umarfarook1/coso-construction-rag/releases/download/v1.0/xenc-coso.zip"
E5 = ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3")
E5_FILES = ["config.json", "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "model.safetensors", "sentencepiece.bpe.model", "1_Pooling/config.json"]


def main() -> None:
    xenc = HERE / "models" / "xenc-coso"
    if not (xenc / "config.json").exists():
        print("fetching the fine-tuned reranker ...", flush=True)
        with urllib.request.urlopen(XENC_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(xenc)
    from huggingface_hub import snapshot_download

    e5 = HERE / "models" / "multilingual-e5-small"
    print("fetching multilingual-e5-small ...", flush=True)
    snapshot_download(E5[0], revision=E5[1], local_dir=e5, allow_patterns=E5_FILES)
    shutil.rmtree(e5 / ".cache", ignore_errors=True)   # download bookkeeping, not needed at runtime
    # the committed vectors must belong to this embedder, or the function would try to re-embed at start
    meta = json.loads((HERE / "data" / "index" / "vectors__intfloat__multilingual-e5-small.json").read_text())
    assert meta["model"] == E5[0], meta["model"]
    assert (xenc / "gate.json").exists() and (e5 / "model.safetensors").exists()
    print("models ready:", sorted(p.name for p in (HERE / "models").iterdir()), flush=True)


if __name__ == "__main__":
    main()
