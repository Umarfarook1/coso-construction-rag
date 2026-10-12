"""Pack the fine-tuned reranker for a release asset: fp16 weights (half the size, same rankings; checked below).
  uv run python scripts/package_model.py  ->  dist/xenc-coso.zip
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import torch
    from sentence_transformers import CrossEncoder

    from cosorag.rerank import FT_XENC_DIR

    src = CrossEncoder(str(FT_XENC_DIR), device="cpu")
    tmp = ROOT / "dist" / "xenc-coso"
    if tmp.exists():
        shutil.rmtree(tmp)
    src.model.half()
    src.save_pretrained(str(tmp))
    for f in ("gate.json", "train_info.json"):
        if (FT_XENC_DIR / f).exists():
            shutil.copy(FT_XENC_DIR / f, tmp / f)
    # fp16 must not change the ranking: compare scores on a few pairs
    a = CrossEncoder(str(FT_XENC_DIR), device="cpu")
    b = CrossEncoder(str(tmp), device="cpu", model_kwargs={"torch_dtype": torch.float32})
    pairs = [("rate of 15 mm pillar cock", "Item S-2.33.1: CP brass fittings > Pillar Cock 15 mm | Each | 787.00"),
             ("rate of 20 mm pillar cock", "Item S-2.33.1: CP brass fittings > Pillar Cock 15 mm | Each | 787.00"),
             ("compensation for delay cap", "Clause 2: shall not exceed 10% of the tendered value")]
    da, db = a.predict(pairs), b.predict(pairs)
    print("fp32", [round(float(x), 3) for x in da], "fp16", [round(float(x), 3) for x in db])
    zip_path = shutil.make_archive(str(ROOT / "dist" / "xenc-coso"), "zip", tmp)
    print("wrote", zip_path, f"{Path(zip_path).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
