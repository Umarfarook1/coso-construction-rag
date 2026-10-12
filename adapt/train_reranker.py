"""Fine-tune the multilingual cross-encoder on corpus-specific pairs (CPU).

Base: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (118M params, trained on mMARCO incl. Hindi; Apache 2.0).
Loss: binary cross-entropy on (query, passage, label) from adapt/build_pairs.py.
The 96M-parameter word-embedding table is frozen: it keeps the multilingual vocabulary intact (less forgetting)
and makes each CPU step cheaper. One epoch, then the gate threshold is picked on the DEV split.
Usage: uv run python adapt/train_reranker.py [--epochs 1] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    from cosorag.config import ROOT
    from cosorag.rerank import BASE_XENC, FT_XENC_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="use only N training pairs (smoke test)")
    ap.add_argument("--out", default=str(FT_XENC_DIR))
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)

    rows = [json.loads(l) for l in (ROOT / "adapt/data/pairs_train.jsonl").open(encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    pos = sum(r["label"] for r in rows)
    print(f"train pairs {len(rows)} (pos {int(pos)}, neg {len(rows) - int(pos)})")
    ds = Dataset.from_dict({"query": [r["query"] for r in rows], "passage": [r["passage"] for r in rows],
                            "label": [float(r["label"]) for r in rows]})

    model = CrossEncoder(BASE_XENC, num_labels=1, max_length=256, device="cpu")
    for name, p in model.model.named_parameters():
        if "word_embeddings" in name:
            p.requires_grad = False
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor((len(rows) - pos) / max(pos, 1)))
    args = CrossEncoderTrainingArguments(
        output_dir=str(ROOT / "runs/tmp/xenc"), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        learning_rate=a.lr, warmup_ratio=0.1, weight_decay=0.01, logging_steps=25, save_strategy="no",
        dataloader_num_workers=0, report_to=[], seed=0, use_cpu=True,
    )
    t = time.time()
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    print(f"trained in {(time.time() - t) / 60:.1f} min")
    model.save_pretrained(a.out)
    (Path(a.out) / "train_info.json").write_text(json.dumps({
        "base": BASE_XENC, "pairs": len(rows), "positives": int(pos), "epochs": a.epochs, "lr": a.lr,
        "batch": a.batch, "minutes": round((time.time() - t) / 60, 1), "frozen": "word_embeddings"}, indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
