# OCR bake-off on the Hindi scans

Six regions (61 lines, 86 numbers) from the Rajasthan PWD tender (`raj*`) and the PMGSY guideline scan (`pm*`),
transcribed by hand in `refs/` from the crops in `regions/`. Each engine's raw output is in `out/`.

```bash
uv run python evals/ocr_bakeoff/score.py evals/ocr_bakeoff/refs evals/ocr_bakeoff/out
```

CER is the Levenshtein distance over the reference length after Unicode NFC, dash and Devanagari-digit
normalisation; "numbers" counts reference number tokens found in the output. Local engines (Tesseract via a
portable conda-forge build, RapidOCR via pip) ran on a 4-core laptop CPU; vision models ran on AWS Bedrock
(`run_vlm.py`). Native PaddleOCR and EasyOCR took 1-3 minutes per region on this CPU and were stopped; Surya now
needs a separate llama.cpp server on CPU.
