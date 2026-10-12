# Results

All numbers below are recomputed from files in `runs/` and `evals/results/` by `evals/make_results.py`.
Intervals are 95% bootstrap intervals. The gold set is a development set (see `evals/README.md`); the blind set is the clean estimate.

## End to end, gold set (96 questions)

| system | n | judge: correct | key facts all found | citation hit | negatives handled | over-refusal | answers w/ unsupported number |
|---|---|---|---|---|---|---|---|
| v1-norerank: first version, no reranker | 96 | 0.786 [0.71, 0.86] | 0.679 [0.58, 0.79] | 0.893 [0.82, 0.95] | 0.700 [0.40, 1.00] | 0.119 [0.05, 0.19] | 0.190 [0.12, 0.27] |
| v2-norerank: after error analysis, no reranker | 96 | 0.839 [0.78, 0.89] | 0.786 [0.69, 0.87] | 0.940 [0.88, 0.99] | 0.600 [0.30, 0.90] | 0.024 [0.00, 0.06] | 0.060 [0.01, 0.12] |
| v2-rerank-base: + off-the-shelf cross-encoder, re-sorted | 96 | 0.818 [0.75, 0.88] | 0.738 [0.64, 0.83] | 0.881 [0.81, 0.95] | 0.700 [0.40, 1.00] | 0.060 [0.01, 0.12] | 0.095 [0.04, 0.17] |
| v2-rerank-base-fused: + off-the-shelf cross-encoder, fused with hybrid order | 96 | 0.865 [0.81, 0.92] | 0.738 [0.64, 0.82] | 0.905 [0.83, 0.96] | 0.600 [0.30, 0.90] | 0.048 [0.01, 0.10] | 0.143 [0.07, 0.23] |
| ft-gate-in-prompt: + fine-tuned cross-encoder, re-sorted, gate note in the prompt (first version) | 96 | 0.807 [0.74, 0.87] | 0.702 [0.61, 0.80] | 0.881 [0.81, 0.94] | 0.700 [0.40, 1.00] | 0.107 [0.05, 0.17] | 0.095 [0.04, 0.17] |
| final: + fine-tuned cross-encoder, fused with hybrid order, gate as review flag (submitted) | 96 | 0.849 [0.79, 0.91] | 0.750 [0.65, 0.85] | 0.929 [0.87, 0.98] | 0.800 [0.50, 1.00] | 0.048 [0.01, 0.10] | 0.083 [0.04, 0.14] |

Paired comparisons (same questions):
- no reranker -> submitted: judge 0.839 -> 0.849 (n=96; 9 questions better, 7 worse; paired bootstrap P(no improvement) = 0.36)
- off-the-shelf fused -> fine-tuned fused (the adaptation alone): judge 0.865 -> 0.849 (n=96; 7 questions better, 8 worse; paired bootstrap P(no improvement) = 0.78)
- first fine-tuned version -> submitted: judge 0.807 -> 0.849 (n=96; 8 questions better, 3 worse; paired bootstrap P(no improvement) = 0.03)

## Blind set (30 questions, written after the code freeze)

Each configuration was run on it exactly once. The first fine-tuned version was frozen and run on the
blind set before its gold-set regression was analysed; the fix (fusion, gate as a flag) was chosen on the
gold set only, and the blind questions were never read. Both runs are reported.

| system | n | judge: correct | key facts all found | citation hit | negatives handled | over-refusal | answers w/ unsupported number |
|---|---|---|---|---|---|---|---|
| no reranker | 30 | 0.850 [0.72, 0.95] | 0.667 [0.48, 0.86] | 0.952 [0.86, 1.00] | 0.667 [0.33, 0.89] | 0.048 [0.00, 0.14] | 0.048 [0.00, 0.14] |
| off-the-shelf cross-encoder, fused | 30 | 0.850 [0.73, 0.93] | 0.619 [0.38, 0.81] | 0.952 [0.86, 1.00] | 0.556 [0.22, 0.89] | 0.048 [0.00, 0.14] | 0.095 [0.00, 0.24] |
| first fine-tuned version (gate note in the prompt) | 30 | 0.867 [0.75, 0.97] | 0.714 [0.52, 0.90] | 0.952 [0.86, 1.00] | 0.778 [0.44, 1.00] | 0.048 [0.00, 0.14] | 0.095 [0.00, 0.24] |
| submitted system (fine-tuned, fused) | 30 | 0.900 [0.80, 0.98] | 0.667 [0.48, 0.86] | 0.952 [0.86, 1.00] | 0.889 [0.67, 1.00] | 0.048 [0.00, 0.14] | 0.095 [0.00, 0.24] |

- no reranker -> submitted: judge 0.850 -> 0.900 (n=30; 4 questions better, 1 worse; paired bootstrap P(no improvement) = 0.23)
- off-the-shelf fused -> fine-tuned fused (the adaptation alone): judge 0.850 -> 0.900 (n=30; 4 questions better, 2 worse; paired bootstrap P(no improvement) = 0.21)

## Submitted system by slice

## final  (bedrock:openai.gpt-oss-120b-1:0, reranker=models/xenc-coso, split=all, n=96)

| slice | n | judge: correct | key facts all found | citation hit | negatives handled | over-refusal | answers w/ unsupported number |
|---|---|---|---|---|---|---|---|
| all | 96 | 0.849 [0.79, 0.91] | 0.750 [0.65, 0.85] | 0.929 [0.87, 0.98] | 0.800 [0.50, 1.00] | 0.048 [0.01, 0.10] | 0.083 [0.04, 0.14] |
| split=dev | 20 | 0.850 [0.72, 0.95] | 0.688 [0.44, 0.88] | 1.000 [1.00, 1.00] | 0.667 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.062 [0.00, 0.19] |
| split=test | 76 | 0.849 [0.78, 0.91] | 0.765 [0.68, 0.87] | 0.912 [0.84, 0.97] | 0.857 [0.57, 1.00] | 0.059 [0.01, 0.12] | 0.088 [0.03, 0.16] |
| gcc | 32 | 0.859 [0.77, 0.95] | 0.821 [0.68, 0.96] | 0.929 [0.82, 1.00] | 0.750 [0.25, 1.00] | 0.036 [0.00, 0.11] | 0.071 [0.00, 0.18] |
| bsr | 36 | 0.847 [0.72, 0.94] | 0.806 [0.68, 0.94] | 0.871 [0.74, 0.97] | 1.000 [1.00, 1.00] | 0.097 [0.00, 0.23] | 0.129 [0.03, 0.26] |
| nit | 28 | 0.839 [0.73, 0.93] | 0.600 [0.40, 0.80] | 1.000 [1.00, 1.00] | 0.667 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.040 [0.00, 0.12] |

refusal precision 0.667 [0.42, 0.92]; parse errors 0; latency p50 7.5s, p95 11.0s; judge pass 0.849 [0.79, 0.91]

## Adaptation before/after on held-out synthetic queries (TEST page blocks, never trained on)

400 real queries, 200 near-miss queries. Candidates: hybrid top-20 plus each document's top-3.

| reranking | Hit@1 | Hit@8 (in context) | MRR@10 |
|---|---|---|---|
| none (hybrid order) | 0.537 | 0.800 | 0.632 |
| off-the-shelf cross-encoder, re-sorted | 0.652 | 0.848 | 0.723 |
| fine-tuned cross-encoder, re-sorted | 0.708 | 0.882 | 0.779 |
| off-the-shelf, fused with hybrid order (RRF) | 0.590 | 0.850 | 0.684 |
| fine-tuned, fused with hybrid order (RRF; submitted) | 0.640 | 0.873 | 0.726 |

| gate (real vs near-miss query) | AUROC on the source passage | AUROC on the top candidate |
|---|---|---|
| off-the-shelf | 0.450 | 0.407 |
| fine-tuned | 0.838 | 0.772 |

## Does the gate transfer to real questions? (gold set, `evals/gate_gold.py`)

Top cross-encoder score per question; 84 answerable vs 12 not answerable. Threshold 0.7 was chosen on synthetic DEV queries.

| model | AUROC answerable vs not | answerable below threshold | not answerable below threshold |
|---|---|---|---|
| off-the-shelf | 0.529 | 21% | 25% |
| fine-tuned | 0.727 | 48% | 75% |

## Retrieval on the gold set (answerable questions, locator-level)

| mode | n | Hit@1 | Hit@5 | Hit@8 (in context) | MRR@10 | page Hit@5 |
|---|---|---|---|---|---|---|
| bm25 | 84 | 0.738 | 0.869 | 0.905 | 0.798 | 0.893 |
| dense | 84 | 0.667 | 0.833 | 0.881 | 0.748 | 0.857 |
| hybrid | 84 | 0.786 | 0.952 | 0.976 | 0.855 | 0.976 |
| hybrid+xenc_base | 84 | 0.774 | 0.940 | 0.952 | 0.856 | 0.964 |
| hybrid+xenc_base_fused | 84 | 0.821 | 0.976 | 0.976 | 0.893 | 0.988 |
| hybrid+xenc_ft | 84 | 0.750 | 0.929 | 0.952 | 0.829 | 0.940 |
| hybrid+xenc_ft_fused | 84 | 0.833 | 0.952 | 0.976 | 0.888 | 0.976 |

## OCR on the Hindi tender

Bake-off on 6 hand-transcribed regions (`evals/ocr_bakeoff/`; engine outputs and references included):

| engine | CER | numbers found | word F1 | sec/page |
|---|---|---|---|---|
| vlm-claude-sonnet-4.6 | 0.006 | 86/86 (100%) | 0.993 | 22.2 |
| vlm-llama4-maverick | 0.020 | 81/86 (94%) | 0.939 | 10.3 |
| vlm-gemma3-27b | 0.043 | 86/86 (100%) | 0.934 | 35.1 |
| tesseract-fast-hin+eng-psm6 | 0.128 | 51/86 (59%) | 0.830 | 3.2 |
| tesseract-best-hin+eng-psm6 | 0.157 | 43/86 (50%) | 0.802 | 7.9 |
| tesseract-best-hin+eng-psm3 | 0.164 | 48/86 (56%) | 0.813 | 8.6 |
| vlm-pixtral-large | 0.201 | 71/86 (83%) | 0.685 | 29.0 |
| rapidocr-detv6-score0 | 0.226 | 78/86 (91%) | 0.806 | 10.5 |
| rapidocr-ppocrv5-devanagari | 0.228 | 78/86 (91%) | 0.813 | 7.0 |
| rapidocr-detv5mobile-score0 | 0.309 | 72/86 (84%) | 0.665 | 7.5 |
| rapidocr-det-v5server | 0.472 | 79/86 (92%) | 0.617 | 39.4 |

Field-level digit accuracy on the full tender against 94 hand-read fields (`evals/ocr_eval.py`):

| reader | model | fields correct |
|---|---|---|
| primary | bedrock:eu.anthropic.claude-sonnet-4-6 | 94/94 (100%) |
| second | bedrock:google.gemma-3-27b-it | 70/94 (74%) |
| third | bedrock:us.meta.llama4-maverick-17b-instruct-v1:0 | 92/94 (98%) |

Disagreements between readers were put to Claude Opus 5.5 with the page image as targeted questions. Two were real misreads by the primary reader and were corrected (p. 3 condition 3: "08.00" to "06.00", the bid deadline; p. 4 fee table: "161 lakh" to "151 lakh"); six were alignment false alarms where the primary reading was confirmed. The 151 slab boundary is not one of the 94 hand-read fields, so 94/94 does not count that error.

