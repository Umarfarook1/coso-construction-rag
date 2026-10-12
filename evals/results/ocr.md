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
