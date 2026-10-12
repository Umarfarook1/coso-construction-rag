# Evaluation data and how to read it

| Set | Size | Written how | Used for |
|---|---|---|---|
| `gold/gcc.jsonl`, `gold/bsr.jsonl`, `gold/nit.jsonl` | 32 + 36 + 28 = 96 | Drafted with an AI assistant directly from the source pages (rendered and zoomed for the scan) before the system was run; every answer re-checked against the page by a second pass. Each item: question, gold answer, key facts, gold page + locator, type, page split. | Development: one round of error analysis on the first run led to general fixes (document routing, prompt rules, verifier arithmetic). Numbers on this set are therefore optimistic. |
| `gold_blind/blind.jsonl` | 30 | Written by a fresh pass after the code freeze, only from held-out (test) pages; never read while building. | The unbiased estimate. Run once. |
| `gold/nit_fields.json` | 96 fields (94 contain digits and are scored) | Every amount, date, budget head and handwritten number on the 6 tender pages, read by eye at high zoom. | OCR digit accuracy (`ocr_eval.py`). |
| `../adapt/data/queries_{dev,test}.jsonl` | 2,664 / 2,565 | Synthetic (gpt-oss-120b) from DEV / TEST page blocks; the reranker was trained only on TRAIN blocks. | Reranker before/after with large n (`adapt/eval_reranker.py`). |

Question types: lookup, interpretation, multi-hop, Schedule-F-dependent, practitioner phrasing (no clause number),
exact rate, child row, ditto row, note modifier (percentage extras), item lookup, ambiguous item number,
cross-lingual (English question, Hindi source), Hindi, Hinglish, table, conditional, and the negatives:
unanswerable, near-miss (a size or clause that does not exist), false premise.

## Metrics

* **Judge: correct**: Claude Sonnet 4.6 compares the answer with the gold answer and key facts (correct = 1,
  partial = 0.5). It is a different model family from the generator and does not see the sources.
* **Key facts all found**: deterministic; every key fact appears in the answer (dates and units format-tolerant,
  numbers exact). A strict lower bound that needs no judge.
* **Citation hit**: at least one cited chunk carries the gold clause or rate item (page level for the tender,
  which has no finer locator).
* **Negatives handled**: on unanswerable, near-miss and false-premise questions, the judge confirms the answer
  declined or rejected the premise (ambiguous item numbers, where the right move is to list each chapter, are
  graded by the judge but kept out of this column).
* **Over-refusal**: the model flagged an answerable question as not answerable.
* **Answers with an unsupported number**: a number that is not in any cited source, not in the question and not
  one arithmetic step from them.
* Retrieval: Hit@1, Hit@5, MRR@10 at locator level (`harness.py retrieval`).

Every headline has a 95% bootstrap interval. With n = 96 a difference of a few points is noise; the intervals say
which differences are not.
