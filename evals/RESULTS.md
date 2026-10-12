# Results

All numbers below are recomputed from files in `runs/` and `evals/results/` by `evals/make_results.py`.
Intervals are 95% bootstrap intervals. The gold set is a development set (see `evals/README.md`); the blind set is the clean estimate.

## End to end, gold set (96 questions)

| system | n | judge: correct | key facts all found | citation hit | refusal recall | over-refusal | answers w/ unsupported number |
|---|---|---|---|---|---|---|---|
| v1-norerank: first version, no reranker | 96 | 0.786 [0.71, 0.86] | 0.679 [0.58, 0.79] | 0.893 [0.82, 0.95] | 0.667 [0.42, 0.92] | 0.119 [0.05, 0.19] | 0.190 [0.12, 0.27] |
| v2-norerank: after error analysis, no reranker | 96 | 0.823 [0.76, 0.89] | 0.738 [0.64, 0.83] | 0.929 [0.87, 0.98] | 0.583 [0.33, 0.83] | 0.083 [0.04, 0.15] | 0.119 [0.06, 0.19] |
