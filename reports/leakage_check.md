# Leakage check

| Setup | Clips | Sources | RF CV macro-F1 |
|---|---|---|---|
| As-is: all clips, clip-level folds | 511 | 247 | 0.482 ± 0.082 |
| Duplicates removed, clip-level folds | 502 | 247 | 0.502 ± 0.082 |
| Duplicates removed, folds grouped by recording source | 502 | 247 | 0.347 ± 0.059 |

5-fold CV × 3 seeds on all clean clips, RF with per-fold balanced class weights. Macro-F1 averages over the classes present in each validation fold.
