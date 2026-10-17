# RF vs CNN on the held-out test split

Same stratified 80/20 split for both models (103 test clips). Brackets: bootstrap 95% CI over test clips (1000 resamples).

| Metric                            | RF              | CNN             |
|:----------------------------------|:----------------|:----------------|
| Test accuracy                     | 78.6% [71%–86%] | 28.2% [19%–36%] |
| Test macro-F1                     | 38.1% [23%–48%] | 25.8% [13%–36%] |
| CV macro-F1 (5-fold, train split) | 0.421 ± 0.054   | 0.254 ± 0.072   |
| F1 belly_pain (n=3)               | 0.40            | 0.25            |
| F1 burping (n=4)                  | 0.67            | 0.17            |
| F1 cold_hot (n=1)                 | 0.00            | 0.29            |
| F1 discomfort (n=6)               | 0.00            | 0.14            |
| F1 hungry (n=77)                  | 0.90            | 0.39            |
| F1 lonely (n=2)                   | 0.00            | 0.00            |
| F1 scared (n=4)                   | 0.80            | 0.67            |
| F1 tired (n=6)                    | 0.29            | 0.15            |

![Confusion matrices](figures/confusion_matrices.png)

## Leakage check (RF only)

| Setup | Clips | Sources | RF CV macro-F1 |
|---|---|---|---|
| As-is: all clips, clip-level folds | 511 | 247 | 0.482 ± 0.082 |
| Duplicates removed, clip-level folds | 502 | 247 | 0.502 ± 0.082 |
| Duplicates removed, folds grouped by recording source | 502 | 247 | 0.347 ± 0.059 |

5-fold CV × 3 seeds on all clean clips, RF with per-fold balanced class weights. Macro-F1 averages over the classes present in each validation fold.
