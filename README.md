# Infant Cry State Recognition

Predict what a baby's cry is about (one of 8 parent-reported states) from a ~7 s audio clip,
first with classical machine learning, then with deep learning. The project is built as a
sequence of notebooks, and every number in this README comes from running them.

The full plan (architecture, evaluation protocol, build order) is in
[`docs/PROJECT_PLAN.pdf`](docs/PROJECT_PLAN.pdf).

| Notebook | Status |
|---|---|
| [`01_data_audit.ipynb`](notebooks/01_data_audit.ipynb): what the data really contains | done |
| [`02_eda.ipynb`](notebooks/02_eda.ipynb): what the cries look like, and whether the classes differ | done |
| [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb): feature set and recording-format leak test | done |
| [`04_evaluation_protocol.ipynb`](notebooks/04_evaluation_protocol.ipynb): grouped cross-validation folds, metrics, baselines | done |
| [`05_machine_learning.ipynb`](notebooks/05_machine_learning.ipynb): five classical models under grouped nested CV | done |
| [`06_error_analysis.ipynb`](notebooks/06_error_analysis.ipynb): what the best model gets wrong, and why | done |
| [`07_deep_learning.ipynb`](notebooks/07_deep_learning.ipynb): CNN on spectrograms (run on Google Colab, T4 GPU) | done |
| `08_conclusions.ipynb`: comparison and model card | planned |

So far no model, classical (notebook 05) or CNN (notebook 07), beats the majority-class baseline when tested on babies it has not seen.

## Data

512 `.wav` files in 8 folders: `belly_pain`, `burping`, `cold_hot`, `discomfort`, `hungry`,
`lonely`, `scared`, `tired`. Labels were entered by parents in a crowd-sourced app, so they are
noisy and there is no known ceiling for accuracy. The audio is not stored in this repo.

Findings from notebook 01:

| class | files | generated copies | label mismatch | **clean clips** | sources |
|---|---|---|---|---|---|
| belly_pain | 16 | 0 | 0 | **16** | 7 |
| burping | 18 | 9 | 0 | **9** | 8 |
| cold_hot | 7 | 0 | 0 | **7** | 3 |
| discomfort | 30 | 0 | 9 | **21** | 14 |
| hungry | 382 | 0 | 0 | **382** | 205 |
| lonely | 11 | 0 | 0 | **11** | 8 |
| scared | 20 | 15 | 0 | **5** | 4 |
| tired | 28 | 0 | 0 | **28** | 17 |
| **total** | 512 | 24 | 9 | **479** | 226 |

- **Generated copies.** 24 files named `generated<k>_<original>` are augmented copies of
  clips that are also in the dataset. They are removed. Keeping them would let a copy and its
  original land on opposite sides of a train/test split.
- **Label mismatch.** 9 files in `discomfort/` carry a different label code in their filename
  (one is byte-identical to a `lonely/` clip). They are removed.
- **Format shortcut.** Files are either 8 kHz mono or 44.1 kHz stereo, and the format nearly
  predicts some labels (all *hungry* clips are 8 kHz; all *cold_hot*, *lonely* and *scared*
  clips are 44.1 kHz). The 44.1 kHz files contain no sound above ~4.5 kHz and their two
  channels are identical, so all audio is converted to **8 kHz mono** with nothing lost.
- **Same baby, many clips.** 116 of the 226 sources (one app user, which in practice means one
  baby) contribute more than one clip. All splits are **grouped by source**, so no baby appears
  in both training and evaluation.
- **Imbalance.** Always answering *hungry* gives accuracy 0.797 but balanced accuracy 0.125
  and macro-F1 0.111. Models are judged on **balanced accuracy and macro-F1** against that
  baseline, not on accuracy.
- *scared*, *cold_hot* and *burping* have fewer than 10 clean clips each, from 3 to 8 babies.
  Scores for these classes will be unreliable whatever the model.

The list of clips used for modelling is in [`metadata/clean_index.csv`](metadata/clean_index.csv).

Findings from notebook 02 (each class compared with one value per baby, not per clip):

- **No single interpretable descriptor separates the classes.** Pitch, voicing, share of time
  crying, burst and pause length and spectral centroid all have Kruskal–Wallis effect sizes of
  ε² ≤ 0.014, and no test survives Holm correction. The largest hungry-vs-rest differences
  (voicing, pitch) have corrected p = 0.29–0.30.
- **Babies differ more than classes do.** Within *hungry*, the intra-class correlation of the
  descriptors across babies is 0.12–0.43. This is why every split is grouped by baby.
- **Some clips hold little or no crying.** 15 clips are voiced in under 10% of their cry frames,
  and the least-voiced ones are steady noise.
- **Possible format trace.** *cold_hot* and *lonely* (all originally 44.1 kHz) are 4–7 dB below
  average at 3.5–4 kHz. Notebook 03 tests this directly.

Findings from notebook 03:

- **225 features per clip** in six groups (log-mel bands, MFCC with deltas, spectral shape, pitch
  and voicing, rhythm, energy dynamics), computed over cry frames only and with no absolute
  loudness. None of them differs between classes on its own after Holm correction (largest
  ε² = 0.038).
- **The recording format still shows through.** A format detector trained on other classes ranks
  the 44.1 kHz clips of *tired*, *discomfort* and *burping* above most 8 kHz clips of the same classes:
  AUROC 0.86 with full-band features, 0.82 band-limited to 150–3800 Hz, and 0.76 without the per-band
  means. This is based on only 6 such clips from 4 babies. The band-limited features are the default; notebook 06
  checks whether the class model actually uses this trace.

Evaluation protocol (notebook 04), used unchanged by every model:

- **25 splits:** 5 folds grouped by baby and stratified by label, repeated with 5 seeds, saved in
  [`metadata/folds.csv`](metadata/folds.csv). Metrics are computed on the pooled out-of-fold
  predictions of each repeat and reported as mean ± std over repeats, with 95% intervals from a
  bootstrap that resamples babies.
- **Baselines** (balanced accuracy / macro-F1): majority 0.125 / 0.111, prior-random 0.132 / 0.132.
  Top-3 accuracy is already 0.900 from listing the three most common classes.
- **Leakage matters.** One untuned logistic regression scores balanced accuracy 0.271 with
  ordinary clip-level folds, but 0.125 when grouped by baby, which is no better than the baselines.
  Clip-level numbers are therefore never reported as results.
- **Noise floor.** Paired-bootstrap intervals are about 0.07 wide, so model differences below
  roughly 0.035–0.04 balanced accuracy cannot be distinguished from noise.

Classical models (notebook 05). Logistic regression, kNN, RBF SVM, random forest and gradient boosting
were tuned by grouped inner CV inside each of the 25 outer splits. Each comes in two variants, plain
argmax and a prior-corrected decision rule whose strength is also tuned:

- **None beats the baselines.** Balanced accuracy is 0.103–0.131 across the 10 variants (majority
  0.125). Every paired 95% interval against majority includes 0. The best, random forest (argmax),
  scores 0.131 (95% CI 0.123–0.152), +0.006 (−0.002, +0.016) vs majority.
- **The models fall back on *hungry*.** The argmax variants reach 0.78–0.80 accuracy (majority 0.797).
  In the best model, *tired* is the only minority class with any correct predictions (recall 0.071).
- **Prior correction hurts.** It lowers balanced accuracy for 4 of 5 models, although the inner loop
  scores the chosen settings at 0.136–0.173. This is selection optimism on very small classes.
- **No ranking signal.** One-vs-rest AUROCs are 0.23–0.60, no log-loss beats the prior's 0.887, and
  no top-3 accuracy beats the baseline's 0.900.
- **Feature subsets and the full band** move the random forest between 0.127 and 0.140. No setting is
  clearly above majority; the highest, B+C at +0.015 (0.000, +0.031), is a post-hoc pick.

Error analysis of the best model, random forest (notebook 06):

- **It is an always-hungry classifier with small exceptions.** 89–100% of every other class's clips
  are predicted *hungry*. Only *tired* is ever right (2.0 of 28 clips per repeat).
- **No feature group matters.** Shuffling any group changes balanced accuracy by at most 0.006 ± 0.005 (MFCCs).
- **Confidence says little.** It separates right from wrong with AUROC 0.549, and the probabilities are
  under-confident (ECE 0.205).
- **Babies with any non-hungry clip** are predicted correctly 32.1% of the time (51 babies), and 18 of
  them are never right. Sex and age code make no visible difference.
- **Some sources are probably not one baby.** 19 of 226 carry more than one sex or age code. One has 15
  clips under 6 labels and accounts for 7 of the 12 most confident mistakes.
- **The recording format leaks into the probabilities.** The six 44.1 kHz clips of *tired*, *discomfort*
  and *burping* get more probability on the 44.1 kHz-only classes than their 8 kHz clips do: AUROC 0.81,
  or 0.78–0.87 when each of the 4 babies is left out. It never changes a prediction, so 05's scores are not
  inflated, but band-limiting did not remove it. The CNN in notebook 07 gets the same check.

CNN on log-mel spectrograms (notebook 07, run on a Colab T4):

- **Nothing it learns transfers to new babies.** Early stopping kept the network after its first epoch in
  most splits (median best epoch 0); after that the validation score only got worse.
- **No gain over the baselines.** With the square-root weighted loss, balanced accuracy is 0.150 ± 0.013
  (95% CI 0.113–0.206), +0.026 (−0.012, +0.073) vs majority. With the class-balanced sampler it is 0.136 ± 0.030.
  Both are within the noise floor of the random forest's 0.131, and log-loss (1.715, 2.230) is worse than the prior's 0.887.
- **Its one partly recognised class, *lonely*** (recall 0.400, precision 0.082), exists only as 44.1 kHz
  recordings. The format check gives AUROC 0.65–0.71, lower than the random forest's 0.81 but not zero.
- **A pitch-shift ablation** on one repeat changes balanced accuracy by +0.019 (−0.044, +0.123): no evidence either way.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # set BABYCRY_DATA_DIR to the folder with the 8 class folders
jupyter lab notebooks/
```

Run the notebooks in order. Notebook 01 writes `metadata/clean_index.csv`, which the later
notebooks read. Notebook 05 takes about 12 minutes on an 11-core laptop. Code shared by several notebooks lives in `src/`, and its tests use synthetic
signals only, so they run without the dataset:

```bash
pytest
```

`tests/test_cnn.py` needs PyTorch and is skipped without it. Notebook 07 runs them on Colab.

### Notebook 07 on Google Colab

1. `python scripts/make_colab_bundle.py` writes `data/colab/babycry_8k.zip` (the 479 clean clips at 8 kHz
   mono, about 43 MB). Upload it to **My Drive/babycry/**.
2. Open `notebooks/07_deep_learning.ipynb` in Colab, choose a T4 GPU runtime and run all cells. If the
   repository is private, first add a read-only GitHub token as the Colab secret `GITHUB_TOKEN`.
3. Every split is saved to Drive as it finishes, so after a disconnect, running all cells again resumes.
4. Download `My Drive/babycry/results/` (predictions and training logs) and the executed notebook into
   this repository.
