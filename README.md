# Infant Cry State Recognition

Predict what a baby's cry is about (one of 8 parent-reported states) from a ~7 s audio clip,
first with classical machine learning, then with deep learning. The project is built as a
sequence of notebooks, and every number in this README comes from running them.

| Notebook | Status |
|---|---|
| [`01_data_audit.ipynb`](notebooks/01_data_audit.ipynb): what the data really contains | done |
| `02_machine_learning.ipynb`: hand-crafted audio features + classical models | in progress |
| `03_deep_learning.ipynb`: CNN on spectrograms | planned |

No model results are reported yet.

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

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # set BABYCRY_DATA_DIR to the folder with the 8 class folders
jupyter lab notebooks/
```

Run the notebooks in order. Notebook 01 writes `metadata/clean_index.csv`, which the later
notebooks read.
