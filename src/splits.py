"""The evaluation protocol: saved grouped folds, inner tuning splits, out-of-fold predictions.

Every model, classical or deep, is scored on the same 25 outer splits from metadata/folds.csv:
5 folds grouped by baby (source) and stratified by label, repeated with 5 seeds. In each repeat
every clip is predicted exactly once by a model that never saw its baby.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .audio import CLASSES, REPO

N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
INNER_FOLDS = 3
FOLDS_CSV = REPO / "metadata" / "folds.csv"
PREDICTIONS_DIR = REPO / "results" / "predictions"
PROBA_COLS = [f"p_{c}" for c in CLASSES]


def make_folds(index, n_folds=N_FOLDS, seeds=SEEDS, grouped=True):
    """Fold number (0..n_folds-1) of every clip, one column per repeat: r0, r1, ...

    grouped=False gives ordinary clip-level stratified folds; it exists only to demonstrate leakage.
    """
    folds = index[["path", "label", "source"]].copy()
    for r, seed in enumerate(seeds):
        if grouped:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            parts = cv.split(index, index["label"], groups=index["source"])
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            parts = cv.split(index, index["label"])
        fold = np.full(len(index), -1)
        for k, (_, test) in enumerate(parts):
            fold[test] = k
        folds[f"r{r}"] = fold
    return folds


def repeats(folds):
    return [c for c in folds.columns if c.startswith("r") and c[1:].isdigit()]


def load_folds(index=None):
    folds = pd.read_csv(FOLDS_CSV)
    if index is not None and folds["path"].tolist() != index["path"].tolist():
        raise ValueError("metadata/folds.csv does not match the clean index; rebuild it in notebook 04")
    return folds


def outer_splits(folds, repeat):
    """(fold, train_idx, test_idx) for each fold of one repeat (positions in the index)."""
    fold = folds[repeat].to_numpy()
    for k in range(fold.max() + 1):
        yield k, np.flatnonzero(fold != k), np.flatnonzero(fold == k)


def inner_cv(seed=0):
    """Grouped, stratified 3-fold splitter for tuning inside an outer training part.

    Pass groups=<source of the training clips> to its split() (or to GridSearchCV.fit)."""
    return StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)


def oof_predict(make_model, X, y, folds, groups=None, fit_groups=False, repeat_list=None):
    """Out-of-fold predictions of a fresh model per outer split.

    make_model() must return an unfitted sklearn-style classifier with predict_proba; y_pred
    comes from its predict() (so a model may use a decision rule other than argmax). With
    fit_groups=True the training clips' groups are passed to fit(), e.g. for a GridSearchCV that
    uses inner_cv(). Returns one row per clip and repeat, with class probabilities in CLASSES order.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = None if groups is None else np.asarray(groups)
    rows = []
    for repeat in repeat_list or repeats(folds):
        for k, train, test in outer_splits(folds, repeat):
            model = make_model()
            kwargs = {"groups": groups[train]} if fit_groups else {}
            model.fit(X[train], y[train], **kwargs)
            proba = np.zeros((len(test), len(CLASSES)))
            proba[:, [CLASSES.index(c) for c in model.classes_]] = model.predict_proba(X[test])
            part = pd.DataFrame(proba, columns=PROBA_COLS)
            part.insert(0, "y_pred", model.predict(X[test]))
            part.insert(0, "y_true", y[test])
            part.insert(0, "fold", k)
            part.insert(0, "repeat", repeat)
            part.insert(0, "clip", test)
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def save_predictions(name, predictions, index):
    """Write out-of-fold predictions to results/predictions/<name>.csv (keyed by clip path)."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = predictions.assign(path=index["path"].to_numpy()[predictions["clip"]]).drop(columns="clip")
    # 6 significant digits, not fixed decimals: rounding a tiny probability to 0 would inflate log-loss
    out.to_csv(PREDICTIONS_DIR / f"{name}.csv", index=False, float_format="%.6g")


def load_predictions(name, index):
    out = pd.read_csv(PREDICTIONS_DIR / f"{name}.csv")
    position = pd.Series(np.arange(len(index)), index=index["path"])
    return out.assign(clip=position.loc[out["path"]].to_numpy()).drop(columns="path")
