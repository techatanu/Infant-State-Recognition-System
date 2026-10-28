"""The fixed model menu and nested tuning used by notebook 05 (and reused by 06 and 08).

For every outer training part:
  1. each grid point is scored by grouped inner 3-fold CV: the inner out-of-fold probabilities are
     pooled and scored by balanced accuracy, as in the outer loop;
  2. the prior-correction decision rule  argmax_k  log p_k - tau * log prior_k  is tuned on the same
     pooled inner probabilities (tau = 0 is plain argmax);
  3. the best setting is refitted on the whole outer training part and predicts the outer test fold.
Two variants come out of one tuning run: "argmax" (best grid point with tau = 0) and
"prior_corrected" (best grid point and tau chosen together). The outer test fold is never used
for any choice.
"""
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .audio import CLASSES
from .metrics import balanced_accuracy, encode
from .splits import PROBA_COLS, inner_cv, outer_splits, repeats

TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]
WEIGHTS = [None, "balanced", "sqrt"]
VARIANTS = ["argmax", "prior_corrected"]


def class_weights(y, mode):
    """None, inverse frequency ('balanced') or its square root ('sqrt'); mean sample weight 1."""
    if mode is None:
        return None
    classes, counts = np.unique(y, return_counts=True)
    w = len(y) / (len(classes) * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
        w *= len(y) / np.sum(w * counts)
    return dict(zip(classes, w))


def _pipe(scale, estimator):
    steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else []) + [estimator]
    return make_pipeline(*steps)


def _grid(**axes):
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


# name -> (build(params, class_weight) -> unfitted estimator, grid). Fixed before any run.
MENU = {
    "logistic_regression": (
        lambda p, w: _pipe(True, LogisticRegression(C=p["C"], class_weight=w, max_iter=5000)),
        _grid(C=[0.001, 0.01, 0.1, 1.0], weight=WEIGHTS)),
    "knn": (
        lambda p, w: _pipe(True, KNeighborsClassifier(n_neighbors=p["k"], weights=p["weights"])),
        _grid(k=[5, 15, 31], weights=["uniform", "distance"])),
    # SVC(probability=True) is deprecated in scikit-learn 1.9 but still works (pinned version);
    # CalibratedClassifierCV, its replacement, fails when a class has fewer clips than its folds.
    "svm_rbf": (
        lambda p, w: _pipe(True, SVC(C=p["C"], gamma="scale", class_weight=w, probability=True, random_state=0)),
        _grid(C=[0.1, 1.0, 10.0], weight=WEIGHTS)),
    "random_forest": (
        lambda p, w: _pipe(False, RandomForestClassifier(n_estimators=300, max_depth=p["max_depth"],
                                                         min_samples_leaf=p["min_leaf"], class_weight=w,
                                                         n_jobs=1, random_state=0)),
        _grid(max_depth=[None, 8], min_leaf=[1, 5], weight=WEIGHTS)),
    "hist_gradient_boosting": (
        lambda p, w: _pipe(False, HistGradientBoostingClassifier(learning_rate=p["lr"], max_depth=p["max_depth"],
                                                                 random_state=0)),
        _grid(lr=[0.05, 0.1], max_depth=[3, None], weight=WEIGHTS)),
}
# HistGradientBoosting re-encodes labels internally, so its class weights are given as per-clip
# sample weights at fit time instead of a class_weight dict
SAMPLE_WEIGHTED = {"hist_gradient_boosting"}


def fit(name, params, X, y):
    """Fit one menu model with one grid point (and its class weighting) on X, y."""
    build, _ = MENU[name]
    weights = class_weights(y, params.get("weight"))
    fit_params = {}
    if name in SAMPLE_WEIGHTED:
        if weights is not None:
            fit_params[f"{build(params, None).steps[-1][0]}__sample_weight"] = np.array([weights[c] for c in y])
        weights = None
    model = build(params, weights)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")
        warnings.filterwarnings("ignore", message="The least populated class")
        model.fit(X, y, **fit_params)
    return model


def predict_proba(model, X):
    """Probabilities in CLASSES order (zero for classes absent from the training part)."""
    proba = np.zeros((len(X), len(CLASSES)))
    proba[:, [CLASSES.index(c) for c in model.classes_]] = model.predict_proba(X)
    return proba


def corrected(proba, prior, tau):
    """Prior-corrected class codes: argmax of log p - tau * log prior."""
    return np.argmax(np.log(np.clip(proba, 1e-12, 1)) - tau * np.log(prior), axis=1)


def class_prior(y):
    """Class shares of y in CLASSES order, as used by the prior-corrected rule."""
    counts = pd.Series(y).value_counts().reindex(CLASSES, fill_value=0).to_numpy()
    return np.maximum(counts, 0.5) / counts.sum()      # absent class: tiny prior, never divides by 0


def tune(name, X, y, groups, seed=0):
    """Choose (params, tau) on grouped inner CV of one training part. Returns a dict per variant."""
    _, grid = MENU[name]
    t, prior = encode(y), class_prior(y)
    with warnings.catch_warnings():     # a class with 2 clips cannot be in all 3 inner folds; expected
        warnings.filterwarnings("ignore", message="The least populated class")
        splits = list(inner_cv(seed).split(X, y, groups=groups))
    best = {v: (-np.inf, None, 0.0) for v in VARIANTS}
    for params in grid:
        proba = np.zeros((len(y), len(CLASSES)))
        for tr, va in splits:
            proba[va] = predict_proba(fit(name, params, X[tr], y[tr]), X[va])
        for tau in TAUS:
            score = balanced_accuracy(t, corrected(proba, prior, tau))
            for variant in VARIANTS:
                if (variant == "argmax" and tau > 0) or score <= best[variant][0]:
                    continue
                best[variant] = (score, params, tau)
    return {v: {"inner_ba": s, "params": p, "tau": tau} for v, (s, p, tau) in best.items()}


def _outer_split(name, X, y, groups, repeat, k, train, test, seed):
    # one native (OpenMP, BLAS) thread per worker, set inside the worker. Without it every worker
    # starts one thread per core, and gradient boosting did not finish in over an hour
    with threadpool_limits(1):
        return _outer_split_body(name, X, y, groups, repeat, k, train, test, seed)


def _outer_split_body(name, X, y, groups, repeat, k, train, test, seed):
    choice = tune(name, X[train], y[train], groups[train], seed)
    prior = class_prior(y[train])
    out, fitted = {}, {}
    for variant, c in choice.items():
        key = repr(c["params"])
        if key not in fitted:
            fitted[key] = predict_proba(fit(name, c["params"], X[train], y[train]), X[test])
        proba = fitted[key]
        part = pd.DataFrame(proba, columns=PROBA_COLS)
        part.insert(0, "y_pred", np.array(CLASSES)[corrected(proba, prior, c["tau"])])
        part.insert(0, "y_true", y[test])
        part.insert(0, "fold", k)
        part.insert(0, "repeat", repeat)
        part.insert(0, "clip", test)
        out[variant] = (part, {"repeat": repeat, "fold": k, **c})
    return out


def nested_oof(name, X, y, groups, folds, n_jobs=None, repeat_list=None):
    """Nested-CV out-of-fold predictions for both variants of one menu model.

    Returns ({variant: predictions table like splits.oof_predict}, tuning log DataFrame).
    """
    X, y, groups = np.asarray(X, dtype=float), np.asarray(y), np.asarray(groups)
    jobs = [(r, k, train, test) for r in (repeat_list or repeats(folds))
            for k, train, test in outer_splits(folds, r)]
    # processes, not threads: threads are held back by the GIL (gradient boosting ran 3x slower),
    # and libsvm's probability calibration draws from C's global rand(), which concurrent threads
    # share, so SVM results changed from run to run. The "multiprocessing" backend is used because
    # joblib's default (loky) cannot start in some sandboxes.
    results = Parallel(n_jobs=n_jobs or os.cpu_count(), backend="multiprocessing")(
        delayed(_outer_split)(name, X, y, groups, r, k, train, test, seed=int(r[1:]) * 10 + k)
        for r, k, train, test in jobs)
    predictions = {v: pd.concat([res[v][0] for res in results], ignore_index=True) for v in VARIANTS}
    log = pd.DataFrame([{"variant": v, **res[v][1]} for res in results for v in VARIANTS])
    return predictions, log
