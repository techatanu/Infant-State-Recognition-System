"""Metrics, per-class tables and cluster-bootstrap intervals.

Conventions:
- Balanced accuracy and macro-F1 average over every class present in y_true (all 8 on the full
  data). A class that is never predicted counts with F1 = 0; it is never silently dropped.
- Clips of one baby are not independent, so intervals resample babies (cluster bootstrap).
- A score over several repeats is the mean of the per-repeat scores, each computed on that
  repeat's pooled out-of-fold predictions.
"""
import numpy as np
import pandas as pd

from .audio import CLASSES

K = len(CLASSES)
RELIABLE_BABIES = 5     # fewer babies than this: per-class numbers are "not reliably estimable"


def encode(labels):
    lookup = {c: i for i, c in enumerate(CLASSES)}
    return np.array([lookup[c] for c in labels])


def confusion(t, p):
    """K x K counts, rows = true class, columns = predicted class (integer codes)."""
    return np.bincount(np.asarray(t) * K + np.asarray(p), minlength=K * K).reshape(K, K)


def _from_confusion(cm):
    diag, true_n, pred_n = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    present = true_n > 0
    recall = diag[present] / true_n[present]
    f1 = 2 * diag[present] / (true_n[present] + pred_n[present])
    return recall.mean(), f1.mean()


def balanced_accuracy(t, p):
    return _from_confusion(confusion(t, p))[0]


def macro_f1(t, p):
    return _from_confusion(confusion(t, p))[1]


def mcc(t, p):
    """Multi-class Matthews correlation (Gorodkin's R_K), from the confusion matrix."""
    cm = confusion(t, p).astype(float)
    n, correct = cm.sum(), np.trace(cm)
    true_n, pred_n = cm.sum(axis=1), cm.sum(axis=0)
    num = correct * n - true_n @ pred_n
    den = np.sqrt((n ** 2 - pred_n @ pred_n) * (n ** 2 - true_n @ true_n))
    return num / den if den > 0 else 0.0


def log_loss(t, proba, eps=1e-15):
    proba = np.clip(np.asarray(proba, dtype=float), eps, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return -np.mean(np.log(proba[np.arange(len(t)), t]))


def brier(t, proba):
    """Multi-class Brier score: mean over clips of the squared distance to the one-hot truth."""
    onehot = np.eye(K)[t]
    return np.mean(np.sum((np.asarray(proba) - onehot) ** 2, axis=1))


def top_k_accuracy(t, proba, k=3):
    top = np.argsort(-np.asarray(proba), axis=1)[:, :k]
    return np.mean((top == np.asarray(t)[:, None]).any(axis=1))


def scores(t, p, proba=None):
    """All headline metrics for one set of predictions (integer codes; proba in CLASSES order)."""
    t, p = np.asarray(t), np.asarray(p)
    ba, f1 = _from_confusion(confusion(t, p))
    out = {"balanced_accuracy": ba, "macro_f1": f1, "accuracy": np.mean(t == p), "mcc": mcc(t, p)}
    if proba is not None:
        out.update({"log_loss": log_loss(t, proba), "brier": brier(t, proba),
                    "top3_accuracy": top_k_accuracy(t, proba, 3)})
    return out


def repeat_scores(predictions):
    """Metrics per repeat from an oof_predict() table: one row per repeat."""
    proba_cols = [f"p_{c}" for c in CLASSES]
    rows = {}
    for repeat, part in predictions.groupby("repeat", sort=True):
        rows[repeat] = scores(encode(part["y_true"]), encode(part["y_pred"]), part[proba_cols].to_numpy())
    return pd.DataFrame(rows).T


def summarize(predictions):
    """Mean and std over repeats of every metric."""
    per_repeat = repeat_scores(predictions)
    return pd.DataFrame({"mean": per_repeat.mean(), "std": per_repeat.std(ddof=1)})


def per_class_table(t, p, groups):
    """Precision, recall, F1 per class with clip and baby counts (integer codes)."""
    cm = confusion(t, p)
    diag, true_n, pred_n = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    groups = np.asarray(groups)
    babies = [len(np.unique(groups[np.asarray(t) == k])) for k in range(K)]
    with np.errstate(invalid="ignore", divide="ignore"):
        table = pd.DataFrame({
            "clips": true_n, "babies": babies, "predicted": pred_n,
            "precision": np.where(pred_n > 0, diag / pred_n, np.nan),
            "recall": np.where(true_n > 0, diag / true_n, np.nan),
            "f1": np.where(true_n + pred_n > 0, 2 * diag / (true_n + pred_n), np.nan),
        }, index=CLASSES)
    table["reliable"] = table["babies"] >= RELIABLE_BABIES
    return table


def _cluster_draws(groups, n_boot, seed):
    """For each bootstrap draw, the clip positions of babies resampled with replacement."""
    groups = np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.integers(0, len(members), len(members))
        yield np.concatenate([members[g] for g in chosen])


def _as_list(preds):
    """One array of predicted codes, or a list of them (one per repeat) -> list of arrays."""
    return [np.asarray(preds)] if np.ndim(preds[0]) == 0 else [np.asarray(q) for q in preds]


def cluster_bootstrap(t, preds, groups, metric=balanced_accuracy, n_boot=1000, seed=0):
    """Bootstrap distribution of a metric, resampling babies.

    preds is one array of predicted codes or a list of them (one per repeat, all aligned with t);
    with several, each draw scores every repeat on the same resampled babies and averages them.
    """
    t, preds = np.asarray(t), _as_list(preds)
    return np.array([np.mean([metric(t[i], q[i]) for q in preds])
                     for i in _cluster_draws(groups, n_boot, seed)])


def paired_bootstrap(t, preds_a, preds_b, groups, metric=balanced_accuracy, n_boot=1000, seed=0):
    """Bootstrap distribution of metric(A) - metric(B) on the same resampled babies."""
    t, a, b = np.asarray(t), _as_list(preds_a), _as_list(preds_b)
    return np.array([np.mean([metric(t[i], q[i]) for q in a]) - np.mean([metric(t[i], q[i]) for q in b])
                     for i in _cluster_draws(groups, n_boot, seed)])


def interval(samples, level=0.95):
    tail = (1 - level) / 2 * 100
    return np.percentile(samples, [tail, 100 - tail])


def predictions_by_repeat(predictions, n_clips):
    """Predicted codes as a list of arrays aligned with the clip order, one per repeat."""
    out = []
    for _, part in predictions.groupby("repeat", sort=True):
        codes = np.full(n_clips, -1)
        codes[part["clip"].to_numpy()] = encode(part["y_pred"])
        out.append(codes)
    return out
