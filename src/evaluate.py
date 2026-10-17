"""Evaluate the RF and CNN on the same held-out test split.

Outputs (reports/):
  results.md, results.csv            RF vs CNN: accuracy, macro-F1 (with bootstrap 95% CI),
                                     CV macro-F1, per-class F1
  classification_report_{rf,cnn}.txt per-class precision / recall / F1
  test_predictions.csv               per-clip labels, predictions and confidences
  figures/confusion_matrices.png     row-normalised confusion matrices, side by side
  leakage_check.md                   RF CV with duplicates removed / grouped by recording source

Usage: python -m src.evaluate [--skip-leakage-check]
"""
import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402

from src import config  # noqa: E402
from src.data import (drop_cross_labelled_duplicates, encode_labels,  # noqa: E402
                      load_clean_metadata, recording_source, train_test_split_df)
from src.features import build_feature_table  # noqa: E402
from src.model import load_cnn, predict_proba  # noqa: E402
from src.train_cnn import load_waves  # noqa: E402

LABELS = list(range(len(config.CLASSES)))
N_BOOTSTRAP = 1000

# Sequential blue ramp (light -> dark) for magnitude in the confusion matrix
SEQ_BLUE = ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"


def predict_rf(test_df):
    bundle = joblib.load(config.MODELS_DIR / "rf.joblib")
    X = build_feature_table(test_df)
    assert not np.isnan(X).any(), "NaN features in test set"
    return bundle["model"].predict_proba(bundle["scaler"].transform(X))


def predict_cnn(test_df):
    model, _ = load_cnn()
    return predict_proba(model, load_waves(test_df["filepath"]))


def bootstrap_ci(y_true, y_pred, metric, seed=config.SEED):
    """95% percentile CI from resampling test clips with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [metric(y_true[idx], y_pred[idx])
              for idx in (rng.integers(0, n, n) for _ in range(N_BOOTSTRAP))]
    return np.percentile(scores, [2.5, 97.5])


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def summarise(name, y_true, proba, cv_metrics):
    y_pred = proba.argmax(1)
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=config.CLASSES,
                                   zero_division=0, digits=3)
    (config.REPORTS_DIR / f"classification_report_{name}.txt").write_text(report)
    print(f"\n=== {name.upper()} ===\n{report}")
    return {
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_ci": bootstrap_ci(y_true, y_pred, accuracy_score),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci": bootstrap_ci(y_true, y_pred, macro_f1),
        "per_class_f1": f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
        "cv": (cv_metrics["cv_macro_f1_mean"], cv_metrics["cv_macro_f1_std"]),
    }


def results_table(res, support):
    rows = [
        ("Test accuracy", *(f"{r['accuracy']:.1%} [{r['accuracy_ci'][0]:.0%}–"
                            f"{r['accuracy_ci'][1]:.0%}]" for r in res.values())),
        ("Test macro-F1", *(f"{r['macro_f1']:.1%} [{r['macro_f1_ci'][0]:.0%}–"
                            f"{r['macro_f1_ci'][1]:.0%}]" for r in res.values())),
        ("CV macro-F1 (5-fold, train split)",
         *(f"{r['cv'][0]:.3f} ± {r['cv'][1]:.3f}" for r in res.values())),
    ]
    for i, c in enumerate(config.CLASSES):
        rows.append((f"F1 {c} (n={support[i]})",
                     *(f"{r['per_class_f1'][i]:.2f}" for r in res.values())))
    return pd.DataFrame(rows, columns=["Metric", *[n.upper() for n in res]])


def plot_confusion_matrices(y_true, res, path):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(res), figsize=(6.4 * len(res), 5.8), facecolor=SURFACE,
                             layout="constrained")
    for ax, (name, r) in zip(axes, res.items()):
        counts = confusion_matrix(y_true, r["y_pred"], labels=LABELS)
        recall = counts / counts.sum(axis=1, keepdims=True)
        mesh = ax.pcolormesh(recall, cmap=cmap, vmin=0, vmax=1,
                             edgecolors=SURFACE, linewidth=2)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if counts[i, j]:
                    ax.text(j + 0.5, i + 0.5, counts[i, j], ha="center", va="center",
                            fontsize=10, color=SURFACE if recall[i, j] > 0.5 else INK)
        ax.set_xticks(np.arange(len(LABELS)) + 0.5, config.CLASSES, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(LABELS)) + 0.5,
                      [f"{c} ({n})" for c, n in zip(config.CLASSES, counts.sum(axis=1))])
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.tick_params(length=0, colors=INK, labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Predicted", color=INK_MUTED)
        if ax is axes[0]:
            ax.set_ylabel("True (test clips)", color=INK_MUTED)
        ax.set_title(f"{name.upper()}  ·  acc {r['accuracy']:.1%}  ·  macro-F1 {r['macro_f1']:.1%}",
                     color=INK, fontsize=12, loc="left")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.7)
    cbar.set_label("Share of true class (recall); numbers are clip counts", color=INK_MUTED)
    cbar.outline.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def leakage_check(seeds=(0, 1, 2)):
    """How much does the RF's CV score depend on (a) the 9 cross-labelled duplicates and
    (b) clips from the same recording source landing in both train and validation?"""
    df = load_clean_metadata()
    variants = {}
    dedup = drop_cross_labelled_duplicates(df)
    for name, d, grouped in [
        ("As-is: all clips, clip-level folds", df, False),
        ("Duplicates removed, clip-level folds", dedup, False),
        ("Duplicates removed, folds grouped by recording source", dedup, True),
    ]:
        X, y, groups = build_feature_table(d), encode_labels(d["label"]), recording_source(d)
        scores = []
        for seed in seeds:
            cv = (StratifiedGroupKFold if grouped else StratifiedKFold)(
                n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
            for tr, va in cv.split(X, y, groups if grouped else None):
                rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            min_samples_leaf=2, random_state=config.SEED,
                                            n_jobs=-1).fit(X[tr], y[tr])
                # macro over classes present in this fold (grouped folds can miss a class)
                scores.append(f1_score(y[va], rf.predict(X[va]), average="macro",
                                       zero_division=0))
        variants[name] = (len(d), groups.nunique(), np.mean(scores), np.std(scores))
        print(f"  {name}: macro-F1 {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    lines = ["| Setup | Clips | Sources | RF CV macro-F1 |", "|---|---|---|---|"]
    lines += [f"| {k} | {n} | {g} | {m:.3f} ± {s:.3f} |" for k, (n, g, m, s) in variants.items()]
    return "\n".join(lines) + (
        f"\n\n{config.N_FOLDS}-fold CV × {len(seeds)} seeds on all clean clips, RF with "
        "per-fold balanced class weights. Macro-F1 averages over the classes present in each "
        "validation fold.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-leakage-check", action="store_true")
    args = parser.parse_args()

    _, test_df = train_test_split_df(load_clean_metadata())
    y_test = encode_labels(test_df["label"])
    support = np.bincount(y_test, minlength=len(LABELS))
    print(f"Test clips: {len(test_df)}")

    metric_files = {"rf": "rf_metrics.json", "cnn": "cnn_cv_metrics.json"}
    cv = {m: json.loads((config.REPORTS_DIR / f).read_text()) for m, f in metric_files.items()}
    probas = {"rf": predict_rf(test_df), "cnn": predict_cnn(test_df)}
    res = {m: summarise(m, y_test, probas[m], cv[m]) for m in probas}

    table = results_table(res, support)
    table.to_csv(config.REPORTS_DIR / "results.csv", index=False)
    plot_confusion_matrices(y_test, res, config.FIGURES_DIR / "confusion_matrices.png")

    preds = pd.DataFrame({"filename": test_df["filename"], "label": test_df["label"]})
    for m, p in probas.items():
        preds[f"{m}_pred"] = [config.CLASSES[i] for i in p.argmax(1)]
        preds[f"{m}_confidence"] = p.max(1).round(3)
    preds.to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)

    md = ["# RF vs CNN on the held-out test split", "",
          f"Same stratified 80/20 split for both models ({len(test_df)} test clips). "
          f"Brackets: bootstrap 95% CI over test clips ({N_BOOTSTRAP} resamples).", "",
          table.to_markdown(index=False), "",
          "![Confusion matrices](figures/confusion_matrices.png)", ""]
    if not args.skip_leakage_check:
        print("\nLeakage check (RF):")
        leak = leakage_check()
        (config.REPORTS_DIR / "leakage_check.md").write_text("# Leakage check\n\n" + leak)
        md += ["## Leakage check (RF only)", "", leak]
    (config.REPORTS_DIR / "results.md").write_text("\n".join(md))
    print("\n" + table.to_markdown(index=False))
    print(f"\nWrote reports/results.md, results.csv, test_predictions.csv, "
          f"figures/confusion_matrices.png")


if __name__ == "__main__":
    main()
