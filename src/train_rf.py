"""Cost-sensitive Random Forest baseline (port of Baby_Cry_Scene.ipynb cells 12-14).

Usage: python -m src.train_rf
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.data import encode_labels, load_clean_metadata, train_test_split_df
from src.features import build_feature_table


def drop_nan_rows(X, y, name):
    ok = ~np.isnan(X).any(axis=1)
    if (~ok).any():
        print(f"Warning: dropping {(~ok).sum()} {name} clips with NaN features")
    return X[ok], y[ok]


def main():
    df = load_clean_metadata()
    print(df["label"].value_counts().to_string())

    train_df, test_df = train_test_split_df(df)
    X_train, y_train = drop_nan_rows(build_feature_table(train_df),
                                     encode_labels(train_df["label"]), "train")
    X_test, y_test = drop_nan_rows(build_feature_table(test_df),
                                   encode_labels(test_df["label"]), "test")
    print(f"Train: {len(y_train)}  Test: {len(y_test)}  Features: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
    X_test_scaled = scaler.transform(X_test)

    present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present, y=y_train)
    class_weight_dict = dict(zip(present, weights))

    rf = RandomForestClassifier(n_estimators=300, class_weight=class_weight_dict,
                                min_samples_leaf=2, random_state=config.SEED, n_jobs=-1)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    cv = cross_val_score(rf, X_train_scaled, y_train, cv=skf, scoring="f1_macro")
    print(f"CV F1-Macro: {cv.mean():.3f} ± {cv.std():.3f}")

    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    labels = list(range(len(config.CLASSES)))
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    print(f"\nAccuracy: {acc * 100:.2f}%  F1 Macro: {macro_f1 * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=config.CLASSES, zero_division=0))

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "model": rf, "classes": config.CLASSES},
                config.MODELS_DIR / "rf.joblib")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "model": "random_forest",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "cv_macro_f1_mean": float(cv.mean()), "cv_macro_f1_std": float(cv.std()),
        "cv_macro_f1_folds": [float(s) for s in cv],
        "test_accuracy": float(acc), "test_macro_f1": float(macro_f1),
        "per_class_f1": dict(zip(config.CLASSES, map(float, f1_score(
            y_test, y_pred, labels=labels, average=None, zero_division=0)))),
    }
    with open(config.REPORTS_DIR / "rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model to {config.MODELS_DIR / 'rf.joblib'} and metrics to reports/")


if __name__ == "__main__":
    main()
