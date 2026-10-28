import numpy as np
import pytest
from sklearn import metrics as sk

from src.audio import CLASSES
from src.metrics import (balanced_accuracy, brier, cluster_bootstrap, encode, log_loss, macro_f1,
                         mcc, paired_bootstrap, per_class_table, scores, top_k_accuracy)

RNG = np.random.default_rng(0)


def test_hand_computed_example():
    # 4 hungry, 2 tired, 2 scared; tired never predicted
    t = encode(["hungry"] * 4 + ["tired"] * 2 + ["scared"] * 2)
    p = encode(["hungry"] * 4 + ["hungry"] * 2 + ["scared", "hungry"])
    assert balanced_accuracy(t, p) == pytest.approx((1 + 0 + 0.5) / 3)
    f1_hungry = 2 * 4 / (4 + 7)
    f1_scared = 2 * 1 / (2 + 1)
    assert macro_f1(t, p) == pytest.approx((f1_hungry + 0 + f1_scared) / 3)


def test_always_hungry_on_the_real_class_counts():
    counts = {"belly_pain": 16, "burping": 9, "cold_hot": 7, "discomfort": 21,
              "hungry": 382, "lonely": 11, "scared": 5, "tired": 28}
    t = encode([c for c, n in counts.items() for _ in range(n)])
    s = scores(t, np.full(len(t), CLASSES.index("hungry")))
    assert s["accuracy"] == pytest.approx(0.797, abs=5e-4)
    assert s["balanced_accuracy"] == pytest.approx(0.125)
    assert s["macro_f1"] == pytest.approx(0.111, abs=5e-4)


def test_agrees_with_sklearn_on_random_predictions():
    t, p = RNG.integers(0, 8, 500), RNG.integers(0, 8, 500)
    proba = RNG.dirichlet(np.ones(8), 500)
    assert balanced_accuracy(t, p) == pytest.approx(sk.balanced_accuracy_score(t, p))
    assert macro_f1(t, p) == pytest.approx(sk.f1_score(t, p, average="macro"))
    assert mcc(t, p) == pytest.approx(sk.matthews_corrcoef(t, p))
    assert log_loss(t, proba) == pytest.approx(sk.log_loss(t, proba, labels=range(8)))
    assert top_k_accuracy(t, proba, 3) == pytest.approx(sk.top_k_accuracy_score(t, proba, k=3, labels=range(8)))
    assert brier(t, proba) == pytest.approx(np.mean(np.sum((proba - np.eye(8)[t]) ** 2, axis=1)))


def test_classes_absent_from_truth_are_not_averaged_in():
    t = encode(["hungry", "tired"])
    assert balanced_accuracy(t, t) == 1 and macro_f1(t, t) == 1


def test_per_class_table_counts_babies_and_flags_small_classes():
    t = encode(["hungry"] * 6 + ["scared"] * 3)
    groups = ["a", "b", "c", "d", "e", "f", "g", "g", "h"]
    table = per_class_table(t, t, groups)
    assert table.loc["hungry", "babies"] == 6 and table.loc["hungry", "reliable"]
    assert table.loc["scared", "babies"] == 2 and not table.loc["scared", "reliable"]
    assert table.loc["scared", "f1"] == 1 and np.isnan(table.loc["tired", "f1"])


def test_cluster_bootstrap_resamples_babies_not_clips():
    t = encode(["hungry"] * 10 + ["tired"] * 10)
    groups = np.repeat(np.arange(4), 5)          # 4 babies with 5 clips each
    perfect = cluster_bootstrap(t, t, groups, n_boot=200)
    assert np.all(perfect == 1)
    # baby 0 is all wrong: a draw's score depends only on how often baby 0 was drawn
    p = t.copy()
    p[:5] = CLASSES.index("tired")
    values = cluster_bootstrap(t, p, groups, metric=lambda a, b: np.mean(a == b), n_boot=500)
    assert set(np.round(values, 6)) <= {round(1 - k / 4, 6) for k in range(5)}


def test_paired_bootstrap_of_identical_models_is_zero_and_averages_repeats():
    t = RNG.integers(0, 8, 100)
    groups = RNG.integers(0, 30, 100)
    p = RNG.integers(0, 8, 100)
    assert np.all(paired_bootstrap(t, p, p, groups, n_boot=100) == 0)
    two = cluster_bootstrap(t, [p, t], groups, n_boot=50, seed=3)
    one = cluster_bootstrap(t, p, groups, n_boot=50, seed=3)
    np.testing.assert_allclose(two, (one + 1) / 2)
