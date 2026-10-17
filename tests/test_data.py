from src import config
from src.data import build_metadata, clean_metadata, train_test_split_df
from tests.conftest import N_PER_CLASS


def test_metadata_indexes_every_class(synthetic_dataset):
    df = build_metadata(synthetic_dataset)
    assert sorted(df["label"].unique()) == config.CLASSES
    assert len(df) == len(config.CLASSES) * N_PER_CLASS + 1


def test_cleaning_drops_too_short_clip(synthetic_dataset):
    df = clean_metadata(build_metadata(synthetic_dataset), verbose=False)
    assert "too_short.wav" not in set(df["filename"])
    assert len(df) == len(config.CLASSES) * N_PER_CLASS


def test_train_test_split_has_no_overlap(synthetic_dataset):
    df = clean_metadata(build_metadata(synthetic_dataset), verbose=False)
    train_df, test_df = train_test_split_df(df)
    train, test = set(train_df["filepath"]), set(test_df["filepath"])
    assert train.isdisjoint(test)
    assert train | test == set(df["filepath"])
    assert len(test) == round(config.TEST_SIZE * len(df))


def test_split_is_stratified_and_reproducible(synthetic_dataset):
    df = clean_metadata(build_metadata(synthetic_dataset), verbose=False)
    _, test_a = train_test_split_df(df)
    _, test_b = train_test_split_df(df.sample(frac=1, random_state=0).sort_values("filepath"))
    assert set(test_a["label"]) == set(config.CLASSES)    # every class in test
    assert list(test_a["filepath"]) == list(test_b["filepath"])
