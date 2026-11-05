"""Task A: leakage and split checks."""
from src.features import (CATEGORICAL_FEATURES, EXCLUDED, NUMERIC_FEATURES,
                          add_features, get_X_y)
from src.split import get_cv, temporal_split


def test_no_excluded_column_is_a_feature():
    features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert not features & set(EXCLUDED)
    assert "ADR" not in features


def test_train_is_before_test(clean_df):
    train, test = temporal_split(add_features(clean_df))
    assert train["ArrivalDate"].max() < test["ArrivalDate"].min()
    assert len(train) + len(test) == len(clean_df)


def test_cv_validates_on_later_rows(clean_df):
    train, _ = temporal_split(add_features(clean_df))
    for tr_idx, val_idx in get_cv(len(train)).split(train):
        assert tr_idx.max() < val_idx.min()


def test_X_has_expected_columns(clean_df):
    X, y = get_X_y(add_features(clean_df))
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert y.name == "ADR"
