"""Task B: leakage and pipeline checks."""
from src import classification as clf
from src.features import add_features


def test_no_leaky_column_is_a_feature():
    features = set(clf.NUMERIC_FEATURES + clf.CATEGORICAL_FEATURES)
    assert not features & set(clf.EXCLUDED)
    assert clf.TARGET not in features


def test_pipeline_gives_probabilities(clean_df):
    df = add_features(clean_df).sample(3000, random_state=42)
    X, y = clf.get_X_y(df)
    pipe = clf.make_pipeline(clf.get_models()["Logistic Regression"]).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
