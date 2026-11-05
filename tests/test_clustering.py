"""Task C: no outcome in the clustering inputs, valid matrix, k choice rule."""
import numpy as np
import pandas as pd

from src import clustering as clu

OUTCOMES = {"IsCanceled", "ReservationStatus", "ReservationStatusDate",
            "AssignedRoomType", "BookingChanges", "RequiredCarParkingSpaces"}


def test_no_outcome_column_used():
    assert not set(clu.NUMERIC_FEATURES + clu.CATEGORICAL_FEATURES) & OUTCOMES


def test_preprocessed_matrix_has_no_nan(clean_df):
    df = clu.add_cluster_features(clean_df.sample(3000, random_state=42))
    X = clu.build_preprocessor().fit_transform(df)
    assert not np.isnan(X).any()


def test_pick_k_takes_first_local_peak():
    scan = pd.DataFrame({"k": [2, 3, 4, 5, 6], "silhouette": [0.1, 0.2, 0.15, 0.3, 0.25]})
    assert clu.pick_k(scan) == 3
