"""Task B: predict IsCanceled at the moment the booking is made."""
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "IsCanceled"

# Prediction point: the moment the booking is entered in the system.
NUMERIC_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "TotalNights",
    "Adults", "Children", "Babies", "TotalGuests",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "DaysInWaitingList", "TotalOfSpecialRequests", "ADR",
]

CATEGORICAL_FEATURES = [
    "ArrivalDateMonth", "Meal", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType", "Agent", "Company",
]

EXCLUDED = {
    "ReservationStatus": "Direct leakage: Canceled/No-Show/Check-Out is the answer itself.",
    "ReservationStatusDate": "Date of the final status; set after the outcome is known.",
    "AssignedRoomType": "Given at check-in. A different room means the guest arrived "
                        "(cancel rate 5.9% vs 45.3%).",
    "BookingChanges": "Counted until check-in or cancellation; final value is in the future.",
    "RequiredCarParkingSpaces": "Recorded at arrival: every booking with parking > 0 "
                                "has IsCanceled = 0.",
    "Country": "Indirect leakage: nationality is often confirmed only at check-in, so "
               "canceled bookings keep the default 'PRT' (PRT cancel rate 65% vs 27%).",
}


def get_X_y(df):
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=50, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def get_models():
    """Untuned (default) models for Milestone 1."""
    return {
        "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    }


def make_pipeline(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def evaluate_cv(pipe, X, y, cv):
    """Time-series CV. Returns mean metrics and the confusion matrix summed over folds."""
    scores, cm = [], np.zeros((2, 2), dtype=int)
    start = time.perf_counter()
    for train_idx, val_idx in cv.split(X):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_val = y.iloc[val_idx]
        pred = pipe.predict(X.iloc[val_idx])
        proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append({
            "Accuracy": accuracy_score(y_val, pred),
            "Precision": precision_score(y_val, pred, zero_division=0),
            "Recall": recall_score(y_val, pred),
            "F1": f1_score(y_val, pred),
            "ROC_AUC": roc_auc_score(y_val, proba),
            "PR_AUC": average_precision_score(y_val, proba),
        })
        cm += confusion_matrix(y_val, pred, labels=[0, 1])
    result = pd.DataFrame(scores).mean().to_dict()
    result["time_s"] = time.perf_counter() - start
    return result, cm


def save_confusion_matrices(matrices, out_dir="outputs/figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4))
    for ax, (name, cm) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(cm, display_labels=["Not canceled", "Canceled"]).plot(
            ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)
    fig.suptitle("IsCanceled – confusion matrices (validation folds, summed)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/classification_confusion_matrices.png", dpi=120)
    plt.close(fig)
