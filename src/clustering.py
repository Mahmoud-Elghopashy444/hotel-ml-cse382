"""Task C: find booking segments without using any outcome column."""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42

# Booking attributes that describe "what kind of booking is this" at booking time.
NUMERIC_FEATURES = [
    "LeadTime", "TotalNights", "WeekendShare", "Adults", "Kids",
    "ADR", "TotalOfSpecialRequests", "IsRepeatedGuest",
]
# Skewed counts/amounts: log1p before scaling so a few huge values do not dominate distances.
LOG_FEATURES = ["LeadTime", "TotalNights", "ADR"]

CATEGORICAL_FEATURES = ["MarketSegment", "CustomerType", "DepositType", "Meal"]

EXCLUDED = {
    "IsCanceled": "Outcome; clusters would just split canceled vs not canceled.",
    "ReservationStatus / ReservationStatusDate": "Final outcome of the booking.",
    "AssignedRoomType, BookingChanges, RequiredCarParkingSpaces":
        "Filled in later in the lifecycle, so they encode the outcome indirectly.",
    "Country": "Often only confirmed at check-in (proxy for not canceled); also very high cardinality.",
    "Agent, Company": "IDs with hundreds of values; one-hot would dominate the distance.",
    "Arrival date columns": "We want customer types, not calendar groups.",
}


def add_cluster_features(df):
    df = df.copy()
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    df["WeekendShare"] = (df["StaysInWeekendNights"] / df["TotalNights"]).fillna(0)
    df["Kids"] = df["Children"] + df["Babies"]
    return df


def build_preprocessor():
    numeric = ColumnTransformer([
        ("log", FunctionTransformer(np.log1p), LOG_FEATURES),
    ], remainder="passthrough")
    numeric = Pipeline([("log", numeric), ("scale", StandardScaler())])
    # Categories stay 0/1 (not scaled), so each category adds a fixed distance.
    categorical = OneHotEncoder(min_frequency=200, handle_unknown="infrequent_if_exist",
                                sparse_output=False)
    order = LOG_FEATURES + [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]
    return ColumnTransformer([
        ("num", numeric, order),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def scan_k(X, k_values=range(2, 11), sample=10000):
    """Inertia (elbow), Silhouette and Davies-Bouldin for each k. No labels used."""
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X[idx], km.labels_[idx]),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def pick_k(scan):
    """First k where Silhouette is higher than both neighbours.

    Silhouette keeps rising slowly for large k (many tiny clusters), so the
    global maximum sits at the edge of the range and is not a real choice.
    The first local peak gives the smallest k with a clear improvement.
    """
    s = scan["silhouette"].to_numpy()
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1]:
            return int(scan["k"].iloc[i])
    return int(scan.loc[scan["silhouette"].idxmax(), "k"])


def profile_clusters(df, labels):
    """Describe each cluster with readable booking characteristics."""
    d = df.assign(Cluster=labels)
    prof = d.groupby("Cluster").agg(
        size=("ADR", "size"),
        lead_time=("LeadTime", "median"),
        nights=("TotalNights", "median"),
        adr=("ADR", "median"),
        adults=("Adults", "mean"),
        kids=("Kids", "mean"),
        special_requests=("TotalOfSpecialRequests", "mean"),
        repeated_guest=("IsRepeatedGuest", "mean"),
    )
    for col in ["MarketSegment", "CustomerType", "DepositType"]:
        prof[col] = d.groupby("Cluster")[col].agg(lambda s: s.mode()[0])
    prof["share"] = prof["size"] / len(d)
    # Only for interpretation AFTER clustering; never used to fit or choose k.
    prof["cancel_rate_post_hoc"] = d.groupby("Cluster")["IsCanceled"].mean()
    return prof.round(3)


def save_plots(X, labels, scan, out_dir="outputs/figures", sample=8000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, col, title in zip(axes, ["inertia", "silhouette", "davies_bouldin"],
                              ["Inertia (elbow)", "Silhouette (higher = better)",
                               "Davies-Bouldin (lower = better)"]):
        ax.plot(scan["k"], scan[col], marker="o")
        ax.set_xlabel("k")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/kmeans_k_scan.png", dpi=120)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    idx = np.random.RandomState(SEED).choice(len(X), size=min(sample, len(X)), replace=False)
    Z = pca.transform(X[idx])
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels[idx], cmap="tab10", s=6, alpha=0.6)
    plt.legend(*sc.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ev = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({ev[0]:.0%} variance)")
    plt.ylabel(f"PC2 ({ev[1]:.0%} variance)")
    plt.title("K-Means clusters in PCA 2-D space (8,000-row sample)")
    plt.savefig(f"{out_dir}/kmeans_pca_2d.png", dpi=120, bbox_inches="tight")
    plt.close()
