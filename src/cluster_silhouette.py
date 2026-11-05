"""Task C: per-cluster silhouette plot for the K-Means (k=5) baseline.

Run with: python -m src.cluster_silhouette
"""
import os

import numpy as np
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from src.clustering import SEED, add_cluster_features, build_preprocessor
from src.data import clean_data, load_data

K = 5
SAMPLE_SIZE = 10000


def plot_silhouette(X, labels, out_path="outputs/figures/kmeans_silhouette.png",
                    sample=SAMPLE_SIZE):
    """Plot silhouette values per cluster on a fixed sample and return their mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.random.RandomState(SEED).choice(len(X), size=min(sample, len(X)), replace=False)
    sample_labels = labels[idx]
    # Compute distances in smaller chunks: same values, without one 763 MB allocation.
    with config_context(working_memory=256):
        values = silhouette_samples(X[idx], sample_labels)
    mean_value = values.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    y_lower = 0
    for cluster in np.unique(sample_labels):
        cluster_values = np.sort(values[sample_labels == cluster])
        y_upper = y_lower + len(cluster_values)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, alpha=0.8)
        ax.text(-0.02, (y_lower + y_upper) / 2, str(cluster), ha="right", va="center")
        y_lower = y_upper + 100

    ax.axvline(mean_value, color="black", linestyle="--", label=f"Mean = {mean_value:.3f}")
    ax.set_xlabel("Silhouette value")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.set_title(f"K-Means silhouette per cluster (k={K}, {len(idx):,}-row sample)")
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return mean_value


def main():
    df = add_cluster_features(clean_data(load_data()))
    X = build_preprocessor().fit_transform(df)
    labels = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X)
    mean_value = plot_silhouette(X, labels)
    print(f"Mean silhouette on the sample: {mean_value:.3f}")
    print("Saved outputs/figures/kmeans_silhouette.png")


if __name__ == "__main__":
    main()
