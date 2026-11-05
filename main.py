import argparse
import os

import pandas as pd

from src.data import load_data, clean_data
from src.eda import data_quality, plot_adr
from src.evaluate import evaluate_cv
from src.features import add_features, get_X_y
from src.models import get_models, make_pipeline
from src.split import temporal_split, get_cv
from src import classification as clf
from src import clustering as clu


def analyze():
    df = load_data()
    print("Raw shape:", df.shape)
    print("Duplicates:", df.duplicated().sum())
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    data_quality(df)
    plot_adr(clean_data(df))


def train():
    df = add_features(clean_data(load_data()))
    train_df, test_df = temporal_split(df)
    print(f"Train: {len(train_df)} rows | Test (held out, not used): {len(test_df)} rows")
    X_train, y_train = get_X_y(train_df)
    cv = get_cv(len(train_df))

    rows = []
    for name, model in get_models().items():
        print("Training:", name, "...")
        rows.append({"Model": name, **evaluate_cv(make_pipeline(model), X_train, y_train, cv)})

    table = pd.DataFrame(rows).round(3)
    print("\n" + table.to_string(index=False))
    os.makedirs("outputs/tables", exist_ok=True)
    table.to_csv("outputs/tables/baseline_results.csv", index=False)
    print("\nSaved outputs/tables/baseline_results.csv")


def classify():
    """Task B: IsCanceled baselines, same temporal split and CV as Task A."""
    df = add_features(clean_data(load_data()))
    train_df, test_df = temporal_split(df)
    print(f"Train: {len(train_df)} rows | Test (held out, not used): {len(test_df)} rows")
    print(f"Cancel rate in train: {train_df['IsCanceled'].mean():.3f}")
    X_train, y_train = clf.get_X_y(train_df)
    cv = get_cv(len(train_df))

    rows, matrices = [], {}
    for name, model in clf.get_models().items():
        print("Training:", name, "...")
        scores, cm = clf.evaluate_cv(clf.make_pipeline(model), X_train, y_train, cv)
        rows.append({"Model": name, **scores})
        matrices[name] = cm

    table = pd.DataFrame(rows).round(3)
    print("\n" + table.to_string(index=False))
    os.makedirs("outputs/tables", exist_ok=True)
    table.to_csv("outputs/tables/classification_baseline_results.csv", index=False)
    clf.save_confusion_matrices(matrices)
    print("\nSaved outputs/tables/classification_baseline_results.csv")
    print("Saved outputs/figures/classification_confusion_matrices.png")


def cluster():
    """Task C: K-Means baseline on the full cleaned data (no outcome columns)."""
    df = clu.add_cluster_features(clean_data(load_data()))
    X = clu.build_preprocessor().fit_transform(df)
    print("Clustering matrix:", X.shape)

    print("Scanning k = 2..10 (Silhouette on a 10,000-row sample) ...")
    scan = clu.scan_k(X)
    print(scan.round(3).to_string(index=False))

    k = clu.pick_k(scan)
    print(f"Preliminary k (first local maximum of Silhouette): {k}")
    labels = clu.KMeans(n_clusters=k, n_init=10, random_state=clu.SEED).fit_predict(X)

    profile = clu.profile_clusters(df, labels)
    print("\n" + profile.to_string())

    os.makedirs("outputs/tables", exist_ok=True)
    scan.round(4).to_csv("outputs/tables/kmeans_k_scan.csv", index=False)
    profile.to_csv("outputs/tables/kmeans_cluster_profiles.csv")
    clu.save_plots(X, labels, scan)
    print("\nSaved outputs/tables/kmeans_*.csv and outputs/figures/kmeans_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSE382 hotel booking ML project")
    parser.add_argument("command", choices=["analyze", "train", "classify", "cluster"],
                        help="train = Task A regression, classify = Task B, cluster = Task C")
    args = parser.parse_args()

    commands = {"analyze": analyze, "train": train, "classify": classify, "cluster": cluster}
    commands[args.command]()