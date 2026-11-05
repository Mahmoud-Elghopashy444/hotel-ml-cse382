import os

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def plot_adr(df, out_dir="outputs/figures"):
    """Save the main ADR plots as PNG files."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    sns.histplot(df["ADR"], bins=80)
    plt.title(f"ADR distribution (skew = {df['ADR'].skew():.2f})")
    plt.savefig(f"{out_dir}/adr_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x="ArrivalDateMonth", y="ADR", order=MONTHS, showfliers=False)
    plt.xticks(rotation=45)
    plt.title("ADR by arrival month")
    plt.savefig(f"{out_dir}/adr_by_month.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="ReservedRoomType", y="ADR", showfliers=False,
                order=sorted(df["ReservedRoomType"].unique()))
    plt.title("ADR by reserved room type")
    plt.savefig(f"{out_dir}/adr_by_room.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("Mean ADR per year:")
    print(df.groupby("ArrivalDateYear")["ADR"].mean().round(1))

    print(f"Saved plots to {out_dir}/")

def data_quality(df, out_dir="outputs"):
    """Table of type, missing values and unique values per column + missing plot."""
    import pandas as pd

    os.makedirs(f"{out_dir}/tables", exist_ok=True)
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    table = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(3),
        "unique": df.nunique(),
    })
    table.to_csv(f"{out_dir}/tables/data_quality.csv", index_label="column")

    missing = table[table["missing"] > 0]["missing"]
    # Agent/Company use the text "NULL" (= no agent/company), so show them too.
    for col in ["Agent", "Company"]:
        missing[f"{col} (= 'NULL' text)"] = (df[col] == "None").sum()
    plt.figure(figsize=(7, 3.5))
    ax = missing.sort_values().plot.barh()
    ax.bar_label(ax.containers[0], padding=3)
    ax.set_xscale("log")
    plt.xlabel("rows (log scale)")
    plt.title("Missing values per column (raw H2.csv)")
    plt.savefig(f"{out_dir}/figures/missing_values.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Saved {out_dir}/tables/data_quality.csv and {out_dir}/figures/missing_values.png")
    return table
