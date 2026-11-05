import pandas as pd


def load_data(path="data/H2.csv"):
    df = pd.read_csv(path)

    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].str.strip()

    df["Agent"] = df["Agent"].replace("NULL", "None")
    df["Company"] = df["Company"].replace("NULL", "None")

    return df


def clean_data(df):
    """Remove invalid rows using fixed rules."""
    before = len(df)

    # Bookings with zero guests are invalid
    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    df = df[guests > 0]

    df = df[df["ADR"] <= 1000]

    # These columns reveal the answer (data leakage)
    df = df.drop(columns=["ReservationStatus", "ReservationStatusDate"])

    # Fill missing values
    df["Children"] = df["Children"].fillna(0)
    df["Country"] = df["Country"].fillna("Unknown")

    print(f"Rows: {before} -> {len(df)} (removed {before - len(df)})")
    return df.reset_index(drop=True)