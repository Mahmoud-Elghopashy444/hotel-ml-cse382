import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def add_arrival_date(df):
    """Combine year + month name + day into one real date column."""
    df = df.copy()
    df["ArrivalDate"] = pd.to_datetime(
        df["ArrivalDateYear"].astype(str) + "-"
        + df["ArrivalDateMonth"] + "-"
        + df["ArrivalDateDayOfMonth"].astype(str),
        format="%Y-%B-%d",
    )
    return df

TEST_START = "2017-05-01"  


def temporal_split(df):
    """Sort by arrival date, then cut: older bookings = train, newer = test."""
    df = add_arrival_date(df).sort_values("ArrivalDate").reset_index(drop=True)
    cutoff = pd.Timestamp(TEST_START)
    train = df[df["ArrivalDate"] < cutoff].reset_index(drop=True)
    test = df[df["ArrivalDate"] >= cutoff].reset_index(drop=True)
    return train, test

def get_cv(n_rows, n_splits=5):
    """Cross-validation that always validates on LATER data than it trains on.

    Each validation block is 10% of the training rows, so even the first fold
    trains on about one full year and has seen every season.
    """
    return TimeSeriesSplit(n_splits=n_splits, test_size=int(0.10 * n_rows))