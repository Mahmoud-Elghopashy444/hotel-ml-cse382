TARGET = "ADR"

NUMERIC_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "TotalNights",
    "Adults", "Children", "Babies", "TotalGuests",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "DaysInWaitingList", "TotalOfSpecialRequests",
]

CATEGORICAL_FEATURES = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType", "Agent", "Company",
]

EXCLUDED = {
    "IsCanceled": "Future outcome; unknown when the price is quoted.",
    "ReservationStatus": "Final outcome (Canceled / Check-Out / No-Show).",
    "ReservationStatusDate": "Date of the final outcome; only known later.",
    "AssignedRoomType": "Decided at check-in; the price follows the reserved room.",
    "BookingChanges": "Keeps counting after booking; final value unknown at booking time.",
    "RequiredCarParkingSpaces": "Recorded at arrival in practice; not known at booking time.",
}


def add_features(df):
    """New columns built from the same row only (safe: no info from other rows)."""
    df = df.copy()
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
    return df


def get_X_y(df):
    """Split the table into inputs (X) and target (y)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y