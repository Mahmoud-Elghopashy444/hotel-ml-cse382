import time
from sklearn.model_selection import cross_validate


def evaluate_cv(pipe, X, y, cv):
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    start = time.perf_counter()
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    return {
        "MAE": -res["test_MAE"].mean(),
        "RMSE": -res["test_RMSE"].mean(),
        "R2": res["test_R2"].mean(),
        "time_s": time.perf_counter() - start,
    }
