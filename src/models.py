from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.pipeline import build_preprocessor

SEED = 42  


def get_models():
    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),  
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=SEED, n_jobs=-1),
    }


def make_pipeline(model):
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", model),
    ])