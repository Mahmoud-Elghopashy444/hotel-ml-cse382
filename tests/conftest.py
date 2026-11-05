import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "H2.csv")


@pytest.fixture(scope="session")
def clean_df():
    if not os.path.exists(DATA):
        pytest.skip("data/H2.csv not found")
    from src.data import load_data, clean_data
    return clean_data(load_data(DATA))
