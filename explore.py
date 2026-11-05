import os

import pandas as pd

from src.data import load_data, clean_data
from src.features import add_features, get_X_y
from src.split import temporal_split, get_cv
from src.models import get_models, make_pipeline
from src.evaluate import evaluate_cv

df = add_features(clean_data(load_data()))
train, test = temporal_split(df)
X_train, y_train = get_X_y(train)
cv = get_cv(len(train))

rows = []
for name, model in get_models().items():
    print("Training:", name, "...")
    scores = evaluate_cv(make_pipeline(model), X_train, y_train, cv)
    rows.append({"Model": name, **scores})

table = pd.DataFrame(rows).round(3)
print()
print(table.to_string(index=False))

os.makedirs("outputs/tables", exist_ok=True)
table.to_csv("outputs/tables/baseline_results.csv", index=False)
print("\nSaved outputs/tables/baseline_results.csv")