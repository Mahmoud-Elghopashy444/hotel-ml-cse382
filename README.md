# Hotel Booking ML – CSE382

CSE382 project on the City Hotel bookings (H2.csv):
- **Task A:** predict the average daily rate (ADR) at the moment of booking.
- **Task B:** predict whether a booking will be canceled (IsCanceled) at the moment of booking.
- **Task C:** find booking/customer segments with clustering (no outcome used).

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Put `H2.csv` in the `data/` folder (it is not uploaded to GitHub).

## Run
```
python main.py analyze   # data checks (types, missing, duplicates) + EDA plots -> outputs/
python main.py train     # Task A: regression baselines (time-series CV) -> outputs/tables/
python main.py classify  # Task B: classification baselines + confusion matrices
python main.py cluster   # Task C: K-Means k scan, cluster profiles, PCA plot
python -m src.cluster_silhouette   # Task C: silhouette value per cluster
```

## Tests
```
python -m pytest   # leakage, split and pipeline checks for Tasks A, B, C
```

## Project structure
```
main.py           command line entry point
src/data.py       load + clean the data
src/eda.py        data quality table + EDA plots
src/features.py   allowed features + leakage exclusions
src/split.py      temporal train/test split + TimeSeriesSplit CV
src/pipeline.py   preprocessing (impute, scale, one-hot)
src/models.py     baseline models
src/evaluate.py   CV metrics (MAE, RMSE, R2)
src/classification.py  Task B: features, leakage exclusions, models, metrics
src/clustering.py      Task C: features, preprocessing, k scan, profiles, plots
src/cluster_silhouette.py  Task C: per-cluster silhouette plot
tests/            pytest checks (leakage, temporal split, pipelines)
reports/          Milestone 1 progress report
```

## Notes
- Test set = arrivals from 2017-05-01. It is not used in Milestone 1.
- Results: `outputs/tables/baseline_results.csv` (A), `classification_baseline_results.csv` (B), `kmeans_*.csv` (C).
- Report: `reports/milestone1_progress_report.md`.

## Team
| Name | Student ID | GitHub | Task |
|---|---|---|---|
| Mahmoud Mohamed Abdelsadek | 2400669 | Mahmoud-Elghopashy444 | Task A, coordination and submission |
| Mostafa Ahmed Mohamed | 2300389 | mostafatata1808-droid | Task B |
| Hazem Mohamed Mahmoud | 2400318 | hazemmhmd78-prog | Task B |
| Youssef Mohamed Elkady | 2201500 | youssefmelkady | Task A |
| Karim Ashraf Abada | 2401337 | karim9116 | Task C |
| Mazen Ashraf Sobhy | 2400706 | mazenashraf547 | Task C |

Repo: https://github.com/Mahmoud-Elghopashy444/hotel-ml-cse382
Submitted commit: `52c3032da7f9af9f5dae028e8a6d774950ded107`