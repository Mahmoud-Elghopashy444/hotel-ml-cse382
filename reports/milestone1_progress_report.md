# CSE382 – Milestone 1 Progress Report

**Repository:** https://github.com/Mahmoud-Elghopashy444/hotel-ml-cse382
**Submitted commit:** `<added after all members push their files>`

## 1. Team members and contributions

| Member | Student ID | GitHub account | Contribution in Milestone 1 |
|---|---|---|---|
| Mahmoud Mohamed Abdelsadek | 2400669 | Mahmoud-Elghopashy444 | Task A (ADR regression, together): data loading/cleaning, EDA, leakage audit, temporal split + CV, preprocessing pipeline, baseline models, CLI; team coordination and submission |
| Mostafa Ahmed Mohamed | 2300389 | mostafatata1808-droid | Task B (IsCanceled classification, together): prediction point, leakage audit (Country, parking, AssignedRoomType...), baseline classifiers + metrics/confusion matrices (src/classification.py, `classify` command) |
| Hazem Mohamed Mahmoud | 2400318 | hazemmhmd78-prog | Task B (IsCanceled classification, together): prediction point, leakage audit (Country, parking, AssignedRoomType...), baseline classifiers + metrics/confusion matrices (src/classification.py, `classify` command) |
| Youssef Mohamed Elkady | 2201500 | youssefmelkady | Task A (ADR regression, together): data loading/cleaning, EDA, leakage audit, temporal split + CV, preprocessing pipeline, baseline models, CLI |
| Karim Ashraf Abada | 2401337 | karim9116 | Task C (clustering, together): feature choice + leakage audit, preprocessing, K-Means k scan (Silhouette/Davies-Bouldin), PCA plot, cluster profiles (src/clustering.py, `cluster` command) |
| Mazen Ashraf Sobhy | 2400706 | mazenashraf547 | Task C (clustering, together): feature choice + leakage audit, preprocessing, K-Means k scan (Silhouette/Davies-Bouldin), PCA plot, cluster profiles (src/clustering.py, `cluster` command) |

## 2. Data verification (H2.csv, City Hotel)

- Raw data: 79,330 bookings × 31 columns. Text values had padding spaces, which we strip on load.
- Missing values: `Children` 4, `Country` 24. `Agent` (8,131) and `Company` (75,641) use the text `NULL`, meaning "no agent/company", so we treat it as its own category (`None`), not as missing.
- Duplicates: 25,902 rows are exact duplicates. We keep them, because the data has no booking ID and identical rows are most likely several rooms in the same group booking.
- Unusual values removed: 1 booking with ADR = 5,400 (next highest is 510, clearly an entry error) and 167 bookings with zero guests.
- ADR = 0 (1,208 bookings, mostly Complementary) is kept: it is a real price policy.
- Final cleaned data: **79,162 rows**. The same cleaned data is used by all three tasks.

## 3. EDA

Figures are in `outputs/figures/`, tables in `outputs/tables/`:
- `data_quality.csv`: type, missing count and number of unique values for every column.
- `missing_values.png`: Children (4) and Country (24) are the only true missing values; Agent and Company use the text `NULL`.
- `adr_distribution.png`: ADR is right-skewed (median ≈ 100, mean ≈ 105).
- `adr_by_month.png`: strong seasonality, summer months are the most expensive.
- `adr_by_room.png`: the reserved room type changes the price a lot.
- Mean ADR rises every year: 86.0 (2015) → 103.5 (2016) → 117.7 (2017). This trend is why we split by time.
- IsCanceled: 41.8% of bookings are canceled, so the classes are only mildly imbalanced.

## 4. Split and validation strategy (Tasks A and B)

- **Test set:** all arrivals from **2017-05-01** to 2017-08-31 (15,191 rows, 19%). It is kept untouched until the final evaluation in Milestone 2.
- **Train set:** arrivals 2015-07-01 to 2017-04-30 (63,971 rows).
- **Cross-validation:** `TimeSeriesSplit` with 5 folds, each validation fold = 10% of train, so the model always trains on the past and validates on the future.
- Task C (clustering) has no target and no test set: it is fitted on all 79,162 cleaned rows, as the specification recommends for K-Means.

## 5. Preprocessing pipelines

All preprocessing lives inside a scikit-learn `Pipeline`, so in CV it is fitted on the training folds only.

- **Tasks A and B** (`src/pipeline.py`, `src/classification.py`): numeric → median imputation + `StandardScaler`; categorical → constant `"Unknown"` imputation + `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)`, so rare agents/countries are grouped and new categories in the future do not break the model.
- **Task C** (`src/clustering.py`): `log1p` on skewed `LeadTime`, `TotalNights`, `ADR`, then `StandardScaler` on all numeric columns (K-Means uses Euclidean distance, so unscaled LeadTime would dominate). Categorical columns are one-hot encoded and left as 0/1 so each category adds a fixed distance.

Engineered features (computed from the same row only, so no information from other rows leaks in): `TotalNights`, `TotalGuests`, and for clustering `WeekendShare` and `Kids`.

## 6. Prediction points and preliminary leakage audit

### 6.1 Task A – ADR regression
**Prediction point:** the moment the booking is made (price quote).

| Excluded column | Reason |
|---|---|
| IsCanceled | Known only after the booking ends |
| ReservationStatus, ReservationStatusDate | Final outcome of the booking (future) |
| AssignedRoomType | Decided at check-in, not at booking |
| BookingChanges | Counted after the booking is made |
| RequiredCarParkingSpaces | Usually filled in at arrival |

Used: 26 features (16 numeric, 10 categorical). Risk to re-check in Milestone 2: `Country` (see 6.2).

### 6.2 Task B – IsCanceled classification
**Prediction point:** the moment the booking is entered in the system. The hotel wants to know right away which bookings are risky (to overbook or ask for a deposit). Any column that is filled in or changes later is not allowed.

| Excluded column | Type of leakage | Evidence / reason |
|---|---|---|
| ReservationStatus | Direct | "Canceled"/"No-Show"/"Check-Out" is the answer itself (1:1 with IsCanceled). |
| ReservationStatusDate | Direct | Date the final status was set, only known after the outcome. |
| AssignedRoomType | Indirect | Assigned at check-in. A room different from the reserved one means the guest arrived: cancel rate 5.9% vs 45.3%. |
| RequiredCarParkingSpaces | Indirect | Recorded at arrival: every booking with parking > 0 has IsCanceled = 0 (0.0% vs 42.8%). |
| BookingChanges | Indirect | Keeps counting until check-in/cancellation; final value is in the future (16.5% vs 45.5% cancel rate). |
| Country | Indirect (subtle) | The dataset paper notes nationality is often confirmed only at check-in, so canceled bookings keep the default "PRT": cancel rate 65.0% for PRT vs 26.9% for others. Adding Country raises Random Forest CV ROC-AUC from 0.847 to **0.894**, which is too good to be explained by booking-time information. |

Kept but flagged:
- `DepositType`: known at booking, so it is allowed. "Non Refund" bookings are 99.8% canceled, which is odd (probably a data-recording artefact), so we will check in Milestone 2 how much the model depends on it.
- `ADR`: the agreed price is known at booking. The paper computes it from lodging transactions, so we will compare results with and without it in Milestone 2.
- `DaysInWaitingList`, `TotalOfSpecialRequests`: set before confirmation / at booking in most cases.

Used: 26 features (17 numeric incl. ADR, 9 categorical).

### 6.3 Task C – Clustering
**Analysis point:** describe bookings as they look when they are made. Clustering must be unsupervised, so no outcome may go in.

| Excluded | Reason |
|---|---|
| IsCanceled, ReservationStatus, ReservationStatusDate | Final outcome: clusters would simply become "canceled vs not canceled". |
| AssignedRoomType, BookingChanges, RequiredCarParkingSpaces | Filled in later and strongly tied to the outcome (see 6.2). |
| Country | Proxy for the outcome (see 6.2) and 160+ categories that would dominate distances. |
| Agent, Company | IDs with hundreds of values; one-hot would dominate the distance. |
| Arrival date columns | We want customer/booking types, not calendar groups. |

Used: `LeadTime`, `TotalNights`, `WeekendShare`, `Adults`, `Kids`, `ADR`, `TotalOfSpecialRequests`, `IsRepeatedGuest`, `MarketSegment`, `CustomerType`, `DepositType`, `Meal`. IsCanceled is used **only after** clustering, to describe the clusters, never to fit them or choose k.

### 6.4 How the steps are kept separate
- `tests/` (run with `python -m pytest`) checks automatically that no excluded column is used as a feature in any task, that every training date is before the test period, and that each CV fold validates on later rows than it trains on.
- Imputation, scaling and encoding are inside the pipeline, so they are re-fitted inside each CV fold.
- The test set (arrivals from 2017-05-01) has not been used for any choice. In Milestone 2, tuning will use only the CV on train, and the test set will be used once at the end.
- PCA/LDA (Milestone 2) will be added as pipeline steps, so they are also fitted on training folds only.

## 7. Initial results (untuned)

### 7.1 Task A – Regression (5-fold time-series CV)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (mean) | 27.52 | 37.95 | -0.164 |
| Linear Regression | **16.39** | **22.84** | **0.560** |
| Random Forest (default) | 16.85 | 24.62 | 0.494 |

Both real models beat the baseline by a large margin. Linear Regression is slightly better than the untuned Random Forest because prices rise every year and a tree cannot predict values higher than what it saw in training.

### 7.2 Task B – Classification (5-fold time-series CV, `python main.py classify`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy (majority) | 0.585 | 0.000 | 0.000 | 0.000 | 0.500 | 0.415 |
| Logistic Regression | **0.780** | 0.844 | **0.594** | **0.687** | 0.839 | **0.836** |
| Random Forest (default) | 0.777 | **0.863** | 0.558 | 0.672 | **0.847** | 0.831 |

Confusion matrices (validation folds summed): `outputs/figures/classification_confusion_matrices.png`.

**Interpretation:** both models clearly beat the majority baseline (ROC-AUC 0.84–0.85 vs 0.50). Precision is high but recall is low: the models miss about 40–44% of cancellations, because with the default 0.5 threshold they only flag bookings that look very risky. In Milestone 2 we will tune the models and the threshold (F1 or PR-AUC as the selection metric). These scores are honest: with the leaky columns from 6.2 the numbers would look better but would not be achievable at booking time.

### 7.3 Task C – K-Means clustering (`python main.py cluster`)

We scanned k = 2..10 on all 79,162 rows (Silhouette computed on a fixed 10,000-row sample for speed):

| k | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Silhouette | 0.123 | 0.149 | 0.150 | **0.158** | 0.151 | 0.142 | 0.155 | 0.165 | 0.174 |
| Davies–Bouldin | 2.568 | 1.911 | 1.832 | **1.660** | 1.572 | 1.639 | 1.554 | 1.462 | 1.489 |

Preliminary choice: **k = 5**, the first local peak of Silhouette with a big drop in Davies–Bouldin. Silhouette keeps rising slowly after k = 8 because K-Means starts splitting into many small clusters; that is not a natural structure. Silhouette values around 0.15 show the segments overlap (booking data is continuous, not separated groups). Plots: `outputs/figures/kmeans_k_scan.png`, `outputs/figures/kmeans_pca_2d.png`, and `outputs/figures/kmeans_silhouette.png` (silhouette value of every booking per cluster: mean 0.158; cluster 0 has the most negative values, so it overlaps most with its neighbours).

| Cluster | Share | Median lead time | Median nights | Median ADR | Typical segment | Description | Cancel rate (post hoc) |
|---|---|---|---|---|---|---|---|
| 0 | 18% | 6 days | 1 | 96 | Online TA | Last-minute short stays | 21% |
| 1 | 38% | 150 days | 3 | 90 | Offline TA/TO | Early bookings through tour operators, few requests | 60% |
| 2 | 35% | 75 days | 4 | 113 | Online TA | Online couples, longer stays, ~1 special request | 36% |
| 3 | 7% | 64 days | 3 | 147 | Online TA | Families with children, most expensive | 35% |
| 4 | 2% | 4 days | 1 | 66 | Corporate | Repeated (business) guests, cheap corporate rate | 22% |

**Interpretation:** the clusters are meaningful business segments. Even though IsCanceled was not used, cancellation rates differ a lot between them (21% to 60%), which shows that booking behaviour at booking time carries real information about risk.

## 8. Plan for Milestone 2

- **Regression:** add Gradient Boosting / XGBoost; tune all models with `RandomizedSearchCV` on the same `TimeSeriesSplit`; try a log target; re-check `Country`.
- **Classification:** add a third model family (XGBoost / SVM / Naive Bayes); tune all with `RandomizedSearchCV` (metric: F1 or PR-AUC); tune the decision threshold on CV only; check reliance on `DepositType` and `ADR`.
- **PCA / LDA:** PCA with explained-variance plot; compare models with and without PCA; compare original vs PCA vs LDA features for one classifier.
- **Clustering:** tune K-Means (k, init), Agglomerative (linkage, k, on a documented random sample with a dendrogram) and DBSCAN (eps via k-distance plot, min_samples) / GMM, using Silhouette and Davies–Bouldin only.
- **Final:** evaluate the selected models once on the untouched test set; save fitted pipelines with `joblib`; add `predict` commands to the CLI that load saved artifacts.

## 9. How to reproduce

```
pip install -r requirements.txt
python main.py analyze    # data checks + EDA plots
python main.py train      # Task A: regression baselines
python main.py classify   # Task B: classification baselines
python main.py cluster    # Task C: K-Means k scan + profiles
python -m src.cluster_silhouette   # Task C: silhouette per cluster
python -m pytest          # automated leakage / split / pipeline checks
```
