# Data and features

How the dataset is loaded and validated, the three derived features, and how the data is split, preprocessed and cross-validated. Back to the [README](../README.md).

## Load the data

```python
from machine_failure.data import load_raw, load_xy

df = load_raw()  # all 14 columns, validated
X, y = load_xy()  # 6 features and the `Machine failure` target
```

`load_raw` raises `DataValidationError` if columns are missing, values are null,
dtypes differ, `Type` is not L/M/H, or a target or flag column is not 0/1.
`load_xy` drops the identifiers and the five failure-mode flags, which would
leak the target.

## Feature engineering

`FeatureEngineer` is a scikit-learn transformer that appends three features to
the output of `load_xy`:

- `temp_diff`: process temperature minus air temperature (K)
- `power`: torque times rotational speed, converted from rpm (W)
- `strain`: tool wear times torque (minNm)

```python
from machine_failure.data import load_xy
from machine_failure.features import FeatureEngineer

X, y = load_xy()
X_features = FeatureEngineer().fit_transform(X)  # 6 columns in, 9 columns out
```

Each feature is computed row by row and nothing is learned from the data, so it
is safe to use inside a `Pipeline` and cross-validation.

## Split, preprocess and evaluate

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from machine_failure.data import load_xy, split_train_test
from machine_failure.evaluation import cross_validate_model, summarize_scores
from machine_failure.features import build_preprocessor

X, y = load_xy()
X_train, X_test, y_train, y_test = split_train_test(X, y)  # stratified 80/20, fixed

model = Pipeline(
    [("preprocess", build_preprocessor()), ("model", LogisticRegression())]
)
scores = cross_validate_model(model, X_train, y_train)  # one row per fold
print(summarize_scores(scores))  # mean and std of each metric
```

- `split_train_test` is stratified and seeded, so both parts keep the 3.4% failure
  rate. The test set is held out and used once, for the final evaluation.
- `build_preprocessor` adds the derived features, one-hot encodes `Type` and
  standardizes the numeric columns (11 named columns out). Put it inside the
  model `Pipeline` so each cross-validation fold learns its own scaling from its
  training part only. Its `feature_set` argument chooses the features: `"all"`
  (default), `"raw"` (`Type` and the five measurements) or `"computed"` (only
  `temp_diff`, `power` and `strain`).
- `cross_validate_model` runs stratified 5-fold cross-validation on the training
  set and returns one row of scores per fold. `summarize_scores` gives the mean
  and standard deviation of each metric.
- Metrics: PR-AUC (the main one), ROC-AUC, recall, precision, F1 and accuracy.
  Precision, recall and F1 use the default 0.5 threshold.
