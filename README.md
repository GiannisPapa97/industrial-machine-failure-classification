# Industrial Machine Failure Classification

Binary classification of machine failure (`Machine failure` target) with
scikit-learn, using the
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(10,000 rows, about 3.4% failures).

## Problem

Predict whether a machine fails from process measurements (air and process
temperature, rotational speed, torque, tool wear, product type).

- Target: `Machine failure` (binary, heavily imbalanced).
- Leakage: `TWF`, `HDF`, `PWF`, `OSF`, `RNF` define the target and must not be
  used as features. `UID` and `Product ID` are identifiers and are dropped.
- Metrics: PR-AUC, recall, precision, F1 and the confusion matrix. Accuracy is
  misleading because of the class imbalance.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync
```

## Get the data

The dataset is not committed. Download it from the UCI repository into
`data/raw/ai4i2020.csv`:

```bash
uv run python -m machine_failure.data
```

An existing file is never replaced unless you pass `--overwrite`. Use `--dest`
to write to a different directory.

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
from machine_failure.preprocessing import build_preprocessor

X, y = load_xy()
X_train, X_test, y_train, y_test = split_train_test(X, y)  # stratified 80/20

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
  training part only.
- `cross_validate_model` runs stratified 5-fold cross-validation on the training
  set and returns one row of scores per fold. `summarize_scores` gives the mean
  and standard deviation of each metric.
- Metrics: PR-AUC (the main one), ROC-AUC, recall, precision, F1 and accuracy.
  Precision, recall and F1 use the default 0.5 threshold.

## Baselines

```bash
uv run python -m machine_failure.baseline
```

Cross-validates a dummy classifier (always predicts "no failure") and a
class-weighted logistic regression on the training set, and prints the mean and
standard deviation of each metric. The test set is not used. Use `--repeats N`
to repeat the cross-validation with different shuffles for steadier estimates.

Results at the time of writing (5-fold, 8,000 training rows, 271 failures):

| Model               | PR-AUC        | ROC-AUC       | Recall        | Precision     |
| ------------------- | ------------- | ------------- | ------------- | ------------- |
| Dummy               | 0.034 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Logistic regression | 0.459 ± 0.047 | 0.923 ± 0.015 | 0.841 ± 0.051 | 0.162 ± 0.009 |

The dummy reaches 96.6% accuracy while catching no failures, which is why
accuracy is not used to compare models. Logistic regression catches most
failures at the default threshold, but most of its alerts are false alarms; the
PR-AUC of about 0.46 is the bar later models need to beat.

## Model comparison

```bash
uv run python -m machine_failure.comparison
```

Cross-validates the two baselines and seven candidate models on the same
training folds and prints one table, ordered by mean PR-AUC. It takes about a
minute. The test set is not used. Use `--repeats N` for steadier estimates.

The candidates are random forest, extra trees, histogram gradient boosting, KNN,
SVM, an MLP and quadratic discriminant analysis, all behind the shared
preprocessing pipeline with default settings and no tuning. Models that support
it weight the classes inversely to their frequency (random forest, extra trees,
gradient boosting and SVM).

Results at the time of writing (5-fold, 8,000 training rows, 271 failures; mean
± standard deviation across folds):

| Model                  | PR-AUC        | ROC-AUC       | Recall        | Precision     | F1            |
| ---------------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| random_forest          | 0.890 ± 0.030 | 0.971 ± 0.014 | 0.749 ± 0.057 | 0.985 ± 0.014 | 0.850 ± 0.040 |
| hist_gradient_boosting | 0.889 ± 0.017 | 0.971 ± 0.009 | 0.823 ± 0.034 | 0.899 ± 0.052 | 0.858 ± 0.023 |
| extra_trees            | 0.824 ± 0.026 | 0.971 ± 0.011 | 0.546 ± 0.055 | 0.947 ± 0.030 | 0.692 ± 0.051 |
| mlp                    | 0.795 ± 0.031 | 0.971 ± 0.011 | 0.668 ± 0.047 | 0.783 ± 0.081 | 0.719 ± 0.043 |
| svm                    | 0.682 ± 0.025 | 0.962 ± 0.011 | 0.871 ± 0.048 | 0.309 ± 0.020 | 0.455 ± 0.024 |
| knn                    | 0.615 ± 0.047 | 0.915 ± 0.011 | 0.240 ± 0.030 | 0.918 ± 0.073 | 0.380 ± 0.041 |
| logistic_regression    | 0.459 ± 0.047 | 0.923 ± 0.015 | 0.841 ± 0.051 | 0.162 ± 0.009 | 0.271 ± 0.012 |
| qda                    | 0.444 ± 0.042 | 0.923 ± 0.015 | 0.362 ± 0.030 | 0.561 ± 0.038 | 0.439 ± 0.030 |
| dummy                  | 0.034 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

How to read it:

- Random forest and gradient boosting are tied on PR-AUC. Their gap is far
  smaller than the spread across folds, so this comparison cannot separate them.
  Both roughly double the logistic regression baseline.
- Precision, recall and F1 use the default 0.5 threshold. Random forest is very
  precise but misses about a quarter of the failures, while gradient boosting
  finds more of them with somewhat lower precision. Choosing the decision
  threshold is a separate step.
- ROC-AUC is nearly identical for the four best models, while PR-AUC ranks them,
  which is why PR-AUC is the main metric.
- These are default settings with no tuning, so close rankings may change.

## Notebooks

Exploration lives in `notebooks/`:

- `01_eda.ipynb`: exploratory data analysis of the dataset.

Open them with `uv run jupyter lab` after downloading the data. Notebooks are
committed without outputs, so clear all outputs before committing. `ruff`
checks and formats the notebooks together with the rest of the code.

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

`pytest` also prints a coverage report (per file and total, with branch
coverage and the uncovered line numbers). In CI, the `test` job adds the same
table to the run summary, visible from the pull request's Checks tab, and
posts it as a single comment on the pull request that is updated on each push.
The tests fail if total coverage drops below 90% (`fail_under` in
`pyproject.toml`).

## Layout

```
src/machine_failure/   package code
  data.py              download, load, validate, train/test split
  features.py          derived features (FeatureEngineer)
  preprocessing.py     encoding and scaling pipeline
  evaluation.py        cross-validated metrics
  baseline.py          baseline models and their comparison
  models.py            candidate models and the full model list
  comparison.py        compare all models, ordered by PR-AUC
tests/                 tests
notebooks/             exploration notebooks
data/raw/              downloaded dataset (not committed)
```
