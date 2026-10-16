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
  training part only. Its `feature_set` argument chooses the features: `"all"`
  (default), `"raw"` (`Type` and the five measurements) or `"computed"` (only
  `temp_diff`, `power` and `strain`).
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

Cross-validates the two baselines and eight candidate models on the same
training folds and prints one table, ordered by mean PR-AUC. It takes about a
minute. The test set is not used. Use `--repeats N` for steadier estimates.

The candidates are random forest, extra trees, histogram gradient boosting,
XGBoost, KNN, SVM, an MLP and quadratic discriminant analysis, all behind the
shared preprocessing pipeline with default settings and no tuning. Models that
support it weight the classes inversely to their frequency (random forest, extra
trees, gradient boosting and SVM; XGBoost through `scale_pos_weight`).

Results at the time of writing (5-fold, 8,000 training rows, 271 failures; mean
± standard deviation across folds):

| Model                  | PR-AUC        | ROC-AUC       | Recall        | Precision     | F1            |
| ---------------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| random_forest          | 0.890 ± 0.030 | 0.971 ± 0.014 | 0.749 ± 0.057 | 0.985 ± 0.014 | 0.850 ± 0.040 |
| hist_gradient_boosting | 0.889 ± 0.017 | 0.971 ± 0.009 | 0.823 ± 0.034 | 0.899 ± 0.052 | 0.858 ± 0.023 |
| xgboost                | 0.855 ± 0.029 | 0.971 ± 0.012 | 0.801 ± 0.027 | 0.828 ± 0.015 | 0.814 ± 0.015 |
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
- XGBoost is third, about 0.035 behind them, which is roughly one standard
  deviation. It has the most balanced precision and recall of the three at the
  default threshold. It has many settings that were not tuned here.
- Precision, recall and F1 use the default 0.5 threshold. Random forest is very
  precise but misses about a quarter of the failures, while gradient boosting
  finds more of them with somewhat lower precision. Choosing the decision
  threshold is a separate step.
- ROC-AUC is nearly identical for the four best models, while PR-AUC ranks them,
  which is why PR-AUC is the main metric.
- These are default settings with no tuning, so close rankings may change.

## Feature set comparison

```bash
uv run python -m machine_failure.ablation
```

Cross-validates all 10 models on each of three feature sets, on the same
training folds, and prints one table with the models as rows and the feature
sets as columns. It takes about four minutes and reports progress on stderr. The
test set is not used. Use `--metric` to show another metric (default `pr_auc`)
and `--repeats N` for steadier estimates.

- `raw`: `Type` and the five measurements (6 features)
- `computed`: `temp_diff`, `power` and `strain` only (3 features)
- `all`: the raw and computed features together (9 features)

PR-AUC at the time of writing (5-fold, 8,000 training rows, 271 failures; mean
± standard deviation across folds):

| Model                  | raw           | computed      | all           |
| ---------------------- | ------------- | ------------- | ------------- |
| random_forest          | 0.755 ± 0.037 | 0.730 ± 0.045 | 0.890 ± 0.030 |
| hist_gradient_boosting | 0.793 ± 0.044 | 0.702 ± 0.067 | 0.889 ± 0.017 |
| xgboost                | 0.779 ± 0.055 | 0.603 ± 0.055 | 0.855 ± 0.029 |
| extra_trees            | 0.696 ± 0.026 | 0.629 ± 0.058 | 0.824 ± 0.026 |
| mlp                    | 0.733 ± 0.022 | 0.681 ± 0.066 | 0.795 ± 0.031 |
| svm                    | 0.661 ± 0.025 | 0.542 ± 0.077 | 0.682 ± 0.025 |
| knn                    | 0.517 ± 0.027 | 0.556 ± 0.086 | 0.615 ± 0.047 |
| qda                    | 0.373 ± 0.042 | 0.555 ± 0.080 | 0.444 ± 0.042 |
| logistic_regression    | 0.441 ± 0.033 | 0.324 ± 0.038 | 0.459 ± 0.047 |
| dummy                  | 0.034 ± 0.000 | 0.034 ± 0.000 | 0.034 ± 0.000 |

How to read it:

- The derived features clearly help the tree-based models. Going from `raw` to
  `all` raises PR-AUC by about 0.14 for random forest, 0.13 for extra trees, 0.10
  for gradient boosting and 0.08 for XGBoost, several times the spread across
  folds. The best results, random forest and gradient boosting at about 0.89,
  both use all 9 features.
- Logistic regression and SVM gain only 0.02, which is within the spread. A
  plausible reason is that linear models cannot use threshold effects even when
  given the combined quantities; this has not been tested.
- The three computed features alone nearly match the raw features for random
  forest (0.73 against 0.76), but combining them with the raw columns is better
  than either. QDA is the exception and does best on the computed features alone.
- Based on this, the remaining steps use all 9 features.

## Hyperparameter tuning

```bash
uv run python -m machine_failure.tune --n-jobs 4
```

Tunes four models with randomized search and compares each with its default
settings. For every model it scores the defaults, estimates the tuned model with
nested cross-validation, and runs one search on the whole training set to find
the setting to use. It uses the training set only and all 9 features, and takes
about 20 minutes on 4 cores. Options: `--models` (a subset), `--n-iter` (random
settings per search, default 20), `--inner-splits` (default 3), `--repeats` and
`--n-jobs`.

**Why these four models.** The search is the expensive step, and the more
settings are compared on 271 failures, the more likely it is that the "best" one
is just the luckiest. So only the most promising models are tuned, chosen from
the default-settings comparison above:

- Random forest, histogram gradient boosting and XGBoost are the top three by
  PR-AUC (0.890, 0.889 and 0.855).
- Extra trees is next (0.824), a cheap relative of random forest.
- The MLP (0.795) ranks fifth, has no class-weight option and many more
  settings to tune, so it was left out to keep the search small. SVM and KNN
  (0.68 and 0.62) and logistic regression and QDA (about 0.45) are far behind.
  The feature-set comparison also showed that all 9 features are best for the
  tree-based models.

The search spaces are in `search_spaces.py`. Each contains the default setting,
so the untuned model is always one of the options, and the class-imbalance
setting is part of the search (class weights, or `scale_pos_weight` for
XGBoost).

**How the tuned scores are estimated.** In each of the 5 outer folds the search
runs on that fold's training part only, with its own 3-fold cross-validation,
and the setting it picks is scored on the held-out fold. The best score a search
reports for itself is optimistic, because the setting was picked on those same
folds; the nested scores are not. They use the same outer folds as the untuned
models, so the two are directly comparable.

Results at the time of writing (5 outer folds, 8,000 training rows, 271
failures; mean ± standard deviation across folds):

| Model                            | PR-AUC        | ROC-AUC       | Recall        | Precision     | F1            |
| -------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| random_forest (default)          | 0.890 ± 0.030 | 0.971 ± 0.014 | 0.749 ± 0.057 | 0.985 ± 0.014 | 0.850 ± 0.040 |
| random_forest (tuned)            | 0.911 ± 0.025 | 0.978 ± 0.017 | 0.819 ± 0.042 | 0.987 ± 0.012 | 0.895 ± 0.024 |
| hist_gradient_boosting (default) | 0.889 ± 0.017 | 0.971 ± 0.009 | 0.823 ± 0.034 | 0.899 ± 0.052 | 0.858 ± 0.023 |
| hist_gradient_boosting (tuned)   | 0.893 ± 0.022 | 0.975 ± 0.012 | 0.819 ± 0.034 | 0.923 ± 0.036 | 0.867 ± 0.024 |
| extra_trees (default)            | 0.824 ± 0.026 | 0.971 ± 0.011 | 0.546 ± 0.055 | 0.947 ± 0.030 | 0.692 ± 0.051 |
| extra_trees (tuned)              | 0.873 ± 0.028 | 0.976 ± 0.018 | 0.764 ± 0.050 | 0.911 ± 0.045 | 0.829 ± 0.023 |
| xgboost (default)                | 0.855 ± 0.029 | 0.971 ± 0.012 | 0.801 ± 0.027 | 0.828 ± 0.015 | 0.814 ± 0.015 |
| xgboost (tuned)                  | 0.870 ± 0.034 | 0.980 ± 0.014 | 0.808 ± 0.044 | 0.842 ± 0.041 | 0.823 ± 0.018 |

Best settings found by the search on the whole training set:

| Model                  | Settings                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| random_forest          | 500 trees, `max_depth` 8, `min_samples_leaf` 2, `max_features` 0.8, no class weights                                      |
| extra_trees            | 300 trees, no depth limit, `min_samples_leaf` 5, `max_features` 0.8, no class weights                                     |
| hist_gradient_boosting | `learning_rate` 0.1, 100 iterations, 31 leaf nodes, `min_samples_leaf` 20, `l2_regularization` 0.1, balanced class weights |
| xgboost                | 300 trees, `learning_rate` 0.05, `max_depth` 6, `min_child_weight` 1, `scale_pos_weight` 5, no subsampling                 |

How to read it:

- Tuning helps modestly. Extra trees gains the most (+0.05 PR-AUC, more than its
  spread). Random forest gains 0.02 and XGBoost 0.015, within one spread, and
  gradient boosting about 0.004, so its defaults were already close to its best.
- Random forest is first at 0.911 and gradient boosting second at 0.893, but the
  gap is smaller than the spread across folds, so this comparison does not
  clearly separate them. Extra trees and XGBoost are tied behind them.
- Strong class weighting was not needed for PR-AUC: the best random forest and
  extra trees use none, and XGBoost prefers a weight of 5 over about 28.
- For every model the search's own best score is within 0.001 of the nested
  estimate, so picking the best of 20 settings barely inflated it here.
- This is one run with one seed and 20 settings per model. Precision, recall and
  F1 still use the default 0.5 threshold; choosing the threshold is a separate
  step.

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
  ablation.py          compare all models across the feature sets
  search_spaces.py     hyperparameter search spaces of the tuned models
  tuning.py            randomized search and nested cross-validation
  tune.py              tune the models and compare with the defaults
tests/                 tests
notebooks/             exploration notebooks
data/raw/              downloaded dataset (not committed)
```
