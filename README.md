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

## Result

The final model is a tuned random forest on nine features (the six raw ones plus
`temp_diff`, `power` and `strain`), flagging a machine when its predicted failure
probability is at least 0.475. On the held-out test set (2,000 machines, 68
failures) it catches 55 failures with 2 false alarms (recall 0.81, precision
0.96, PR-AUC 0.90), in line with the cross-validation estimate. It misses the
failures that the data generates at random, which no model of these features can
predict. Details, intervals and limits are in
[Final evaluation](#final-evaluation).

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

## Threshold tuning

A model gives each machine a failure probability, and a machine is flagged when
its probability is at least the **threshold**. A low threshold catches more
failures (high recall) but raises more false alarms (low precision); a high one
does the opposite. The default of 0.5 is arbitrary, so this step shows the
trade-off for the four tuned models and what each recall target costs.

```bash
uv run python -m machine_failure.thresholds --plot pr_curves.png
```

It uses the training set only; the test set is not touched. Options: `--targets`
(recall targets to compare, default `0.80 0.85 0.90`) and `--plot FILE` (save the
precision-recall curves as a PNG). It takes about a minute.

How it works:

- `tuned_models.py` fixes the four winning configurations from the tuning step
  (`build_tuned_models`).
- `predictions.py` gives every training row a failure probability from a model
  that never saw it (`out_of_fold_predictions`): stratified 5-fold, the same
  folds as the cross-validation elsewhere. These probabilities behave like
  predictions on new data, so a threshold chosen on them is not tuned on the
  data the model was trained on.
- `thresholds.py` computes the counts (TP, FP, FN, TN), precision, recall and F1
  at every threshold, and finds the highest threshold that reaches a recall
  target, which is the one with the fewest false alarms.

Results (out-of-fold on the 8,000 training rows, 271 failures). For each recall
target, the highest threshold that reaches it:

| Recall target | Model                  | Threshold | Correct alerts | False alarms | Missed | Precision |    F1 |
| ------------- | ---------------------- | --------: | -------------: | -----------: | -----: | --------: | ----: |
| 80%           | random_forest          |     0.657 |            217 |            1 |     54 |     0.995 | 0.888 |
|               | hist_gradient_boosting |     0.923 |            217 |            6 |     54 |     0.973 | 0.879 |
|               | xgboost                |     0.495 |            217 |           32 |     54 |     0.871 | 0.835 |
|               | extra_trees            |     0.328 |            217 |           39 |     54 |     0.848 | 0.824 |
| 85%           | random_forest          |     0.147 |            231 |           55 |     40 |     0.808 | 0.829 |
|               | xgboost                |     0.199 |            231 |           80 |     40 |     0.743 | 0.794 |
|               | hist_gradient_boosting |     0.100 |            231 |          112 |     40 |     0.673 | 0.752 |
|               | extra_trees            |     0.148 |            231 |          173 |     40 |     0.572 | 0.684 |
| 90%           | random_forest          |     0.060 |            244 |          216 |     27 |     0.530 | 0.668 |
|               | xgboost                |     0.027 |            244 |          320 |     27 |     0.433 | 0.584 |
|               | hist_gradient_boosting |     0.008 |            244 |          331 |     27 |     0.424 | 0.577 |
|               | extra_trees            |     0.068 |            244 |          433 |     27 |     0.360 | 0.515 |

At the threshold with the best F1:

| Model                  | Threshold | Recall | Precision |    F1 | False alarms |
| ---------------------- | --------: | -----: | --------: | ----: | -----------: |
| random_forest          |     0.475 |  0.827 |     0.987 | 0.900 |            3 |
| hist_gradient_boosting |     0.864 |  0.815 |     0.969 | 0.886 |            7 |
| extra_trees            |     0.416 |  0.764 |     0.945 | 0.845 |           12 |
| xgboost                |     0.654 |  0.790 |     0.907 | 0.844 |           22 |

How to read it:

- Random forest is the best model at every recall target. It was only tied with
  gradient boosting on PR-AUC, but at a fixed recall it has the fewest false
  alarms, and the gap widens as the target rises.
- There is a knee near 83% recall. Up to there, random forest finds failures with
  almost no false alarms (3 false alarms among more than 7,700 normal machines at
  the best-F1 threshold). Beyond it, each extra point of recall is expensive: 85%
  costs about 55 false alarms (81% of alerts real), and 90% costs 216 (about one
  alert in two is false).
- This fits the data: about 13 to 15% of the failures get a probability below 0.1
  from every model. They match the failures the data generation makes random
  (tool wear failures within 200-240 minutes, random failures and failures with no
  recorded cause), so they cannot be caught without flagging many normal machines.
- Thresholds are not comparable across models. Class-weighted models such as
  gradient boosting give failures extra weight when they train, which inflates
  their probabilities (best threshold 0.86, against 0.48 for random forest).
  Compare models at the same recall, not at the same threshold.
- Caveats: the thresholds are chosen on the same out-of-fold predictions they are
  evaluated on, so they are slightly optimistic, and differences of a few false
  alarms between models (for example 1 against 6 at 80% recall) are within noise.
  The larger gaps at 85% and 90% are more convincing.

The choice of model and operating point (which recall to aim for) depends on how
costly a false alarm is compared with a missed failure. The final evaluation
uses random forest at its best-F1 threshold, 0.475 (about 83% recall with almost
no false alarms). A threshold near 0.147 would reach 85% recall at the price of
about 55 false alarms on the training data; change `FINAL_THRESHOLD` in
`final_model.py` if false alarms are cheaper than missed failures for your use.

## Final evaluation

```bash
uv run python -m machine_failure.final_model                 # train and save
uv run python -m machine_failure.final_evaluation --confirm  # evaluate once
uv run python -m machine_failure.interpretation              # what it relies on
```

- `final_model.py` trains the chosen tuned model (`FINAL_MODEL`, random forest) on
  the whole training set and saves it with its threshold (`FINAL_THRESHOLD`,
  0.475) to `models/final_model.joblib`. The `models/` folder is not committed,
  and an existing file is only replaced with `--overwrite`. The threshold was
  chosen on out-of-fold predictions of models trained on 80% of the training
  rows, so it is an approximation for the model refit on all of them.
- `final_evaluation.py` scores the saved model on the test set. The test set is
  meant to be used once, on the final model, so the command refuses to run
  without `--confirm`. It reports every metric with a 95% bootstrap interval
  (1,000 resamples of the test rows, keeping the number of failures fixed) next
  to the cross-validation estimate at the same threshold.
- `interpretation.py` studies the model on the training set only, on held-out
  folds and out-of-fold predictions, so the test set is not looked at again.

Test result (random forest, threshold 0.475; 2,000 rows, 68 failures):

| Metric    | Test set | 95% interval   | Cross-validation |
| --------- | -------: | -------------- | ---------------: |
| PR-AUC    |    0.904 | [0.842, 0.958] |            0.911 |
| ROC-AUC   |    0.986 | [0.972, 0.997] |            0.976 |
| Recall    |    0.809 | [0.721, 0.897] |            0.823 |
| Precision |    0.965 | [0.914, 1.000] |            0.987 |
| F1        |    0.880 | [0.817, 0.938] |            0.897 |
| Accuracy  |    0.993 | [0.989, 0.996] |            0.994 |

That is 55 failures caught, 2 false alarms, 13 failures missed and 1,930 normal
machines correctly left alone. Every cross-validation figure lies inside the
test interval, so the estimate held up and the threshold chosen on the training
data carried over to the test set.

What the model relies on (drop in PR-AUC when one input is shuffled, on held-out
training folds):

| Feature             | Drop in PR-AUC |
| ------------------- | -------------: |
| Torque              |          0.803 |
| Rotational speed    |          0.636 |
| Tool wear           |          0.429 |
| Air temperature     |          0.394 |
| Process temperature |          0.242 |
| Type                |          0.088 |

Torque and speed dominate, because the power and strain features are built from
them. The values are not additive: shuffling one input also disturbs the derived
features that use it, and the two temperatures matter together through
`temp_diff`.

Which failures it misses (out-of-fold on the training set, 48 of 271 missed):

| Failure kind                                      | Failures | Caught | Missed |
| ------------------------------------------------- | -------: | -----: | -----: |
| heat, power and overstrain failures (HDF/PWF/OSF) |      228 |    222 |      6 |
| involving tool wear (TWF)                         |       36 |      1 |     35 |
| no flag at all                                    |        7 |      0 |      7 |

The model has learned the deterministic failure modes almost perfectly (97% of
them caught). It misses almost all tool wear failures, which the data generates
at random within 200-240 minutes of wear (only about 5% of machines in that range
fail), so they look like healthy machines: the missed failures have the same
median torque and speed as normal rows and differ only in a high tool wear
(median 209 against 106 minutes). Catching them would mean flagging every machine
in that range. Setting those 43 failures aside leaves 228 of 271 (84%) as the
most any threshold can catch without a flood of false alarms, close to the
model's 82%.

Limits:

- The data is synthetic, generated from known rules plus random failures. A high
  score shows that the model recovers those rules, not that it would predict
  failures on real machines.
- With 68 test failures, every score is uncertain: recall could plausibly be
  anywhere from 72% to 90%.
- The 13 test failures it misses are mostly the random kind, which no model of
  these features can predict.
- The operating point (threshold 0.475) trades missed failures against false
  alarms and should be set from the real cost of each.
- The saved model is a pickle: load only files you created, with the same
  scikit-learn version.

Using the saved model:

```python
from machine_failure.final_model import load_final_model

final = load_final_model()
flagged = final.predict(X_new)  # X_new has Type and the five measurements
probability = final.probability(X_new)
```

## Notebooks

Exploration lives in `notebooks/`:

- `01_eda.ipynb`: exploratory data analysis of the dataset.
- `02_model_comparison.ipynb`: the default models on every metric, default against
  tuned settings, and for the four tuned models the precision-recall and ROC
  curves, metrics against the threshold, how well the probabilities separate the
  classes, confusion matrices at the candidate operating points, and an
  interactive threshold slider. It takes about two minutes to run.

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
  tuned_models.py      the tuned model configurations
  predictions.py       out-of-fold predicted probabilities
  thresholds.py        threshold analysis and recall targets
  final_model.py       train, save and load the final model
  final_evaluation.py  one-shot test set evaluation with intervals
  interpretation.py    feature importance and missed failures
tests/                 tests
notebooks/             exploration notebooks
data/raw/              downloaded dataset (not committed)
```
