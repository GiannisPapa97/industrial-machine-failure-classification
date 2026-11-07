# Industrial Machine Failure Classification

[![CI](https://github.com/GiannisPapa97/industrial-machine-failure-classification/actions/workflows/ci.yaml/badge.svg)](https://github.com/GiannisPapa97/industrial-machine-failure-classification/actions/workflows/ci.yaml)

Binary classification of machine failure (`Machine failure` target) with
scikit-learn, using the
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(10,000 rows, about 3.4% failures).

**Status:** complete (November 2026). The held-out test set has been used once,
for the final evaluation.

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
[the final model](docs/final-model.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync
```

## Get the data

The dataset is not committed. Download it from the UCI repository into
`data/raw/ai4i2020.csv`:

```bash
uv run mf download
```

An existing file is never replaced unless you pass `--overwrite`. Use `--dest`
to write to a different directory.

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

## License and credits

The code is released under the [MIT License](LICENSE).

The dataset is the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine
Learning Repository ([doi:10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is
not included in this repository; `mf download` fetches it. It was introduced in
S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications", 2020 Third International Conference on Artificial Intelligence
for Industries (AI4I), 2020
([doi:10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)).
