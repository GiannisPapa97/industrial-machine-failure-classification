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

df = load_raw()    # all 14 columns, validated
X, y = load_xy()   # 6 features and the `Machine failure` target
```

`load_raw` raises `DataValidationError` if columns are missing, values are null,
dtypes differ, `Type` is not L/M/H, or a target or flag column is not 0/1.
`load_xy` drops the identifiers and the five failure-mode flags, which would
leak the target.

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## Layout

```
src/machine_failure/   package code (data.py: dataset download and loading)
tests/                 tests
data/raw/              downloaded dataset (not committed)
```
