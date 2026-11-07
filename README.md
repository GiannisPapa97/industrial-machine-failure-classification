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

## Approach

Every choice was made with cross-validation on the training set (80% of the
rows); the held-out test set was used once, at the end.

1. **Baselines.** A dummy classifier (PR-AUC 0.034) and a class-weighted logistic
   regression (PR-AUC 0.459) set the bar.
2. **Model comparison.** Ten models with default settings on the same folds;
   random forest (0.890) and histogram gradient boosting (0.889) lead.
3. **Features.** The three derived features (`temp_diff`, `power`, `strain`)
   clearly help the tree-based models, so all nine features are kept.
4. **Tuning.** The four best models are tuned with randomized search, scored by
   nested cross-validation; the tuned random forest reaches PR-AUC 0.911.
5. **Threshold.** The decision threshold is chosen on out-of-fold predictions:
   0.475, the best F1 for the random forest.
6. **Final evaluation.** The model is trained on all training rows and scored
   once on the test set, with bootstrap intervals.

Details: [data and features](docs/data-and-features.md),
[model selection](docs/model-selection.md) and
[the final model](docs/final-model.md).

## How it is built

- **Layered package.** `cli → data → features → evaluation → models → config →
  pipelines → experiments`; each package only imports from the ones before it,
  and a test enforces that.
- **Clear interfaces.** A `Model` base class with a registry, typed and validated
  configs read from TOML, reusable pipelines for training, comparison, inference
  and the final evaluation, and one `mf` command for every workflow.
- **A model file that can be checked.** The saved model records its training
  config, where its threshold came from, the library versions and a fingerprint
  of the held-out rows; the evaluation refuses to score any other rows.
- **Tested.** Strict `mypy` and `ruff`, unit tests on synthetic data with a 90%
  coverage minimum for every package (currently 100%), and integration tests that
  pin the results on the real dataset. CI runs all of them on every pull request.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync                               # install
uv run mf download                    # fetch the dataset into data/raw/
uv run mf compare                     # compare the models (about a minute)
uv run mf train                       # train and save the final model
uv run mf predict rows.csv            # score new rows with the saved model
```

The dataset is not committed; `mf download` saves it to `data/raw/ai4i2020.csv`
(`--dest` chooses another directory) and never replaces an existing file unless
you pass `--overwrite`. Every command and option is described in
[commands](docs/commands.md); `uv run mf --help` lists them too.

## Documentation

- [Data and features](docs/data-and-features.md): loading and validation, the
  derived features, splitting, preprocessing and cross-validation.
- [Model selection](docs/model-selection.md): baselines, model and feature-set
  comparisons, hyperparameter tuning and threshold choice, with all the results.
- [The final model](docs/final-model.md): training, the one-time test
  evaluation, scoring new rows and what the model relies on.
- [Commands](docs/commands.md): the `mf` command, its subcommands, output and
  errors.
- [Development](docs/development.md): checks and tests, code layout, configs and
  how to add a model.

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
