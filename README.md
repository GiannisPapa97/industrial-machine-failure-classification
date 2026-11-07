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

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest -m "not integration"
uv run python scripts/coverage_by_package.py
```

`pytest` also prints a coverage report (per file and total, with branch
coverage and the uncovered line numbers). `scripts/coverage_by_package.py` reads
the coverage data of the last run and prints one row per package of
`src/machine_failure` plus a total, and exits with an error if any package is
below the minimum: 90% for each package (`fail_under` in the
`[tool.package-coverage]` section of `pyproject.toml`; `--fail-under` overrides
it). Coverage counts statements and branches together, as coverage.py does.

In CI, the `test` job puts that package table, with the per-file table in a
collapsed section below it, in the run summary (visible from the pull request's
Checks tab) and posts it as a single comment on the pull request that is updated
on each push. The report is written before the job fails, so a package below
the minimum is visible in the comment.

### Unit and integration tests

The unit tests in `tests/` use small synthetic data and need no download. The
integration tests in `tests/integration/` run the real workflows on the real
dataset and compare the results with the numbers recorded in this README, so a
refactoring that changes a result is caught. They are marked `integration`
automatically, and are skipped when the dataset has not been downloaded
(`uv run mf download`). Tests that take more than about 30
seconds are also marked `slow`. One unit test (`tests/test_tracked_files.py`) fails
if `.gitignore` hides a file under `src`, `tests`, `scripts` or `configs`, which
git would otherwise skip silently. It is meant for your own machine; in CI, an
ignored file is simply absent from the checkout.

```bash
uv run pytest                                    # everything
uv run pytest -m "not integration"               # unit tests only, with coverage
uv run pytest -m integration --no-cov            # integration tests only
uv run pytest -m "integration and not slow" --no-cov
```

Add `--no-cov` when running a subset, since its coverage is not meaningful and
the measurement slows the tests down.

In CI, the `test` job runs the unit tests with coverage (`-m "not integration"`),
and a separate `integration` job downloads the dataset (cached between runs) and
runs the integration tests, which takes several minutes. The job fails if the
dataset is missing, so a failed download cannot pass as skipped tests. Together
they check that a change keeps both the code covered and the recorded results
unchanged.

## Layout

```
src/machine_failure/
  cli/           shared by the commands: short error messages and logging
  data/          download, schema and validation, loading, train/test split
  features/      derived features (FeatureEngineer), encoding and scaling
  evaluation/    cross-validation, comparison, out-of-fold predictions,
                 thresholds, hyperparameter search, feature importance
  models/        the Model base class and registry, the baselines and
                 candidates as model classes, search spaces, tuned
                 configurations, the final model
  config/        typed, validated settings for the workflows, read from TOML
  pipelines/     the workflows as reusable pipelines: training, comparison,
                 inference and the final test evaluation
  experiments/   the `mf` command (app.py) and one module per subcommand
configs/         example config files
scripts/         tooling that is not part of the package (coverage by package)
tests/
  cli/ data/ features/ evaluation/ models/ config/ pipelines/ experiments/   unit tests, one folder per package
  integration/                                                              tests on the real dataset
notebooks/       exploration notebooks
data/raw/        downloaded dataset (not committed)
```

Dependencies only point downwards: `cli` is at the bottom (any package may use it), then `data`, used by `features`, which is used by
`evaluation`, then `models`, then `config` (which checks model names against the
model registry), then `pipelines`, then `experiments`. A test (`tests/test_layering.py`) fails if a
package imports from one above it. Each package re-exports its
public names, so code imports from the package
(`from machine_failure.models import build_candidates`), not from its files.

### Config files

Workflow settings are frozen dataclasses in `machine_failure.config`, checked when
they are created, so a typo or an out-of-range value fails at once with a clear
message: `DataConfig` (dataset path; the split is fixed), `ComparisonConfig` (models,
feature set, folds, repeats), `TrainingConfig` (model, tuned settings or not,
threshold, output path; the threshold defaults to the tuned model's own),
`EvaluationConfig` (saved model, bootstrap resamples, whether to add the
cross-validation reference), `TuningConfig` (models, random settings per search,
inner folds, repeats, parallel jobs), `ThresholdsConfig` (recall targets, plot),
`InterpretationConfig` (model, threshold, importance repeats) and
`InferenceConfig` (saved model). Their defaults are the values the commands used
before, so an empty file changes nothing. A model that gives no probabilities
(the SVM) cannot be trained as the final model, since alerts come from a
probability threshold.

`load_config(path, ConfigClass)` reads a TOML file. Top-level keys are the config's
fields and `[data]` holds the nested data settings. Keys left out keep their
default; unknown keys, wrong types and invalid values raise a `ConfigError` that
names the file and the key. The examples are in `configs/`, and a test loads every
one of them.

```python
from pathlib import Path

from machine_failure.config import ComparisonConfig, load_config

config = load_config(Path("configs/compare_trees.toml"), ComparisonConfig)
```

`mf compare`, `mf tune`, `mf thresholds`, `mf train`, `mf evaluate` and
`mf interpret` read them with `--config`.

### Adding a model

Every model is a subclass of `Model` (`models/base.py`) registered with
`@register`. A subclass sets `name` and `role` (`"baseline"` or `"candidate"`) and
returns its bare estimator from `make_estimator()`; `build(feature_set)` puts it
behind the shared preprocessing. Models that are tuned also set `search_space`
(keys like `model__max_depth`), `tuned_settings` (bare names like `max_depth`) and
`threshold` (the best-F1 threshold of the tuned model, from `mf thresholds`).
Registration order is the order of every table and comparison.

```python
@register
class MyModel(Model):
    name = "my_model"
    role = "candidate"

    def make_estimator(self) -> BaseEstimator:
        return MyClassifier(random_state=RANDOM_STATE)
```

`get_model(name)` and `all_models(role=None)` look models up. The functions
`build_baselines`, `build_candidates`, `build_all_models` and `build_tuned_models`
return fresh pipelines built from the registry, so a new model shows up in the
comparison commands once it is defined. Adding one changes the pinned
comparison results (the integration tests), which need updating on purpose.

Commands: one entry point, `mf`, installed with the project (run it with
`uv run mf COMMAND`, or `uv run python -m machine_failure COMMAND`). `mf --help`
lists the commands and `mf COMMAND --help` shows the options of one.

| Command         | What it does                                        | Section                    |
| --------------- | --------------------------------------------------- | -------------------------- |
| `mf download`   | download the dataset                                | Get the data               |
| `mf compare`    | compare models with cross-validation; `--feature-sets` across feature sets | Baselines, Model comparison, Feature set comparison |
| `mf tune`       | tune four models with nested cross-validation       | Hyperparameter tuning      |
| `mf thresholds` | threshold analysis at recall targets                | Threshold tuning           |
| `mf train`      | train and save the final model                      | Final evaluation           |
| `mf evaluate`   | evaluate the final model once on the test set       | Final evaluation           |
| `mf predict`    | score new rows with the saved model                 | Final evaluation           |
| `mf interpret`  | feature importance and the failures it misses       | Final evaluation           |

Every command that reads the dataset accepts `--data FILE`, and every workflow
command except `download` and `predict` accepts `--config FILE.toml` (examples in
`configs/`); options given on the command line override the file.

**Output, progress and errors.** Results (tables, scores, predictions) go to stdout.
Progress messages, such as `cross-validating random_forest`, go to stderr through
Python's standard `logging`. Every command accepts `-v`/`--verbose` (also show
diagnostics, with time and source) and `-q`/`--quiet` (only warnings and
errors). Using the library from Python or a notebook stays quiet unless you
configure logging (`machine_failure.cli.configure_logging()`).

Mistakes you can fix yourself (a missing file, an unknown model, a bad config, a
setting out of range, a CSV without the feature columns or with infinite values,
an output file that already exists, using the test set without `--confirm`) end
the command with a one-line `error: ...` message and exit status 1. Mistakes in
the command line itself, such as an unknown option or a model that is not one of
the choices, are reported by argparse with the usage text and exit status 2. Anything else
is a bug and shows the full traceback. In code, these errors derive from
`machine_failure.cli.UserError` (`ConfigError`, `DataValidationError`,
`ModelFileError`, `SplitMismatchError`, `UnconfirmedTestSetError`) or are
`FileNotFoundError` / `FileExistsError`.

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
