# Development

How to run the checks and tests, how the code is laid out, how configs work, and
how to add a model. Back to the [README](../README.md).

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
dataset and compare the results with the numbers recorded in these docs, so a
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

### Figures

The figures in `docs/figures/` are made from the training set only, with the
commands that report the same results:

```bash
uv run mf compare --plot docs/figures/model_comparison.png
uv run mf thresholds --plot docs/figures/precision_recall.png
```

Regenerate them when a change moves those results.

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
