# The final model

Training the chosen model, evaluating it once on the held-out test set, scoring
new rows with it, and what it relies on. Back to the [README](../README.md).

## Final evaluation

```bash
uv run mf train                  # train and save
uv run mf train --config configs/train_final.toml   # the same, from a config file
uv run mf evaluate --confirm     # evaluate once
uv run mf predict rows.csv       # score new rows with the saved model
uv run mf interpret              # what it relies on
```

- `pipelines/training.py` (`TrainingPipeline`) trains the model a `TrainingConfig`
  describes (by default the chosen tuned model, `FINAL_MODEL`, random forest) on
  the whole training set: load, split, fit on the training rows only. `mf train`
  runs it and saves the model with its threshold (the model's own, 0.475 for
  random forest, unless one is set) to `models/final_model.joblib`. The `models/` folder is not committed,
  and an existing file is only replaced with `--overwrite`. The threshold was
  chosen on out-of-fold predictions of models trained on 80% of the training
  rows, so it is an approximation for the model refit on all of them.
  `--config FILE.toml` reads the settings from a file (see `configs/`), and
  `--model`, `--threshold` and `--output` override it. Use it in Python with
  `TrainingPipeline(config).run()`, then `.save(result)`.
- The model file also records how the model was made (`FinalModel.metadata`):
  the training config, where the threshold came from, the number of training rows
  and failures, the library versions (Python, scikit-learn, XGBoost, NumPy,
  pandas), when it was trained, and a fingerprint of the data and of the
  held-out rows. Loading warns when the model was saved with another
  scikit-learn version, and a file that is not a saved model is reported in one
  line. A model saved before this metadata existed still loads, with a warning;
  retrain it with `uv run mf train --overwrite` (training is deterministic, so
  the model is the same) to record it.
- The train/test split is fixed for the project (stratified 80/20, seed 42) and is
  not a setting: the model, its settings, the features and the threshold were
  all chosen on the same training rows, so another split would put rows that
  shaped those choices into the test set.
- `pipelines/evaluation.py` (`EvaluationPipeline`) scores the saved model on the test
  set. The test set is meant to be used once, on the final model, so `run` refuses
  to start without `confirm=True` and `mf evaluate` without `--confirm`, in both
  cases before any data is read. It reports every metric with a 95% bootstrap
  interval (1,000 resamples of the test rows, keeping the number of failures
  fixed) next to the cross-validation estimate at the same threshold. The model is
  judged at the threshold saved with it, and it is only scored on the rows that
  were held out when it was trained: the evaluation compares the data and split
  fingerprints with the ones saved with the model, and refuses on a mismatch or
  when the model has none. The cross-validation estimate refits copies of the
  saved model, so it describes that model even when it was trained untuned or on
  other features. `--config FILE.toml` reads the settings
  from a file; `--model-path`, `--n-bootstrap` and `--skip-reference` override it.
- `pipelines/inference.py` (`InferencePipeline`) and `mf predict` score new rows with
  the saved model. The input is a CSV with `Type`, `Air temperature`, `Process
  temperature`, `Rotational speed`, `Torque` and `Tool wear` (other columns are
  ignored). The columns and values are checked first, so a missing column or an
  unknown machine type fails with a clear message. The output has the failure
  probability and a 0/1 alert at the saved threshold for every input row; it is
  printed, or written with `--output` (an existing file is only replaced with
  `--overwrite`). Saved models are pickles: only use model files you created.
- `evaluation/interpretation.py` studies the model on the training set only, on held-out
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
  scikit-learn version (loading warns when it differs).

Using the saved model:

```python
from machine_failure.models import load_final_model

final = load_final_model()
flagged = final.predict(X_new)  # X_new has Type and the five measurements
probability = final.probability(X_new)
```
