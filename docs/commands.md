# Commands

Every workflow runs through one command, `mf`. Back to the
[README](../README.md).

## The `mf` command

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
