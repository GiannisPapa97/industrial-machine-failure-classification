"""Evaluate the final model once on the held-out test set.

The test set is meant to be used a single time, on the model and threshold chosen
with cross-validation on the training set. Nothing in this module should be used
to tune the model or the threshold.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

from machine_failure.config import N_BOOTSTRAP, EvaluationConfig, load_config
from machine_failure.models import MODEL_PATH
from machine_failure.pipelines import EvaluationPipeline, UnconfirmedTestSetError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="required: confirms you want to use the test set, which is meant to "
        "be done once",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML file with the evaluation settings (see configs/evaluate_final."
        "toml); the options below override it",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help=f"the saved final model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        help=f"bootstrap resamples for the intervals (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="do not compute the cross-validation reference (saves about a minute)",
    )
    args = parser.parse_args()

    config = load_config(args.config, EvaluationConfig) if args.config else None
    config = config or EvaluationConfig()
    overrides = {
        "model_path": args.model_path,
        "n_bootstrap": args.n_bootstrap,
        "reference": False if args.skip_reference else None,
    }
    config = dataclasses.replace(
        config, **{key: value for key, value in overrides.items() if value is not None}
    )

    try:
        result = EvaluationPipeline(config).run(confirm=args.confirm)
    except UnconfirmedTestSetError:
        print(
            "The test set is meant to be used once, on the final model. "
            "Run again with --confirm to evaluate on it.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(result.format())


if __name__ == "__main__":
    main()
