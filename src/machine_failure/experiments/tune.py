"""Tune the selected models and compare them with their default settings."""

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from machine_failure.config import TuningConfig, load_or_default
from machine_failure.data import RANDOM_STATE
from machine_failure.evaluation import (
    MAIN_METRIC,
    N_SPLITS,
    best_settings,
    cross_validate_model,
    describe_run,
    format_comparison,
    nested_cross_validate,
    summarize_scores,
    tune_model,
)
from machine_failure.experiments.options import (
    add_config_option,
    add_data_option,
    data_option,
)
from machine_failure.models import SEARCH_SPACES, TUNED_MODELS, build_candidates
from machine_failure.pipelines.splits import load_split

logger = logging.getLogger(__name__)

HELP = "tune the selected models with nested cross-validation"


@dataclass
class TuningResult:
    """What tuning found for one model."""

    default_scores: pd.DataFrame  # untuned model, one row per outer fold
    tuned_scores: pd.DataFrame  # nested cross-validation, one row per outer fold
    best_settings: dict[str, Any]  # best setting, found on the whole training set
    search_score: float  # PR-AUC of that setting on the search's own folds


def tune_models(
    X: pd.DataFrame,
    y: pd.Series,
    names: tuple[str, ...] = TUNED_MODELS,
    *,
    n_iter: int = 20,
    inner_splits: int = 3,
    n_repeats: int = 1,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> dict[str, TuningResult]:
    """Tune each model on the 9-feature set; compare with its default settings.

    For every model this scores the default settings, estimates the tuned
    model with nested cross-validation (on the same outer folds), and runs the
    search once on all of ``X`` and ``y`` to find the setting to use. Pass the
    training set only. Progress is logged, since a search takes minutes.
    """

    results = {}
    for name in names:
        model, space = build_candidates()[name], SEARCH_SPACES[name]
        logger.info("%s: scoring the default settings", name)
        default = cross_validate_model(
            model, X, y, n_repeats=n_repeats, random_state=random_state
        )
        logger.info("%s: nested cross-validation of the tuned model", name)
        tuned = nested_cross_validate(
            model,
            space,
            X,
            y,
            n_iter=n_iter,
            inner_splits=inner_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        logger.info("%s: searching on the whole training set", name)
        search = tune_model(
            model,
            space,
            X,
            y,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        results[name] = TuningResult(
            default_scores=default,
            tuned_scores=tuned,
            best_settings=best_settings(search),
            search_score=float(search.best_score_),
        )
    return results


def comparison_table(results: dict[str, TuningResult]) -> pd.DataFrame:
    """Mean and std of every metric for each model, untuned and tuned."""
    summaries = {}
    for name, result in results.items():
        summaries[f"{name} (default)"] = summarize_scores(result.default_scores).stack()
        summaries[f"{name} (tuned)"] = summarize_scores(result.tuned_scores).stack()
    return pd.DataFrame(summaries).T


def format_results(results: dict[str, TuningResult]) -> str:
    """The default against tuned table, then the best settings of each model."""
    lines = [format_comparison(comparison_table(results)), ""]
    for name, result in results.items():
        nested = result.tuned_scores[MAIN_METRIC].mean()
        lines.append(f"{name}: best settings {result.best_settings}")
        lines.append(
            f"    PR-AUC on the search's own folds {result.search_score:.3f} "
            f"(optimistic) against {nested:.3f} in nested cross-validation"
        )
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``mf tune``."""
    add_config_option(parser, "tune_default.toml")
    add_data_option(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TUNED_MODELS,
        help="models to tune (default: all of them)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        help="random settings to try per search (default: 20)",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        help="folds inside each search (default: 3)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="repeat the outer cross-validation this many times (default: 1)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="parallel jobs; -1 uses every core (default: 1)",
    )


def run(args: argparse.Namespace) -> None:
    """``mf tune``: tune the models and compare them with their defaults."""
    config = load_or_default(
        args.config,
        TuningConfig,
        models=tuple(args.models) if args.models else None,
        n_iter=args.n_iter,
        inner_splits=args.inner_splits,
        n_repeats=args.repeats,
        n_jobs=args.n_jobs,
        data=data_option(args),
    )
    X_train, _, y_train, _ = load_split(config.data)  # the test rows are not used
    results = tune_models(
        X_train,
        y_train,
        config.models,
        n_iter=config.n_iter,
        inner_splits=config.inner_splits,
        n_repeats=config.n_repeats,
        n_jobs=config.n_jobs,
    )

    print(describe_run(X_train, y_train, config.n_repeats))
    print(
        f"All 9 features. Tuned models: {config.n_iter} random settings per "
        f"search, {config.inner_splits}-fold inner cross-validation, scores from "
        f"nested cross-validation on the same {N_SPLITS} outer folds as the default."
    )
    print(format_results(results))
