from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from machine_failure.config import EvaluationConfig, TrainingConfig
from machine_failure.data import split_train_test
from machine_failure.models import FinalModel, load_final_model, with_preprocessing
from machine_failure.pipelines import (
    EvaluationPipeline,
    TrainingPipeline,
    UnconfirmedTestSetError,
    splits,
)
from machine_failure.pipelines import evaluation as final_evaluation
from machine_failure.pipelines.evaluation import (
    METRICS,
    bootstrap_indices,
    cross_validated_reference,
    evaluate_final,
    format_evaluation,
    metrics_at_threshold,
    reference_model,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]

# A case that can be worked out by hand: 3 failures and 3 normal rows.
LABELS = np.array([1, 1, 0, 1, 0, 0])
PROBABILITIES = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])


def cheap_models() -> dict[str, Pipeline]:
    forest = RandomForestClassifier(n_estimators=10, random_state=0)
    return {"random_forest": with_preprocessing(forest)}


def fitted_final(
    make_xy: MakeXY, threshold: float = 0.5
) -> tuple[FinalModel, pd.DataFrame, pd.Series]:
    """A quick final model trained on a synthetic split, and the test part of it."""
    X, y = make_xy(signal=6.0)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = cheap_models()["random_forest"].fit(X_train, y_train)
    return FinalModel("random_forest", model, threshold), X_test, y_test


def test_metrics_match_a_hand_calculation() -> None:
    scores = metrics_at_threshold(LABELS, PROBABILITIES, 0.65)

    assert (scores["tp"], scores["fp"], scores["fn"], scores["tn"]) == (2, 1, 1, 2)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(2 / 3)
    assert scores["accuracy"] == pytest.approx(4 / 6)
    assert scores["pr_auc"] == pytest.approx((1 + 1 + 3 / 4) / 3)
    assert scores["roc_auc"] == pytest.approx(8 / 9)


def test_a_probability_at_the_threshold_is_flagged() -> None:
    scores = metrics_at_threshold(LABELS, PROBABILITIES, 0.7)

    assert scores["tp"] + scores["fp"] == 3  # 0.9, 0.8 and 0.7


def test_nothing_flagged_gives_zero_precision_and_recall() -> None:
    scores = metrics_at_threshold(LABELS, PROBABILITIES, 0.99)

    assert (scores["precision"], scores["recall"], scores["f1"]) == (0, 0, 0)
    assert scores["roc_auc"] == pytest.approx(8 / 9)  # does not depend on the threshold


def test_bootstrap_keeps_the_number_of_failures_fixed() -> None:
    labels = np.array([1] * 10 + [0] * 90)
    rng = np.random.default_rng(0)

    rows = bootstrap_indices(labels, rng)

    assert len(rows) == 100
    assert labels[rows].sum() == 10
    assert set(rows[:10]) <= set(range(10))  # the failures resampled among failures
    assert len(set(rows)) < 100  # sampled with replacement


def test_evaluate_final_scores_the_test_set_with_intervals(make_xy: MakeXY) -> None:
    final, X_test, y_test = fitted_final(make_xy)

    result = evaluate_final(final, X_test, y_test, n_bootstrap=100)

    assert (result.n_rows, result.n_failures) == (120, 12)
    scores = result.scores
    assert scores["tp"] + scores["fn"] == 12
    assert scores["tp"] + scores["fp"] + scores["fn"] + scores["tn"] == 120
    expected = metrics_at_threshold(
        y_test.to_numpy(), final.probability(X_test).to_numpy(), 0.5
    )
    assert scores == expected
    for metric in METRICS:
        low, high = result.intervals[metric]
        assert low <= high
        assert 0 <= low and high <= 1
        assert low <= scores[metric] <= high


def test_the_intervals_are_reproducible_and_depend_on_the_seed(make_xy: MakeXY) -> None:
    final, X_test, y_test = fitted_final(make_xy)

    first = evaluate_final(final, X_test, y_test, n_bootstrap=100)
    again = evaluate_final(final, X_test, y_test, n_bootstrap=100)
    other = evaluate_final(final, X_test, y_test, n_bootstrap=100, random_state=7)

    assert first == again
    assert first.intervals != other.intervals
    assert first.scores == other.scores  # the point estimates do not use the seed


def test_cross_validated_reference_uses_out_of_fold_predictions(
    make_xy: MakeXY,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        final_evaluation, "reference_model", lambda name: cheap_models()[name]
    )
    X, y = make_xy(signal=6.0)

    reference = cross_validated_reference("random_forest", 0.5, X, y)

    assert set(METRICS) <= set(reference)
    assert reference["tp"] + reference["fn"] == int(y.sum())
    assert reference["roc_auc"] > 0.6  # better than chance, with a weak signal


def test_format_shows_the_scores_intervals_and_counts(make_xy: MakeXY) -> None:
    final, X_test, y_test = fitted_final(make_xy)
    result = evaluate_final(final, X_test, y_test, n_bootstrap=50)

    text = format_evaluation(result)

    assert (
        "random_forest at threshold 0.5 on the test set (120 rows, 12 failures)" in text
    )
    assert "95% interval" in text and "cross-validation" not in text
    for metric in METRICS:
        assert metric in text
    assert "correct alerts" in text and "false alarms" in text


def test_format_adds_the_cross_validation_column_when_given_a_reference(
    make_xy: MakeXY,
) -> None:
    final, X_test, y_test = fitted_final(make_xy)
    result = evaluate_final(final, X_test, y_test, n_bootstrap=50)
    reference = dict(result.scores)

    text = format_evaluation(result, reference)

    assert "cross-validation" in text


@pytest.fixture
def saved_model(make_xy: MakeXY, tmp_path: Path) -> Path:
    """A quick k-nearest-neighbours model trained on a synthetic training split."""
    X, y = make_xy(signal=6.0)
    config = TrainingConfig(
        model="knn", tuned=False, threshold=0.5, output_path=tmp_path / "final.joblib"
    )
    pipeline = TrainingPipeline(config)
    X_train, _, y_train, _ = split_train_test(X, y)
    return pipeline.save(pipeline.fit(X_train, y_train))


@pytest.fixture
def synthetic_data(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy(signal=6.0))


def test_the_pipeline_refuses_to_start_without_confirmation(
    monkeypatch: pytest.MonkeyPatch, saved_model: Path
) -> None:
    def must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("nothing may be read without confirmation")

    monkeypatch.setattr(splits, "load_xy", must_not_run)
    monkeypatch.setattr(final_evaluation, "load_final_model", must_not_run)
    pipeline = EvaluationPipeline(EvaluationConfig(model_path=saved_model))

    with pytest.raises(UnconfirmedTestSetError, match="confirm"):
        pipeline.run()
    with pytest.raises(UnconfirmedTestSetError):
        pipeline.run(confirm=False)


@pytest.mark.usefixtures("synthetic_data")
def test_run_scores_the_saved_model_on_the_test_rows(saved_model: Path) -> None:
    config = EvaluationConfig(model_path=saved_model, n_bootstrap=50)

    result = EvaluationPipeline(config).run(confirm=True)

    evaluation = result.evaluation
    assert (evaluation.n_rows, evaluation.n_failures) == (120, 12)  # 20% of 600
    assert evaluation.name == "knn" and evaluation.threshold == 0.5
    assert result.reference is not None and set(METRICS) <= set(result.reference)
    assert result.config is config
    assert "cross-validation" in result.format()


@pytest.mark.usefixtures("synthetic_data")
def test_the_reference_can_be_left_out(saved_model: Path) -> None:
    config = EvaluationConfig(model_path=saved_model, n_bootstrap=20, reference=False)

    result = EvaluationPipeline(config).run(confirm=True)

    assert result.reference is None
    assert "cross-validation" not in result.format()


@pytest.mark.usefixtures("synthetic_data")
def test_the_test_rows_are_the_ones_the_training_pipeline_left_out(
    saved_model: Path, make_xy: MakeXY
) -> None:
    X, y = make_xy(signal=6.0)
    _, X_test, _, y_test = split_train_test(X, y)
    config = EvaluationConfig(model_path=saved_model, n_bootstrap=20, reference=False)

    result = EvaluationPipeline(config).run(confirm=True)

    final = load_final_model(saved_model)
    expected = metrics_at_threshold(
        y_test.to_numpy(), final.probability(X_test).to_numpy(), final.threshold
    )
    assert result.evaluation.scores == expected


@pytest.mark.usefixtures("synthetic_data")
def test_a_missing_model_points_to_the_command_that_creates_it(
    tmp_path: Path,
) -> None:
    config = EvaluationConfig(model_path=tmp_path / "nope.joblib")

    with pytest.raises(FileNotFoundError, match="mf-train"):
        EvaluationPipeline(config).run(confirm=True)


def test_the_reference_model_is_tuned_when_it_can_be() -> None:
    tuned = reference_model("random_forest").get_params()
    default = reference_model("knn").get_params()

    assert tuned["model__max_depth"] == 8
    assert default["model__n_neighbors"] == 15
