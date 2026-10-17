"""The pipelines give the same results as the functions they are built from.

Only training rows are used: the test set is evaluated once, by the final
evaluation command, and never by a test.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from machine_failure.config import ComparisonConfig, InferenceConfig, TrainingConfig
from machine_failure.evaluation import compare_models
from machine_failure.experiments.ablation import compare_feature_sets
from machine_failure.models import build_all_models
from machine_failure.pipelines import (
    ComparisonPipeline,
    InferencePipeline,
    TrainingPipeline,
)

QUICK_MODELS = ("dummy", "logistic_regression", "qda")


def test_the_comparison_pipeline_scores_like_compare_models(
    real_data: SimpleNamespace,
) -> None:
    pipeline = ComparisonPipeline(ComparisonConfig(models=QUICK_MODELS))

    result = pipeline.compare(real_data.X_train, real_data.y_train)

    models = build_all_models()
    expected = compare_models(
        {name: models[name] for name in QUICK_MODELS},
        real_data.X_train,
        real_data.y_train,
    )
    pd.testing.assert_frame_equal(result.table, expected)
    assert (result.n_rows, result.n_failures) == (8000, 271)


def test_the_feature_set_comparison_scores_like_compare_models(
    real_data: SimpleNamespace,
) -> None:
    table = compare_feature_sets(
        real_data.X_train,
        real_data.y_train,
        feature_sets=("raw", "computed"),
        models=QUICK_MODELS,
    )

    for feature_set in ("raw", "computed"):
        models = build_all_models(feature_set)
        expected = compare_models(
            {name: models[name] for name in QUICK_MODELS},
            real_data.X_train,
            real_data.y_train,
        )
        columns = [c for c in table.columns if c[0] == feature_set]
        pd.testing.assert_frame_equal(
            table.loc[:, columns].droplevel(0, axis=1), expected
        )


def test_the_inference_pipeline_scores_like_the_final_model(
    real_data: SimpleNamespace, tmp_path: Path
) -> None:
    config = TrainingConfig(output_path=tmp_path / "final.joblib")
    trainer = TrainingPipeline(config)
    trained = trainer.fit(real_data.X_train, real_data.y_train)
    trainer.save(trained)

    scores = InferencePipeline(InferenceConfig(model_path=config.output_path)).predict(
        real_data.X_train
    )

    pd.testing.assert_series_equal(
        scores["probability"], trained.final.probability(real_data.X_train)
    )
    assert int(scores["flagged"].sum()) == 232  # pinned in test_pinned_final_model
    assert scores.index.equals(real_data.X_train.index)
