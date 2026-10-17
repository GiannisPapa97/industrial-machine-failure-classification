"""The pipelines give the same results as the functions they are built from.

Only training rows are used: the test set is evaluated once, by the final
evaluation command, and never by a test.
"""

from types import SimpleNamespace

import pandas as pd

from machine_failure.config import ComparisonConfig
from machine_failure.evaluation import compare_models
from machine_failure.models import build_all_models
from machine_failure.pipelines import ComparisonPipeline

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
