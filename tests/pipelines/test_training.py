from machine_failure.config import TrainingConfig
from machine_failure.models import FinalModel
from machine_failure.pipelines import TrainingResult


def test_describe_says_what_was_trained() -> None:
    final = FinalModel("random_forest", object(), 0.475)
    result = TrainingResult(final, TrainingConfig(), n_rows=8000, n_failures=271)

    assert result.describe() == (
        "Trained random_forest on the training set (8,000 rows, 271 failures); "
        "threshold 0.475."
    )
