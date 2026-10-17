"""The workflows as configurable, reusable pipelines."""

from machine_failure.pipelines.comparison import ComparisonPipeline, ComparisonResult
from machine_failure.pipelines.training import TrainingPipeline, TrainingResult

__all__ = [
    "ComparisonPipeline",
    "ComparisonResult",
    "TrainingPipeline",
    "TrainingResult",
]
