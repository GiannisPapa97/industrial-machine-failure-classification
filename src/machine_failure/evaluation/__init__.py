"""Cross-validated evaluation: scores, comparisons, thresholds, tuning, importance."""

from machine_failure.evaluation.comparison import (
    METRICS,
    compare_models,
    describe_cross_validation,
    describe_run,
    format_comparison,
)
from machine_failure.evaluation.cross_validation import (
    N_SPLITS,
    SCORING,
    cross_validate_model,
    summarize_scores,
)
from machine_failure.evaluation.interpretation import (
    group_medians,
    missed_by_failure_mode,
    permutation_importances,
)
from machine_failure.evaluation.predictions import out_of_fold_predictions
from machine_failure.evaluation.thresholds import (
    DEFAULT_TARGETS,
    best_f1_per_model,
    best_f1_threshold,
    compare_at_recall_targets,
    format_table,
    metrics_at_threshold,
    plot_precision_recall,
    threshold_curve,
    threshold_for_recall,
)
from machine_failure.evaluation.tuning import (
    MAIN_METRIC,
    SETTING_PREFIX,
    best_settings,
    make_search,
    nested_cross_validate,
    search_results,
    tune_model,
)

__all__ = [
    "DEFAULT_TARGETS",
    "MAIN_METRIC",
    "METRICS",
    "N_SPLITS",
    "SCORING",
    "SETTING_PREFIX",
    "best_f1_per_model",
    "best_f1_threshold",
    "best_settings",
    "compare_at_recall_targets",
    "compare_models",
    "cross_validate_model",
    "describe_cross_validation",
    "describe_run",
    "format_comparison",
    "format_table",
    "metrics_at_threshold",
    "group_medians",
    "make_search",
    "missed_by_failure_mode",
    "nested_cross_validate",
    "out_of_fold_predictions",
    "permutation_importances",
    "plot_precision_recall",
    "search_results",
    "summarize_scores",
    "threshold_curve",
    "threshold_for_recall",
    "tune_model",
]
