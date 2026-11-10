"""Evaluation helpers for matching quality."""

from .evaluate import CONFIGURATIONS, compare_configurations, evaluate_configuration, evaluate_match_report, format_evaluation_report
from .synthetic_dataset import RELEVANCE_LABELS, generate_synthetic_dataset

__all__ = [
    "CONFIGURATIONS",
    "RELEVANCE_LABELS",
    "compare_configurations",
    "evaluate_configuration",
    "evaluate_match_report",
    "format_evaluation_report",
    "generate_synthetic_dataset",
]
