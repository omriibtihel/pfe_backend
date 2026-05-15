"""
pipeline/metrics — public re-exports.

All consumers that previously imported from the flat
`app.services.training.pipeline.metrics` module continue to work
unchanged through this package __init__.
"""
from .binary import (
    _detect_classification_type,
    _is_zero_one_label_set,
    get_class_labels,
    get_proba_or_score,
    is_binary,
)
from .classification import (
    classification_metrics,
    compute_classification_metrics,
)
from .curves import (
    compute_calibration_curve,
    compute_multiclass_ovr_curves,
    compute_roc_pr_curves,
)
from .regression import (
    regression_metrics,
    safe_confusion_matrix,
)

__all__ = [
    # binary
    "_detect_classification_type",
    "_is_zero_one_label_set",
    "get_class_labels",
    "get_proba_or_score",
    "is_binary",
    # classification
    "classification_metrics",
    "compute_classification_metrics",
    # curves
    "compute_calibration_curve",
    "compute_multiclass_ovr_curves",
    "compute_roc_pr_curves",
    # regression
    "regression_metrics",
    "safe_confusion_matrix",
]
