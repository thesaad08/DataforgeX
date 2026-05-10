from .classification import (
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_curve,
    auto_cross_validate,
    evaluate_model,
    plot_correlation
)

from .regression import (
    evaluate_regression,
    plot_residuals
)

from .utils import (
    _is_classifier,
    _check_fitted
)

__all__ = [
    "evaluate_model",
    "evaluate_classification",
    "evaluate_regression",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_residuals",
    "auto_cross_validate",
    "_is_classifier",
    "_check_fitted",
    "plot_correlation"
]
