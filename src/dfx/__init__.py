__version__ = "0.1.5"

# Cleaning
from .cleaning import (
    suggest_fill_strategy,
    handle_missing_values,
)

# Model Evaluation
from .model_eval import (
    evaluate_model,
    evaluate_classification,
    evaluate_regression,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_residuals,
    auto_cross_validate,
    plot_correlation,
)

# Preprocessing
from .preprocessing import (
    detect_outliers,
    auto_encode,
    auto_fix_dtypes,
    remove_outliers,
    cap_outliers,
    scale_data,
)

__all__ = [
    # Cleaning
    "suggest_fill_strategy",
    "handle_missing_values",

    # Evaluation
    "evaluate_model",
    "evaluate_classification",
    "evaluate_regression",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_residuals",
    "auto_cross_validate",
    "plot_correlation",

    # Preprocessing
    "detect_outliers",
    "auto_encode",
    "auto_fix_dtypes",
    "remove_outliers",
    "cap_outliers",
    "scale_data",
]