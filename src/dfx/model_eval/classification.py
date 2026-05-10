import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import _check_fitted, _is_classifier
from .regression import evaluate_regression, plot_residuals

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve
)


from sklearn.model_selection import cross_val_score

def evaluate_classification(model, X_test, y_test, average="weighted"):
    _check_fitted(model)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }


def plot_confusion_matrix(model, X_test, y_test):
    _check_fitted(model)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


def plot_roc_curve(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        raise ValueError("ROC curve requires predict_proba().")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()



# -------------------------
# Cross Validation
# -------------------------

def auto_cross_validate(model, X, y, cv=5, scoring=None):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    print(f"Mean Score: {scores.mean():.4f}")
    print(f"Std Dev  : {scores.std():.4f}")

    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "all_scores": scores
    }


# -------------------------
# Auto Evaluation (Best for Beginners)
# -------------------------

def evaluate_model(model, X_test, y_test):
    if _is_classifier(model):
        print("Detected Model Type: Classification\n")
        evaluate_classification(model, X_test, y_test)
        plot_confusion_matrix(model, X_test, y_test)
    else:
        print("Detected Model Type: Regression\n")
        evaluate_regression(model, X_test, y_test)
        plot_residuals(model, X_test, y_test)

# -------------------------
# Correlation Plotting
# -------------------------

def plot_correlation(
    df,
    method="pearson",
    cmap="coolwarm",
    figsize=(8, 6),
    title="Correlation Matrix",
    value_fontsize=8,
    label_fontsize=9,
    show_values=True
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation")

    corr = numeric_df.corr(method=method)

    plt.figure(figsize=figsize)
    plt.imshow(corr, cmap=cmap)
    plt.colorbar()
    plt.title(title)

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90,
        fontsize=label_fontsize
    )
    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
        fontsize=label_fontsize
    )

    if show_values:
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(
                    j, i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=value_fontsize
                )

    plt.tight_layout()
    plt.show()

    return corr