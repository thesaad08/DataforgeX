import numpy as np
def _is_classifier(model):
    return hasattr(model, "predict_proba") or hasattr(model, "classes_")


def _check_fitted(model):
    try:
        model.predict(np.zeros((1, model.n_features_in_)))
    except Exception:
        raise ValueError("Model is not fitted. Train the model first.")