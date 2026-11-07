"""The six evaluation metrics required by the assignment, in one place.

Both train_models.py and app.py call score_model(), so the numbers written into
model/metrics.json and the numbers the Streamlit app computes on an uploaded CSV
are produced by exactly the same code path.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Fixed column order for the comparison table, so every table in the README and
# in the app lists the metrics the same way.
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def decision_scores(model, X) -> np.ndarray:
    """Return a continuous score per row for the positive class, for AUC.

    Most of the classifiers expose predict_proba. SVC does not unless it is
    built with probability=True, which runs an internal 5-fold Platt
    calibration and is far slower than it is worth here - its decision_function
    gives a signed distance from the margin, which ranks the samples just as
    well and so yields the same ROC curve.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise AttributeError(
        f"{type(model).__name__} exposes neither predict_proba nor decision_function"
    )


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Compute the six required metrics for a binary classification result.

    zero_division=0 keeps precision defined if a model predicts no positives at
    all, which a badly tuned classifier on a 15%-positive dataset can do.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def score_model(model, X, y_true) -> dict:
    """Run a fitted model over X and return its six metrics against y_true."""
    y_pred = model.predict(X)
    y_score = decision_scores(model, X)
    return compute_metrics(y_true, y_pred, y_score)


def confusion(y_true, y_pred) -> np.ndarray:
    """Confusion matrix with the label order pinned to [negative, positive].

    Passing labels explicitly stops the matrix from collapsing to 1x1 if an
    uploaded test file happens to contain only one class.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def roc_points(y_true, y_score):
    """False-positive and true-positive rates for plotting an ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr
