"""Train the five required classifiers and persist them to model/.

Each classifier is stored as a complete Pipeline - preprocessor plus estimator -
rather than a bare estimator. That is what lets the Streamlit app score a raw
uploaded CSV directly: the saved object already knows how to scale and encode
its input, so there is no way for the app to apply a different transform from
the one the model was fitted on.

Hyperparameters are chosen by 5-fold stratified cross-validation on the training
split only. The held-out test set is touched exactly once, at the end, to
produce the reported metrics - it never influences model selection.

Usage:
    python src/train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluate import METRIC_ORDER, score_model
from preprocessing import (
    RANDOM_STATE,
    build_preprocessor,
    load_dataset,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
METRICS_JSON = MODEL_DIR / "metrics.json"

# Models are selected on F1 rather than accuracy. Only ~15% of sessions end in a
# purchase, so a model that predicts "no purchase" everywhere already scores 85%
# accuracy while being worthless; F1 forces the search to care about the
# minority class it is actually supposed to find.
SEARCH_SCORING = "f1"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Display name -> (filename stem, estimator, parameter grid). The filename stem
# is what app.py loads, so the two must stay in step; keeping them in one dict is
# the simplest way to guarantee that.
#
# class_weight="balanced" is set wherever the estimator supports it, for the same
# imbalance reason. GaussianNB and KNeighborsClassifier have no equivalent
# parameter, which is itself informative when comparing the results.
MODELS = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # Regularisation strength is the only knob worth searching here; the
        # one-hot matrix is wide enough that too little shrinkage hurts.
        {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # Left unrestricted the tree grows to pure leaves and memorises the
        # training set, so depth and leaf size are what the search decides.
        {
            "classifier__max_depth": [4, 6, 8, 12, None],
            "classifier__min_samples_leaf": [1, 10, 20],
        },
    ),
    "KNN": (
        "knn",
        KNeighborsClassifier(),
        # Small k is risky at a 15% positive rate - a tight neighbourhood is
        # dominated by negatives - so the grid reaches well past the usual k=5.
        {
            "classifier__n_neighbors": [5, 15, 25, 50],
            "classifier__weights": ["uniform", "distance"],
        },
    ),
    "Naive Bayes": (
        "naive_bayes",
        # Gaussian rather than Multinomial: the scaled numeric columns contain
        # negative values, which MultinomialNB cannot accept.
        GaussianNB(),
        # var_smoothing genuinely matters on this dataset. A one-hot column for
        # a rare browser code has almost zero within-class variance, and at the
        # 1e-9 default the Gaussian likelihood blows up and the model predicts
        # the positive class almost everywhere (test accuracy ~0.28). The grid
        # spans the full useful range and CV picks the level.
        {"classifier__var_smoothing": np.logspace(-9, 0, 10)},
    ),
    "Support Vector Machine": (
        "svm",
        SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # Kept deliberately small: an RBF SVM is the slowest model here, and
        # each extra candidate costs five more fits on ~9,800 rows.
        {"classifier__C": [0.5, 1.0, 5.0]},
    ),
}


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator behind a freshly built preprocessor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def train_all():
    """Fit every model, score it on the held-out split, and save the artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_dataset()
    X_train, X_test, y_train, y_test = split_data(frame)
    print(
        f"Train: {len(X_train):,} rows ({y_train.mean():.1%} positive) | "
        f"Test: {len(X_test):,} rows ({y_test.mean():.1%} positive)"
    )

    results = {}
    chosen_params = {}
    for display_name, (stem, estimator, grid) in MODELS.items():
        print(f"\nTraining {display_name} ...")
        search = GridSearchCV(
            build_pipeline(estimator),
            param_grid=grid,
            scoring=SEARCH_SCORING,
            cv=CV,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        # best_estimator_ is itself a refitted Pipeline, so what gets saved is a
        # plain preprocessor+classifier object - no search machinery is pickled.
        pipeline = search.best_estimator_
        metrics = score_model(pipeline, X_test, y_test)
        results[display_name] = metrics

        # Record the chosen settings with the estimator prefix stripped, so the
        # README can quote them without the pipeline plumbing.
        chosen_params[display_name] = {
            key.replace("classifier__", ""): _jsonable(value)
            for key, value in search.best_params_.items()
        }

        destination = MODEL_DIR / f"{stem}.joblib"
        joblib.dump(pipeline, destination)
        print(f"  best params: {chosen_params[display_name]}")
        print(f"  CV {SEARCH_SCORING}: {search.best_score_:.4f}")
        print(f"  saved -> {destination.relative_to(PROJECT_ROOT)}")
        print("  " + "  ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_ORDER))

    # Persist the held-out scores alongside the models so the README table and
    # the app's "reference metrics" come from the recorded run, not a re-fit.
    payload = {
        "dataset": "UCI Online Shoppers Purchasing Intention",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "search_scoring": SEARCH_SCORING,
        "best_params": chosen_params,
        "metrics": results,
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {METRICS_JSON.relative_to(PROJECT_ROOT)}")

    table = pd.DataFrame(results).T[METRIC_ORDER]
    print("\nHeld-out test set comparison:")
    print(table.round(4).to_string())
    return table


def _jsonable(value):
    """Convert a NumPy scalar from the grid into something json can write."""
    return value.item() if hasattr(value, "item") else value


if __name__ == "__main__":
    train_all()
