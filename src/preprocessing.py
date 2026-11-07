"""Dataset loading, the feature schema, and the shared preprocessing transformer.

Every classifier in this project is wrapped in a Pipeline whose first step is the
transformer built here. Keeping the definition in one module means the training
script and the Streamlit app cannot drift apart in how they treat a column.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"

TARGET = "Revenue"

# Continuous counts and durations. These go through StandardScaler because the
# distance-based (KNN) and margin-based (SVM) models are not scale invariant.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Month, VisitorType and Weekend are obviously categorical. OperatingSystems,
# Browser, Region and TrafficType arrive as integers but they are nominal ID
# codes, not quantities - browser 7 is not "more" than browser 3 - so they are
# one-hot encoded rather than scaled.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV and drop exact duplicate rows.

    The 125 duplicates are dropped before the train/test split: identical rows
    landing on both sides would leak training data into the held-out set and
    flatter every model's reported scores.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/download_data.py` first."
        )

    frame = pd.read_csv(path)
    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before:,} rows, kept {len(frame):,} after dropping duplicates.")

    # Revenue and Weekend come in as booleans; cast the target to int so the
    # metric functions get a clean binary {0, 1} label.
    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def split_data(frame: pd.DataFrame):
    """Stratified 80/20 split, returning X_train, X_test, y_train, y_test.

    The split is stratified because only ~15% of sessions end in a purchase; a
    random split could easily hand the test set a different positive rate and
    make the metrics incomparable between runs.
    """
    X = frame[FEATURES]
    y = frame[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """Scale the numeric columns and one-hot encode the categorical ones.

    handle_unknown="ignore" matters for the Streamlit app: a user-uploaded CSV
    may contain a Browser or TrafficType code that never appeared in training,
    and we want that row scored rather than rejected.

    sparse_output=False keeps the result a dense array, which GaussianNB
    requires - it cannot consume a sparse matrix.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
