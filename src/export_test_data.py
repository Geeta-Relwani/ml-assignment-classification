"""Write the held-out 20% of the dataset to test_data.csv.

This is the file to upload in the Streamlit app. It contains the raw, untouched
feature columns plus the Revenue target - no scaling or encoding - because the
saved pipelines carry their own preprocessing and expect raw input.

It is produced by the same load_dataset/split_data pair that train_models.py
uses, with the same RANDOM_STATE, so these rows are exactly the ones no model
was ever fitted on.

Usage:
    python src/export_test_data.py
"""

from pathlib import Path

from preprocessing import TARGET, load_dataset, split_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_ROOT / "test_data.csv"


def export_test_data() -> Path:
    """Rebuild the held-out split and write it to test_data.csv."""
    frame = load_dataset()
    _, X_test, _, y_test = split_data(frame)

    # Reattach the target so the app can score predictions against it.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test
    test_frame.to_csv(TEST_CSV, index=False)

    print(
        f"Wrote {TEST_CSV.name}: {len(test_frame):,} rows x "
        f"{test_frame.shape[1]} columns ({y_test.mean():.1%} positive)"
    )
    return TEST_CSV


if __name__ == "__main__":
    export_test_data()
