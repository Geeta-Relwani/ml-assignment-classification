"""Fetch the Online Shoppers Purchasing Intention dataset from the UCI repository.

The dataset is distributed as a zip archive containing a single CSV. Running this
module writes that CSV to data/online_shoppers_intention.csv so that the rest of
the project never has to touch the network.

Usage:
    python src/download_data.py
"""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

# UCI dataset 468 - Online Shoppers Purchasing Intention Dataset.
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)

# Project root is the parent of src/, so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "online_shoppers_intention.csv"


def download_dataset(force: bool = False) -> Path:
    """Download and extract the dataset, returning the path to the raw CSV.

    If the CSV is already present the download is skipped unless force=True.
    """
    if RAW_CSV.exists() and not force:
        print(f"Dataset already present at {RAW_CSV}, skipping download.")
        return RAW_CSV

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {DATA_URL} ...")
    with urlopen(DATA_URL, timeout=120) as response:
        archive_bytes = response.read()

    # The archive holds one CSV; pull it out by extension rather than by name so
    # a future rename upstream does not break this.
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected exactly one CSV in the archive, found {csv_names}")
        RAW_CSV.write_bytes(archive.read(csv_names[0]))

    print(f"Wrote {RAW_CSV} ({RAW_CSV.stat().st_size:,} bytes)")
    return RAW_CSV


if __name__ == "__main__":
    path = download_dataset()

    # Print a quick shape check so the download is visibly correct when run.
    import pandas as pd

    frame = pd.read_csv(path)
    print(f"Shape: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"Target distribution:\n{frame['Revenue'].value_counts()}")
