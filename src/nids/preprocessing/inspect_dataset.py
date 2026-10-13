
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")


def main() -> None:
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR.resolve()}"
        )

    print(f"Found {len(files)} CSV files\n")

    for path in files:
        print("=" * 80)
        print(f"FILE: {path.name}")
        print(f"SIZE: {path.stat().st_size / (1024 * 1024):.1f} MB")

        # Read only a small sample; do not load the entire file.
        sample = pd.read_csv(path, nrows=5, low_memory=False)

        # CICIDS2017 headers often contain surrounding whitespace.
        sample.columns = sample.columns.astype(str).str.strip()

        print("\nColumns:")
        for column in sample.columns:
            print(f"  - {column}")

        label_columns = [
            column for column in sample.columns
            if column.strip().lower() == "label"
        ]

        if label_columns:
            label_column = label_columns[0]
            print("\nSample labels:")
            print(
                sample[label_column]
                .astype(str)
                .value_counts(dropna=False)
                .to_string()
            )
        else:
            print("\nWARNING: No column named 'Label' found.")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
