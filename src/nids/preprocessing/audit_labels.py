
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
CHUNK_SIZE = 100_000


def main() -> None:
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR.resolve()}"
        )

    total_labels = Counter()
    total_rows = 0

    for path in files:
        print(f"\nAuditing: {path.name}")
        file_labels = Counter()
        file_rows = 0

        try:
            reader = pd.read_csv(
                path,
                usecols=lambda col: col.strip().lower() == "label",
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk in reader:
                if chunk.shape[1] != 1:
                    raise ValueError(
                        f"Expected one Label column in {path.name}"
                    )

                label_col = chunk.columns[0]
                labels = (
                    chunk[label_col]
                    .astype("string")
                    .str.strip()
                    .fillna("MISSING")
                )

                counts = labels.value_counts(dropna=False)
                file_labels.update(counts.to_dict())
                file_rows += len(chunk)

        except ValueError as exc:
            raise RuntimeError(
                f"Could not read the Label column in {path.name}: {exc}"
            ) from exc

        total_labels.update(file_labels)
        total_rows += file_rows

        print(f"Rows scanned: {file_rows:,}")
        for label, count in file_labels.most_common():
            print(f"  {label}: {count:,}")

    print("\n" + "=" * 60)
    print(f"TOTAL ROWS: {total_rows:,}")
    print("COMBINED LABEL DISTRIBUTION:")

    for label, count in total_labels.most_common():
        percentage = 100 * count / total_rows if total_rows else 0
        print(f"{label}: {count:,} ({percentage:.3f}%)")

    print("\nLabel audit completed.")


if __name__ == "__main__":
    main()
