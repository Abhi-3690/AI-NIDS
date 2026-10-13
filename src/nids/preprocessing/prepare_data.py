
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
CHUNK_SIZE = 100_000
MAX_PER_CLASS = 30_000
RANDOM_STATE = 42


def normalize_label(value: str) -> str:
    label = str(value).strip().lower()

    if label == "benign":
        return "Normal"
    if "ddos" in label:
        return "DDoS"
    if label.startswith("dos "):
        return "DoS"
    if "portscan" in label:
        return "PortScan"
    if "patator" in label or "brute force" in label:
        return "BruteForce"
    if "web attack" in label or "xss" in label or "sql injection" in label:
        return "WebAttack"
    if label == "bot":
        return "Bot"
    if "infiltration" in label:
        return "Infiltration"
    if "heartbleed" in label:
        return "Heartbleed"

    return "Other"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files in {RAW_DIR.resolve()}")

    samples = defaultdict(list)
    rng = np.random.default_rng(RANDOM_STATE)

    for path in files:
        print(f"Reading {path.name}")

        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            chunk.columns = chunk.columns.astype(str).str.strip()

            label_columns = [
                col for col in chunk.columns
                if col.lower() == "label"
            ]
            if len(label_columns) != 1:
                raise ValueError(f"Expected one Label column in {path.name}")

            label_col = label_columns[0]
            labels = chunk[label_col].map(normalize_label)
            features = chunk.drop(columns=[label_col]).copy()

            # Remove duplicate column names after trimming whitespace.
            features = features.loc[:, ~features.columns.duplicated()]

            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors="coerce")

            features = features.replace([np.inf, -np.inf], np.nan)

            # Drop unusable rows and columns with no numeric observations.
            features = features.dropna(axis=1, how="all")
            valid = features.notna().any(axis=1)
            features = features.loc[valid].fillna(np.nan)
            labels = labels.loc[valid]

            # Keep a bounded random reservoir per class.
            for class_name in labels.unique():
                positions = np.flatnonzero(
                    labels.to_numpy() == class_name
                )
                current = samples[class_name]

                records = features.iloc[positions].copy()
                remaining = MAX_PER_CLASS - len(current)

                if remaining <= 0:
                    # Reservoir replacement prevents always favoring
                    # the earliest chunks in each file.
                    for row in records.itertuples(index=False, name=None):
                        j = int(rng.integers(0, len(current) + 1))
                        if j < len(current):
                            current[j] = row
                    continue

                new_rows = list(
                    records.iloc[:remaining].itertuples(
                        index=False, name=None
                    )
                )
                current.extend(new_rows)

                # Reservoir-sample any additional rows from this chunk.
                for row in records.iloc[remaining:].itertuples(
                    index=False, name=None
                ):
                    j = int(rng.integers(0, len(current) + 1))
                    if j < len(current):
                        current[j] = row

    # All feature rows must share the same ordered schema.
    # Read a header once to define the expected columns.
    header = pd.read_csv(files[0], nrows=0)
    header.columns = header.columns.astype(str).str.strip()
    label_col = next(c for c in header.columns if c.lower() == "label")
    feature_columns = [
        c for c in header.columns if c != label_col
    ]
    feature_columns = list(dict.fromkeys(feature_columns))

    frames = []
    for class_name, rows in samples.items():
        frame = pd.DataFrame(rows, columns=feature_columns)
        frame["target"] = class_name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, how="all")

    # Exclude constant columns and duplicate feature columns.
    X = data.drop(columns=["target"])
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    y = data["target"]

    counts = y.value_counts()
    eligible = counts[counts >= 3].index
    keep = y.isin(eligible)
    X, y = X.loc[keep], y.loc[keep]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=RANDOM_STATE, stratify=y_temp
    )

    # Persist CSV splits so the next stage can inspect and reuse them.
    for name, features, labels in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        output = features.copy()
        output["target"] = labels
        output.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(output):,} rows")

    print("\nTraining class counts:")
    print(y_train.value_counts().to_string())
    print(f"\nFeatures: {X.shape[1]}")
    print(f"Saved splits to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
