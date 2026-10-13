
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

SEED = 42
BATCH_SIZE = 512
EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_data.py first."
        )

    frame = pd.read_csv(path, low_memory=False)
    if "target" not in frame.columns:
        raise ValueError(f"'target' column missing from {path.name}")

    y = frame.pop("target").astype(str).str.strip()
    X = frame.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.shape[1] == 0:
        raise ValueError(f"No feature columns found in {path.name}")

    return X, y


class NIDSClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            predictions = logits.argmax(dim=1).cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(labels.numpy())

    return np.concatenate(all_targets), np.concatenate(all_predictions)


def main() -> None:
    set_seed()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    # Ensure identical feature names and order across all splits.
    columns = X_train.columns.tolist()
    if not columns:
        raise ValueError("Training data has no feature columns.")

    for name, frame in [
        ("validation", X_val),
        ("test", X_test),
    ]:
        missing = sorted(set(columns) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(columns))
        if missing or extra:
            raise ValueError(
                f"{name} schema differs from training. "
                f"Missing={missing}; extra={extra}"
            )

    X_val = X_val[columns]
    X_test = X_test[columns]

    # Learn imputation and scaling on training data only.
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_train_np = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_np = preprocessor.transform(X_val).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)

    for name, array in [
        ("train", X_train_np),
        ("validation", X_val_np),
        ("test", X_test_np),
    ]:
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite features in {name} data.")

    # All target classes must be present in training.
    classes = sorted(y_train.unique().tolist())
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    for name, labels in [
        ("validation", y_val),
        ("test", y_test),
    ]:
        unseen = sorted(set(labels) - set(classes))
        if unseen:
            raise ValueError(
                f"Unseen labels in {name} split: {unseen}"
            )

    y_train_np = y_train.map(class_to_id).to_numpy(dtype=np.int64)
    y_val_np = y_val.map(class_to_id).to_numpy(dtype=np.int64)
    y_test_np = y_test.map(class_to_id).to_numpy(dtype=np.int64)

    class_ids = np.arange(len(classes))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=y_train_np,
    )

    # Cap extreme weights to reduce instability for very rare classes.
    weights = np.clip(weights, 0.25, 10.0).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training rows: {len(y_train_np):,}")
    print(f"Validation rows: {len(y_val_np):,}")
    print(f"Test rows: {len(y_test_np):,}")
    print(f"Input features: {X_train_np.shape[1]}")
    print(f"Classes: {classes}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train_np),
        torch.from_numpy(y_train_np),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_np),
        torch.from_numpy(y_val_np),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_np),
        torch.from_numpy(y_test_np),
    )

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    model = NIDSClassifier(
        input_dim=X_train_np.shape[1],
        num_classes=len(classes),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_path = ARTIFACT_DIR / "nids_model.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        seen = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            loss_sum += loss.item() * len(labels)
            seen += len(labels)

        val_true, val_pred = evaluate(model, val_loader, device)
        val_macro_f1 = f1_score(
            val_true,
            val_pred,
            average="macro",
            zero_division=0,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={loss_sum / max(seen, 1):.4f} | "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train_np.shape[1],
                    "classes": classes,
                    "best_epoch": best_epoch,
                    "validation_macro_f1": best_f1,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        best_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Save preprocessing and schema alongside the model.
    joblib.dump(
        preprocessor, ARTIFACT_DIR / "preprocessor.joblib"
    )

    metadata = {
        "classes": classes,
        "feature_columns": columns,
        "input_dim": len(columns),
        "best_epoch": best_epoch,
        "validation_macro_f1": best_f1,
        "device_used_for_training": str(device),
        "seed": SEED,
        "class_weights": weights.tolist(),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Evaluate the held-out test split only after model selection.
    test_true, test_pred = evaluate(model, test_loader, device)

    report = classification_report(
        test_true,
        test_pred,
        labels=class_ids,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    print("\nTEST CLASSIFICATION REPORT")
    print(
        classification_report(
            test_true,
            test_pred,
            labels=class_ids,
            target_names=classes,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(test_true, test_pred, labels=class_ids))

    normal_id = class_to_id.get("Normal")
    if normal_id is not None:
        benign_mask = test_true == normal_id
        false_positives = int(
            np.sum((test_pred != normal_id) & benign_mask)
        )
        true_negatives = int(
            np.sum((test_pred == normal_id) & benign_mask)
        )
        fpr = false_positives / max(
            false_positives + true_negatives, 1
        )
        print(f"Normal-class false-positive rate: {fpr:.6f}")

    results = {
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            test_true, test_pred, labels=class_ids
        ).tolist(),
        "macro_precision": precision_score(
            test_true, test_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            test_true, test_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            test_true, test_pred, average="macro", zero_division=0
        ),
        "test_rows": int(len(test_true)),
    }

    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved model: {best_path}")
    print(f"Saved evaluation: {ARTIFACT_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
