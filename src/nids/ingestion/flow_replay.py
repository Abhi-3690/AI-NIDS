
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
METADATA_PATH = ROOT / "artifacts" / "metadata.json"
ALERTS_PATH = ROOT / "artifacts" / "alerts.jsonl"

BENIGN_LABELS = {"BENIGN", "NORMAL"}


def load_feature_columns(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {path}. Train the model first."
        )

    metadata = json.loads(path.read_text(encoding="utf-8"))
    columns = metadata.get("feature_columns")

    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "metadata.json must contain a non-empty feature_columns list."
        )

    if not all(isinstance(name, str) for name in columns):
        raise ValueError("Feature names must be strings.")

    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate feature names in metadata.")

    return columns


def predict_flow(
    features: dict[str, float],
    api_url: str,
    timeout: float,
) -> dict:
    payload = json.dumps({"features": features}).encode("utf-8")
    request = Request(
        api_url.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Prediction API returned HTTP {exc.code}: {body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to prediction API: {exc}"
        ) from exc


def replay_csv(
    csv_path: Path,
    api_url: str,
    timeout: float,
    limit: int | None,
    delay: float,
    alerts_path: Path,
) -> None:
    feature_columns = load_feature_columns(METADATA_PATH)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    failed = 0
    alert_count = 0

    with csv_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as source, alerts_path.open(
        "a", encoding="utf-8"
    ) as alert_file:
        
        reader = csv.reader(source)
        raw_headers = next(reader, None)

        if not raw_headers:
            raise ValueError("CSV has no header row.")

        # Normalize headers and preserve duplicate columns using
        # pandas-compatible names such as "Feature" and "Feature.1".
        headers = []
        seen = set()
        counts = {}

        for raw_header in raw_headers:
            base = raw_header.strip()
            count = counts.get(base, 0)
            name = base if count == 0 else f"{base}.{count}"

            while name in seen:
                count += 1
                name = f"{base}.{count}"

            headers.append(name)
            seen.add(name)
            counts[base] = count + 1

        header_index = {
            name: index for index, name in enumerate(headers)
        }

        missing = sorted(set(feature_columns) - set(header_index))

        if missing:
            raise ValueError(
                "CSV does not match the model's feature schema. "
                f"Missing columns: {missing[:15]}"
            )



        for row_number, row in enumerate(reader, start=2):
            if limit is not None and processed >= limit:
                break

            features: dict[str, float] = {}
            try:
                for name in feature_columns:
                    raw_value = (
                        row[header_index[name]]
                        if header_index[name] < len(row)
                        else ""
                    )
                    if raw_value is None or not raw_value.strip():
                        raise ValueError(f"Empty feature: {name}")

                    value = float(raw_value.strip())

                    if not math.isfinite(value):
                        raise ValueError(
                            f"Non-finite value in feature: {name}"
                        )

                    features[name] = value

                result = predict_flow(features, api_url, timeout)
                predicted = str(result["predicted_class"])
                confidence = float(result["confidence"])

                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("API returned invalid confidence.")

                processed += 1

                print(
                    f"Row {row_number}: {predicted} "
                    f"(confidence={confidence:.3f})"
                )

                if predicted.strip().upper() not in BENIGN_LABELS:
                    alert = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source_file": csv_path.name,
                        "source_row": row_number,
                        "predicted_class": predicted,
                        "confidence": confidence,
                        "probabilities": result.get("probabilities", {}),
                    }
                    alert_file.write(json.dumps(alert) + "\n")
                    alert_file.flush()
                    alert_count += 1

                if delay:
                    time.sleep(delay)

            except (ValueError, KeyError, RuntimeError) as exc:
                failed += 1
                print(f"Row {row_number} failed: {exc}")

    print("\nReplay summary")
    print(f"Successfully classified: {processed}")
    print(f"Failed rows:             {failed}")
    print(f"Alerts written:          {alert_count}")
    print(f"Alert log:               {alerts_path}")

    if failed:
        print(
            "Some rows were skipped. Review the errors before relying "
            "on the replay results."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay CICIDS-compatible network-flow CSV records."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8001",
        help="Base URL of the running AI-NIDS API.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows to replay; use 0 for all rows.",
    )
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--alerts", type=Path, default=ALERTS_PATH)
    args = parser.parse_args()

    if args.timeout <= 0 or args.delay < 0:
        parser.error("timeout must be positive and delay non-negative.")

    if args.limit < 0:
        args.limit = None

    replay_csv(
        csv_path=args.csv,
        api_url=args.api,
        timeout=args.timeout,
        limit=args.limit,
        delay=args.delay,
        alerts_path=args.alerts,
    )


if __name__ == "__main__":
    main()
