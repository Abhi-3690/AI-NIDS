# AI-NIDS — AI-Based Network Intrusion Detection System

An experimental network-flow intrusion detection system built with Python, PyTorch, Scikit-learn, Pandas, and FastAPI. It classifies CICIDS2017 network-flow features into Normal and attack categories.

## Features

- CICIDS2017 CSV ingestion and label auditing
- Chunk-based dataset inspection
- Numeric feature preprocessing and standardization
- Multiclass PyTorch neural network
- Class-weighted training and early stopping
- Precision, recall, F1-score, and confusion matrix evaluation
- FastAPI inference endpoint with feature-schema validation
- Automated API integration tests

## Technology Stack

- Python 3.12
- PyTorch
- Scikit-learn
- Pandas and NumPy
- FastAPI and Uvicorn
- Scapy for future packet-processing integration
- Pytest

## Project Structure

```text
data/
  raw/
  processed/
artifacts/
src/nids/
  ingestion/
  features/
  preprocessing/
  models/
  inference/
  alerting/
tests/
  unit/
  integration/
```

Dataset files and trained model artifacts are kept outside Git version control.

## Setup

Create and activate a Python virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download CICIDS2017 from the official Canadian Institute for Cybersecurity dataset page and place its CSV files in `data/raw/`.

## Data Preparation

Inspect the CSV structure:

```powershell
python src/nids/preprocessing/inspect_dataset.py
```

Audit attack labels:

```powershell
python src/nids/preprocessing/audit_labels.py
```

Prepare the training, validation, and test splits:

```powershell
python src/nids/preprocessing/prepare_data.py
```

## Model Training

```powershell
python src/nids/models/train_model.py
```

The training process saves the model checkpoint, fitted preprocessor, metadata, and evaluation results under `artifacts/`.

## Run the API

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn nids.inference.api:app --host 127.0.0.1 --port 8001
```

Open the interactive documentation at `http://127.0.0.1:8001/docs`.

Check service health at `http://127.0.0.1:8001/health`.

## Run Tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -v
```

## Evaluation and Limitations

The initial model achieved approximately 98% test accuracy, 0.87 macro F1-score, and a 4% normal-class false-positive rate on its current sampled test split. These are preliminary results, not production performance guarantees.

The dataset contains severe class imbalance. Rare classes such as Infiltration and Heartbleed have too few examples to support reliable performance claims. Evaluation on independent capture sessions and representative benign enterprise traffic is required.

The current API accepts numeric flow features; it does not yet implement live packet capture, complete flow aggregation, enterprise alert delivery, or production authentication.

## Security

Run locally by default. Do not expose the API to untrusted networks without authentication, rate limiting, secure deployment, access controls, and monitoring.

## License

See the repository's LICENSE file.
