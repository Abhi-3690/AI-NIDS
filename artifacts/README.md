# AI-NIDS Model Artifacts

This directory stores locally generated model artifacts and runtime outputs.

Generated files include:

- `nids_model.pt` — trained PyTorch model checkpoint
- `preprocessor.joblib` — fitted preprocessing pipeline
- `metadata.json` — feature schema and class metadata
- `evaluation.json` — evaluation metrics
- `alerts.jsonl` — runtime prediction alerts

These files are generated locally and are excluded from Git by default. Run the documented data preparation and training commands to regenerate the model artifacts.

Only use model artifacts produced by a trusted training workflow.
