# Data audit

Written by `scripts/preprocess.py` (protocol v1.0, `config/preprocessing.yaml`):

- `schema_profile.{json,md}`: headers, labels and timestamps of every dataset checked against the config (samples).
- `<dataset>/report.json`: everything the data card shows, plus input and output SHA-256, config SHA-256 and the
  code commit. Data cards in `docs/data_cards/` are rendered from these files.
- `cross_dataset_overlap.json`: share of external-target rows identical (on the shared features) to a CICIDS2017
  training row.
