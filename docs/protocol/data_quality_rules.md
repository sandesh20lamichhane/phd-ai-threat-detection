# Data-quality rules

Applied to every dataset before training by `scripts/preprocess.py`, configured in
`config/preprocessing.yaml`, and checked by `src/data/leakage.py`. Full rationale in
[preregistration.md §4–5](preregistration.md#4-data-preparation-fixed-before-modelling).

- Use corrected labels for CICIDS2017 and CSE-CIC-IDS2018 (Engelen et al. 2021; Liu et al. 2022); keep the raw
  label string next to the canonical label.
- "Attempted" flows: excluded in the primary analysis, relabelled benign in sensitivity S3 (ADR-0003).
- Identifier columns (flow ID, IPs, ports, timestamps, row ids) never enter the feature matrix; time, day and
  destination port are kept as `_`-prefixed audit metadata.
- ±inf → NaN; all-NaN rows dropped; no imputation before the split.
- Exact duplicates (features + label) removed before splitting; conflicting-label duplicates kept one per label and
  flagged; near-duplicates (equal at 3 significant figures) reported, not removed.
- Split before balancing; any balancing on training folds only.
- Every learned transform fitted on the source training partition only.
- Harmonise features only through the documented mappings in `config/feature_mapping/`.
- Bot-IoT replaced by CIC-IoT2023 (ADR-0001).
