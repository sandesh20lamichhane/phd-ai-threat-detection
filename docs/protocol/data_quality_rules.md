# Data-quality rules

Applied to every dataset before training (implemented in src/data/clean.py, checked by src/data/leakage.py).

- Use corrected labels for CICIDS2017 and CSE-CIC-IDS2018 (Engelen et al. 2021; Liu et al. 2022).
- Remove identifier columns: Flow ID, IPs, ports, timestamps.
- Remove exact duplicates before splitting.
- Split before balancing; SMOTE or undersampling on training folds only.
- Harmonise features with one documented mapping (config/feature_mapping/).
- Record why Bot-IoT was dropped (docs/decisions/ADR-0001-drop-bot-iot.md).
