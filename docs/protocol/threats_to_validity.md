# Threats to validity

| Type | Threat | Mitigation |
|---|---|---|
| Internal | Duplicate and near-duplicate flows across partitions inflate scores | exact dedup before split; block split with purging; near-duplicate cross-partition rate reported; sensitivity S7 scores without near-duplicates of fitting/tuning data |
| Internal | Outputs built under older rules or altered after the run | dataset config fingerprint checked at every load; `--verify` checks SHA-256 of all outputs and of every S7 sidecar and record |
| Conclusion | CICIDS2017 test holds few attack-bearing 300 s blocks (17 after S7; whole-test count in `near_duplicates.json`), so its block-bootstrap CI rests on few clusters | reported beside H1a; S7 for H1a and the S7 Δ for H1b/H1c reported as not estimable under the fixed minimum |
| Internal | Identifier shortcuts (IPs, ports, timestamps, flow ids) | removed from features; ports tested only as sensitivity S2 |
| Internal | Preprocessing fitted on test or target data | all learned transforms fitted on source train only, hashed and frozen |
| Internal | Label errors in CIC releases | corrected releases; raw label kept; "Attempted" handled by a fixed rule with sensitivity S3 |
| Internal | Analysis choices made after seeing results | protocol frozen 2026-10-25; amendments log; prior exposure disclosed |
| Construct | Macro-F1 depends on class prevalence, which differs between datasets | per-class recall and balanced accuracy reported beside every Δ |
| Construct | CICFlowMeter and Argus features are not the same quantities | mapping table with exact/approximate/unavailable status; sensitivity S5 |
| Construct | Binary task hides which attacks transfer | multiclass family analysis S6 |
| Conclusion | Correlated flows treated as independent | block bootstrap over 300 s blocks, LANL bootstrap over red-team accounts |
| Conclusion | Many comparisons | confirmatory family with Holm; secondary comparisons labelled as such |
| Conclusion | Training variability not in the CI | 5 seeds; seed spread reported |
| External | Lab benchmarks are not production networks | conclusions limited to the datasets used |
| External | CIC-IoT2023 has no time or host fields, so only a within-file order split is possible | stated as a limitation; not used as a source dataset |
| Construct | Same column name does not guarantee same meaning across CIC releases | both corrected releases use the same fixed CICFlowMeter; shared list frozen; UNSW only through the documented mapping, S5 |
