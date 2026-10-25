# Threats to validity

| Type | Threat | Mitigation |
|---|---|---|
| Internal | Duplicate and near-duplicate flows across partitions inflate scores | exact dedup before split; block split with purging; near-duplicate cross-partition rate reported |
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
| External | CIC-IoT2023 has no time or host fields, so only a random split is possible | stated as a limitation; not used as a source dataset |
