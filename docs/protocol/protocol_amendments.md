# Protocol amendments

**Author:** Sandesh Lamichhane, Assistant Professor, Yenepoya (Deemed to be University)

Dated log of every change to the protocol after it was frozen, with the reason and whether any result had been
seen at the time.

| Date | Version | Change | Reason | Results seen? |
|---|---|---|---|---|
| 2026-10-25 | 1.0 | Protocol frozen: X = 0.05, Y = 50 % at <= 1 % FA per host-day, Z = 0.05; three-outcome CI rules; binary primary task; block bootstrap; Holm family (preregistration.md). | Initial freeze, before any Level 2–4 run. | No confirmatory results exist. |
| 2026-10-25 | 1.0 | Level 1 split changed from by-day (Mon–Wed / Thu / Fri) to within-day chronological 300 s block split; by-day kept as sensitivity S1. | The by-day split puts only unseen attack types (Botnet, PortScan, DDoS) in the test set, so H1a would measure novel-attack detection, not within-dataset learning. | No. |
| 2026-10-25 | 1.0 | Added author line, the no-shortcut rule (§10: full data, 5 seeds, B = 10 000; pilot outputs refused by the loader) and the records rule (§11: every record in both Drive and GitHub, run logs kept). | Author's standing instructions. Tightens the protocol; no analysis choice changed. | No. |
| 2026-10-25 | 1.1 | After the schema profile (no model trained): (a) CIC-IoT2023 labels read from the class folder (this release has no label column) and its split changed from stratified random to within-file order 70/15/15; (b) CICIoMT2024 validation taken from the end of each train capture instead of a random 15 %; (c) block strata use the raw attack label, so each attack variant (e.g. DoS Hulk vs GoldenEye) is split in time, not just each family; (d) ToN_IoT `uid` (Zeek connection id, present only in Network_dataset_6) treated as an identifier; (e) files read in natural order. | (a, d) schema facts the profile revealed; (b, c) the CICIoMT2024 report showed 53 % of validation rows near-duplicating a training row under random carving, and the CICIDS2017 report showed whole DoS variants landing only in val or test under family strata. All make the evaluation stricter. | No. |
| 2026-10-25 | 1.1 | LANL: the profile shows red-team events only on days 2–30 (0 compromised hosts in days 41–58), so the contingency rule of §7, fixed before the profile was run, applies: train days 1–10, val 11–13, test 14–58 (44 compromised hosts in test). Not a change to the protocol. | Rule applied as preregistered. | No. |
