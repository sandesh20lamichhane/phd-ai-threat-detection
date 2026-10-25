# Protocol amendments

Dated log of every change to the protocol after it was frozen, with the reason and whether any result had been
seen at the time.

| Date | Version | Change | Reason | Results seen? |
|---|---|---|---|---|
| 2026-10-25 | 1.0 | Protocol frozen: X = 0.05, Y = 50 % at <= 1 % FA per host-day, Z = 0.05; three-outcome CI rules; binary primary task; block bootstrap; Holm family (preregistration.md). | Initial freeze, before any Level 2–4 run. | No confirmatory results exist. |
| 2026-10-25 | 1.0 | Level 1 split changed from by-day (Mon–Wed / Thu / Fri) to within-day chronological 300 s block split; by-day kept as sensitivity S1. | The by-day split puts only unseen attack types (Botnet, PortScan, DDoS) in the test set, so H1a would measure novel-attack detection, not within-dataset learning. | No. |
