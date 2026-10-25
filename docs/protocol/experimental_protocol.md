# Experimental protocol

Results are reported at four levels, each harder than the last. Only levels 2 to 4 support claims of real-world
use. The frozen details (tasks, splits, statistics, decision rules) are in [preregistration.md](preregistration.md)
and the thresholds in [hypotheses.md](hypotheses.md).

| Level | Train on | Test on | Shows |
|---|---|---|---|
| 1 Within dataset | CICIDS2017, earliest 70 % of each day's 300 s blocks per attack family (15 % val) | latest 15 % of blocks; by-day split as sensitivity S1 | the model can learn the attacks |
| 2 Across datasets | CICIDS2017 train partition | CSE-CIC-IDS2018 (shared CICFlowMeter features); UNSW-NB15 (canonical mapped features) | works on unseen traffic |
| 3 Across time | LANL days 1-40 (contingency rule in preregistration §7) | LANL days 41-58 | flags compromise early (lead time) |
| 4 Across domains | see preregistration §9 (design to be fixed before the first run) | CICIoMT2024, ToN_IoT, CIC-IoT2023 | holds in healthcare, IoT and government settings |

**Primary task.** Binary benign vs attack; multiclass family task as sensitivity S6.
**Metrics.** Macro-F1 (primary), per-class precision/recall/F1, balanced accuracy, FPR, PR-AUC, FPR at a fixed
detection rate, detection lead time. Never accuracy alone.
**Response metrics.** Mean time to contain, containment rate, disruption cost; versus rule-based playbook and random agent.
**Statistics.** 5 seeds (config/seeds.yaml); block bootstrap 95 % CIs (B = 10 000); three-outcome decision rules;
Holm across the confirmatory family; Wilcoxon or Friedman + Nemenyi for secondary model comparisons.
