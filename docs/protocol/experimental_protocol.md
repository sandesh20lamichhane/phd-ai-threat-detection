# Experimental protocol

Results are reported at four levels, each harder than the last. Only levels 2 to 4 support claims of real-world use.

| Level | Train on | Test on | Shows |
|---|---|---|---|
| 1 Within dataset | CICIDS2017 by day (70/15/15) + 5-fold CV | held-out days | the model can learn the attacks |
| 2 Across datasets | CICIDS2017 | CSE-CIC-IDS2018 (shared features); UNSW-NB15 (mapped) | works on unseen traffic |
| 3 Across time | LANL days 1-40 | LANL days 41-58 | flags compromise early (lead time) |
| 4 Across domains | best model of 1-3 | CICIoMT2024, ToN_IoT, CIC-IoT2023, advisories | holds in government, healthcare, finance |

**Metrics.** Macro-F1, per-class recall, PR-AUC, FPR at a fixed detection rate, detection lead time. Never accuracy alone.
**Response metrics.** Mean time to contain, containment rate, disruption cost; versus rule-based playbook and random agent.
**Statistics.** At least 5 seeds (config/seeds.yaml), mean and std, Wilcoxon signed-rank or Friedman + Nemenyi.
