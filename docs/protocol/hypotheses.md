# Hypotheses (frozen 2026-10-25, protocol v1.0)

X, Y and Z were set on 2026-10-25, before any Level 2, 3 or 4 run of this study. Any later change goes in
[protocol_amendments.md](protocol_amendments.md) with a date and reason. Estimands, tests and decision rules are
in [preregistration.md](preregistration.md); the prior use of these datasets by the author is disclosed in
[prior_exposure.md](prior_exposure.md).

| | Hypothesis | Threshold | Primary estimand |
|---|---|---|---|
| H1a | The proposed detector reaches macro-F1 >= 0.95 at Level 1. | 0.95 | macro-F1 on the CICIDS2017 test partition, binary task, mean over 5 seeds |
| H1b | It degrades by less than **X = 0.05** macro-F1 from CICIDS2017 to CSE-CIC-IDS2018. | X = 0.05 | Δ = F1(source test) − F1(target), shared CICFlowMeter features |
| H1c | It degrades by less than **X = 0.05** macro-F1 from CICIDS2017 to UNSW-NB15. | X = 0.05 | same Δ, canonical mapped features |
| H2 | On LANL the predictive model alerts before the red-team event for at least **Y = 50 %** of compromised hosts, at a false-alarm rate of at most 1 % of host-days. | Y = 0.50, FA <= 0.01 | proportion of test-period compromised hosts with an alert in the 24 h before their first compromise |
| H3 | The RL response agent contains attacks faster than the rule-based playbook at equal or lower disruption cost. | (no margin) | difference in mean time to contain; non-inferiority of disruption cost (margin set before the first response run) |
| H4 | Performance at Level 4 stays within **Z = 0.05** macro-F1 of Level 2 in each of healthcare (CICIoMT2024), IoT (CIC-IoT2023) and government/industrial (ToN_IoT) settings. | Z = 0.05 | Δ4 = F1(Level 2, CIC 2017 → 2018) − F1(Level 4 target) |

## Decision rules (three outcomes)

Every threshold claim ends in one of three preregistered outcomes, read from the 95 % confidence interval of the
estimand (group-aware block bootstrap, [preregistration.md §6](preregistration.md#6-statistical-analysis)):

| Claim | Supported | Not supported | Inconclusive |
|---|---|---|---|
| H1a: F1 >= 0.95 | lower bound >= 0.95 | upper bound < 0.95 | CI contains 0.95 |
| H1b, H1c: generalisation preserved (Δ < X) | upper bound < 0.05 | lower bound > 0.05 (meaningful degradation) | CI contains 0.05 |
| H2: early-alert rate >= Y | lower bound >= 0.50, and above the random-alert baseline | upper bound < 0.50 | CI contains 0.50 |
| H4: Level 4 within Z of Level 2 (per domain) | upper bound of Δ4 < 0.05 | lower bound of Δ4 > 0.05 | CI contains 0.05 |

An observed drop above 0.05 is not by itself called a failure to generalise; it is "meaningful degradation" only
when the whole CI lies above 0.05. The thresholds are proposed practical-importance margins for this study, not
community standards, and are reported as such.

## Why these values

- **X = Z = 0.05 macro-F1** (five percentage points): the smallest change in a balanced error measure that would
  alter a deployment decision for a SOC, and about twice the seed-to-seed spread usually reported for tree
  ensembles on these benchmarks. Using the same margin for H1 and H4 keeps the two transfer claims comparable.
- **Y = 50 % at <= 1 % false alarms per host-day**: a predictive model that warns about half of compromises
  before they happen, at a budget of about one false alarm per hundred hosts per day, is operationally useful;
  below half it is no better than a coin-flip triage aid. The false-alarm budget is what stops "alert on
  everything" from passing.
- **No severe-degradation threshold (0.10)** is used for a decision: the hypotheses file needs only one margin.
  The 0.10 band is reported descriptively.
- **95 % CI, alpha = 0.05 (two-sided; 0.025 per direction)**, Holm across the confirmatory family.
