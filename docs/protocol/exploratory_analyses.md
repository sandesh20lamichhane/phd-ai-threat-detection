# Exploratory analyses

**Author:** Sandesh Lamichhane, Assistant Professor, Yenepoya (Deemed to be University)

Analyses planned after the confirmatory results were seen. None of them is a hypothesis test of this study: they
sit outside the Holm family of preregistration §6, carry no p-values, and cannot change any confirmatory decision.
Each plan is written and committed here before its analysis is run, and the thesis and paper label every one of
them exploratory (preregistration §12).

## E1. Ranking versus operating point of the frozen proposed detector

*Planned 2026-10-27, after the one-time test (notebook 08, commit 9630eeb). Results seen when planning:
everything in `results/confirmatory/` (H1a inconclusive, H1b, H1c and H2 not supported).*

**Why.** An external review of the confirmatory results pointed out that the random forest still ranks target
attacks above benign rows better than chance (PR-AUC 0.238 on CSE-CIC-IDS2018 and 0.236 on UNSW-NB15, against
attack shares of 0.077 and 0.044), while its frozen threshold finds 8.7 % and 31.3 % of target attacks. The per-family
recall in `level1_2.json` puts most of the CSE-CIC-IDS2018 failure in two families (DoS 0.006 and DDoS 0.128,
together 92.7 % of its attack rows). E1 describes where the transfer fails: in the ranking, in the threshold, or
in particular attack families. The question is close to Jin and Lee (2026, *Electronics* 15(17):3761), who report
ranking, threshold transfer and alert budgets for source-only transfer between HIKARI-2021, CICIDS2017 and
CIC-IoT2023; E1 asks it for this study's frozen pipeline and pairs.

**What is analysed.** The frozen proposed detector (random forest), all five seeds, on the four (evaluation set,
feature space) pairs scored in notebook 08: the CICIDS2017 test partition in the `cic` and `unsw_canonical` spaces
(the in-domain references), CSE-CIC-IDS2018 (`cic`) and UNSW-NB15 (`unsw_canonical`). Primary rows only. Only
the proposed detector has per-row scores, so the baselines are not part of E1.

**Nothing is re-scored or refitted.** E1 reads the per-row scores written once in notebook 08
(`<Drive>/scores/<eval>/<space>/random_forest/seed<s>.parquet`) and the labels, attack families and 300 s blocks
of the same rows. Before any figure is computed it checks, and stops if any check fails:

1. the Levels 1-2 freeze is intact (`seal.require_frozen("l12")`);
2. every scores file matches the SHA-256 in its unit record, and the unit was scored with the frozen model file;
3. the rows read now are the scored rows, in the same order (`_key` equal row for row);
4. the rows reproduce the confirmatory groups file exactly (row count of every block x family x label x S7 group);
5. alerting at the frozen threshold reproduces the alerted-row count of every group in the unit record exactly;
6. PR-AUC and the FPR at 95 % detection reproduce the unit record. The scores are stored as float32, the unit
   record was computed from float64, so these two are compared within 1e-6 and the largest difference is reported.

**Measures** (alert when score >= t; per seed, then the mean over seeds):

| Group | Measure | Definition |
|---|---|---|
| Ranking | PR-AUC | average precision (step-wise, as scikit-learn) |
| | ROC-AUC | ties count one half |
| | prevalence, AP-lift, lift ratio | attack share π; PR-AUC − π; PR-AUC / π |
| Frozen threshold | recall, FPR, precision, alert share | at the frozen val threshold of each seed (as scored) |
| Oracle FPR points | recall and precision at FPR = 0.1 %, 1 %, 5 % | alert on every row scored above the cut-off value and on the share of the rows at the cut-off value that makes the benign alert rate exactly the level (ties split in proportion: the straight segment of the ROC curve between two thresholds, the convention under which ROC-AUC counts ties one half). **Uses target labels: an upper bound no deployed detector could choose.** |
| Alert budgets | recall, FPR, precision at the top 1 % and 5 % of rows | the same rule with the alert share of all rows exactly at the budget. Uses no labels. |
| Ties | FPR at 95 % detection | as in the unit records (first threshold reaching 95 %, ties not split), and with ties at the cut-off split in proportion |
| | attack rows undetectable below full FPR | share of attack rows scored at or below the lowest-scored benign rows: they are alerted only if every row is |
| | rows at the lowest score | share of attack and of benign rows at the lowest score value, and that value |
| Per attack family | recall | at the frozen threshold, the three oracle FPR points and the two budgets (thresholds from all rows, as above) |
| | ROC-AUC against all benign rows | below 0.5 means the family scores lower than benign traffic |
| | undetectable below full FPR | as above, for the family's rows |

A random forest of 300 trees gives many rows the same score (for example every row on which all trees agree), so a
threshold alone often cannot reach a given FPR or budget; splitting the tie in proportion is the expected result of
breaking it at random. The frozen threshold is reported as scored, without any split. For a later figure, recall at
the oracle FPR points 1e-4, 2e-4, 5e-4, …, 1 is also stored per seed (no CI).

**Uncertainty.** The block bootstrap of the confirmatory analysis: 300 s blocks resampled with replacement,
B = 10 000, seed 20261025, the draws generated exactly as `stats.block_bootstrap` generates them for one evaluation
set, one draw applied to all five seeds, statistic = the seed mean, percentile 95 % CI. The oracle and budget
thresholds are chosen again inside every replicate; the frozen threshold stays fixed. Every replicate is computed
exactly from per-block counts at every distinct score value (no binning). A measure that is undefined in a
replicate (for example a family absent from the drawn blocks) is left out of that replicate, and the number of
replicates used is reported.

**What E1 does not do.** No threshold found here is applied to any confirmatory result, no model is refitted, no
S7 subset is analysed, and no hypothesis is tested. The oracle figures are called oracle wherever they appear.

**Outputs.** `results/exploratory/operating_point/<eval>__<space>.json` (per-seed values, seed means, CIs, the
checks), `results/exploratory/operating_point/summary.md`, run logs in `results/exploratory/logs/`; code in
`src/evaluation/operating_point.py`, run by `scripts/explore.py --operating-point` (notebook 09). Each result file is
written once; a second run skips it.
