# Preregistered evaluation protocol (v1.0, frozen 2026-10-25)

This document fixes, before any confirmatory run, what is estimated, how the data are prepared and split, which
comparisons are confirmatory, and how each result is decided. The preprocessing it describes is implemented in
`src/data/` and configured in [`config/preprocessing.yaml`](../../config/preprocessing.yaml); the SHA-256 of that
file is written into every preprocessing report, so a result can always be traced to the exact rules that built
its data. Changes after the freeze go in [protocol_amendments.md](protocol_amendments.md).

Status of the evidence: the author has worked with CICIDS2017, CSE-CIC-IDS2018 and UNSW-NB15 before
([prior_exposure.md](prior_exposure.md)). This study is preregistered in the sense that its analysis plan is
frozen before its own confirmatory runs; it does not claim that the author is blind to these datasets.

## 1. Research question and scope

Does a detector trained on one enterprise network benchmark keep its performance on traffic it has never seen
(other networks, later time, other sectors), once the known benchmark artefacts (duplicates, identifier
shortcuts, label errors, temporal leakage) are removed?

Conclusions are limited to the benchmarks used. Transfer among three or four lab-generated datasets is evidence
about those datasets, not about every operational network.

## 2. Datasets and roles

| Dataset | Role | Confirmatory? |
|---|---|---|
| CICIDS2017, corrected (Engelen et al. 2021) | Level 1 source and Level 2 training data | yes |
| CSE-CIC-IDS2018, corrected (Liu et al. 2022) | Level 2 external target (same flow extractor) | yes |
| UNSW-NB15 (official CSVs 1–4) | Level 2 external target (different flow extractor) | yes |
| LANL Cyber1 | Level 3 temporal / predictive test | yes |
| CICIoMT2024, CIC-IoT2023, ToN_IoT (network) | Level 4 domain targets | yes, once the Level 4 design in §9 is fixed |
| NSL-KDD | historical baseline only | no |
| Edge-IIoTset | reference copy, not in the finalised plan | no (profiled only) |
| ids-adapt legacy parquets | earlier project's output, probably uncorrected labels | no, never used for any reported number |
| CSE-CIC-IDS2018 original | disabled; label-correction comparison only | no |

No target dataset is used for anything other than its final evaluation: no feature selection, scaling, threshold
choice, early stopping or model selection may look at target data.

## 3. Prediction tasks and label spaces

**Primary task (all confirmatory claims): binary, benign vs attack.** It is the only label space all Level 2
targets share; CICIDS2017 and UNSW-NB15 have no common attack taxonomy, and merging categories to make
multiclass scores look comparable would hide the very shift the study measures.

**Secondary task: multiclass on a family taxonomy**, reported for Level 1 and for CICIDS2017 → CSE-CIC-IDS2018
only. Families: Benign, DoS, DDoS, BruteForce, WebAttack, Bot, PortScan, Infiltration, Heartbleed. The mapping
from every raw label string to a family is a regex table in `config/preprocessing.yaml`; a raw label that matches
no rule stops the pipeline. A family enters the multiclass task only if it has at least 3 time blocks and at
least 50 flows after deduplication; smaller families stay in the binary task and are reported.

**"Attempted" flows** (corrected CIC releases: attack-tool flows that carried no attack payload, e.g. a TCP
handshake that was never followed up): **excluded** from training and evaluation in the primary analysis and
counted in the data card. Sensitivity analysis S3 relabels them as benign. Reason: they are behaviourally benign
at the flow level, so counting them as attacks rewards shortcut learning and counting them as benign is a
contestable relabelling (ADR-0003).

**Original labels are preserved.** Each processed row keeps the raw label string (`_label_raw`) next to its
canonical label. The corrected CIC releases regenerated the flows with a fixed CICFlowMeter, so there is no
row-by-row correction log against the original CSVs; correction provenance is recorded at release level
(source URL, citation, SHA-256 in `results/data_manifest/`), and ADR-0002 explains the choice.

## 4. Data preparation (fixed before modelling)

Applied identically to every dataset by `scripts/preprocess.py`:

1. **Read** the files listed in the dataset's manifest; record input SHA-256 values.
2. **Normalise** column names (whitespace), map labels (§3), parse timestamps.
3. **Move identifiers out of the feature matrix.** Flow IDs, IP addresses, ports, timestamps and row ids are not
   model inputs. Timestamps, day, source file and (for the ports sensitivity analysis) destination port are kept
   as audit metadata columns prefixed `_`, which every model loader drops.
4. **Non-finite values**: ±inf becomes NaN; no imputation at this stage (imputation is a learned transform, §5).
   Rows that are entirely NaN in the features are dropped and counted.
5. **Exact duplicates** (identical feature vector and identical canonical label) are removed before splitting,
   keeping the first occurrence in file/time order. Counted per class.
6. **Conflicting-label duplicates** (identical feature vector, different canonical labels) are kept, one row per
   label, and flagged `_conflict = 1`. Primary analysis keeps them, which is conservative: these are the flows no
   flow-level model can classify, and dropping them would inflate test scores. Sensitivity S4 drops them.
7. **Near-duplicates** are defined once, here: two rows are near-duplicates if every feature is equal after
   rounding to 3 significant figures. They are not removed; their within-partition and cross-partition rates are
   reported (§5).
8. **Types**: numeric features stored as float32; low-cardinality categoricals as strings; free-text fields
   (URIs, user agents, DNS queries, certificate subjects) excluded as identifier-like.

## 5. Splits and leakage controls

| Dataset | Split | Grouping unit for split and bootstrap |
|---|---|---|
| CICIDS2017 (primary) | within-day chronological block split: each day cut into 300 s blocks by flow start; each block gets its majority family; for each (day, family) the earliest 70 % of blocks → train, next 15 % → val, last 15 % → test; a family with fewer than 3 blocks on a day stays in train for that day | 300 s block |
| CICIDS2017 (secondary S1) | by day: Mon–Wed train, Thu val, Fri test (unseen-attack analysis) | 300 s block |
| CSE-CIC-IDS2018, UNSW-NB15 | none: whole dataset is the external test set | 300 s block |
| LANL | chronological by day (§7) | host |
| CICIoMT2024 | official train/test files; val = 15 % of train, stratified, seed 11 | file × 300 s block if timestamps exist, else none |
| CIC-IoT2023 | stratified random 70/15/15, seed 11 (the CSVs carry no time or host fields) | none: limitation, reported |
| ToN_IoT network | same chronological block rule as CICIDS2017 | 300 s block |
| NSL-KDD | official KDDTrain+ / KDDTest+ | none |

**Purging.** Where two neighbouring blocks of a day fall in different partitions, flows of the earlier block whose
end time (start + duration) passes the boundary are removed. Counted.

**Frozen splits.** Split membership is written to `data/splits/<dataset>/<split>.parquet` (row key → partition)
with its SHA-256 recorded in the repo, before any model is trained.

**Integrity checks** (`src/data/leakage.py`, must pass before results are reported): no identifier column in any
feature matrix; zero exact feature+label overlap across partitions; partitions disjoint by row key and by block;
near-duplicate cross-partition rate reported; exact feature overlap between CICIDS2017 train and each external
target reported.

**Learned transformations** (imputer, scaler, encoder, feature selector, class balancing, decision threshold) are
fitted on the source training partition only, saved with a hash, and applied unchanged to validation, test and
external data (`src/features/transform.py`). Balancing, if any, is applied to training folds only. The binary
decision threshold is the one that maximises macro-F1 on the source validation partition.

## 6. Statistical analysis

**Metrics.** Primary: macro-F1. Always reported with it: per-class precision, recall and F1; balanced accuracy;
false-positive rate; PR-AUC (attack class); FPR at 95 % detection rate. Accuracy is never reported alone.
Macro-F1 depends on class prevalence, and prevalence differs between datasets; per-class recall and balanced
accuracy, which do not, are reported beside every Δ so a drop can be split into "the model got worse" and "the
class mix changed".

**Seeds.** Every model is trained with the five seeds in `config/seeds.yaml`. The estimand is the seed-mean.

**Uncertainty: group-aware block bootstrap.** Flows within a block are strongly correlated, so the resampling unit
is the block (§5), never the flow. Per block and seed, the confusion counts are stored; each bootstrap replicate
resamples blocks with replacement (separately in source test and target), sums the counts, computes macro-F1
per seed and averages over seeds. B = 10 000 replicates, percentile 95 % intervals, seed 20261025
(`src/evaluation/stats.py: block_bootstrap`). Training-set variability is captured only through the seeds; this
is a stated limitation.

**Tests and multiplicity.** Each threshold claim (§ hypotheses) is a one-sided test against its margin; the
bootstrap one-sided p-value is the share of replicates on the wrong side of the margin. The 95 % CI rule in
[hypotheses.md](hypotheses.md) is equivalent to alpha = 0.025 per direction. The confirmatory family is
{H1a, H1b, H1c, H2, H4-healthcare, H4-IoT, H4-government} for the proposed model; Holm's step-down procedure
controls the family-wise error rate at 0.025 per direction. H3 is tested in its own family (response
experiments). Unadjusted CIs are always shown; claims use the Holm-adjusted decisions.

**Model comparisons** (proposed vs baselines) are secondary: Wilcoxon signed-rank across seeds × targets, or
Friedman + Nemenyi for more than two models, Holm-adjusted within each table.

## 7. Level 3 (LANL) operational definitions

- Unit: host-hour feature vectors aggregated from auth, proc, flows and dns events (`src/data/lanl.py`); alerts
  are issued at the end of each hour.
- Compromise: the first red-team event whose destination is a given host. Hosts compromised in the training
  period are removed from the test population.
- Early alert: an alert on the host within the 24 h before its compromise time (prediction horizon 24 h).
- False alarm: a host-day with at least one alert, among host-days that are not within 24 h before a compromise
  and not after that host's compromise. The alert threshold is fixed on the training period so that this rate is
  <= 1 %, and the realised test rate is reported.
- Split: days 1–40 train, 41–58 test. Contingency, fixed now: if fewer than 30 compromised hosts fall in days
  41–58, the cut moves to the latest day D such that at least 30 hosts are first compromised on or after D, and
  the earlier part keeps its first 80 % of days for training and the rest for validation. This rule depends only
  on the red-team file, never on model output.
- Baselines: (i) random alerting at the same false-alarm budget; (ii) a reactive detector that scores the hour of
  the compromise itself (lead time <= 0 by construction). H2's CI is a bootstrap over red-team accounts, since
  compromises by one account are not independent.

## 8. Feature policy and alignment

**CICIDS2017 → CSE-CIC-IDS2018.** Both corrected releases were produced with the same fixed CICFlowMeter. The
shared set is every non-identifier column present in both with the same name and meaning; the pipeline computes
the intersection and compares it with the list in `config/feature_mapping/cic_shared_features.yaml`, and stops if
they differ.

**CICIDS2017 → UNSW-NB15.** CICFlowMeter and Argus/Bro define flows differently, so names are never matched
directly. `config/feature_mapping/cic_to_unsw_mapping.yaml` records, per canonical feature, the source column in
each dataset, definition, unit, transform and a status of `exact`, `approximate` or `unavailable`. The primary
shared set uses `exact` and `approximate` features whose transform is documented; sensitivity S5 uses only the
packet-count, duration and timing features (no byte counts, whose header accounting differs between tools).

**Sensitivity analyses** (prespecified; none is used to pick the primary pipeline):

| | Change from primary |
|---|---|
| S1 | CICIDS2017 by-day split (unseen-attack analysis) |
| S2 | add destination port as a feature (service information that can also act as a shortcut) |
| S3 | "Attempted" flows relabelled as benign |
| S4 | conflicting-label duplicate groups removed |
| S5 | CIC → UNSW with timing/count features only |
| S6 | multiclass family task (Level 1 and CIC 2017 → 2018) |

## 9. Open items that must be closed before the runs they affect

1. **Level 4 transfer design (blocks H4 runs, not preprocessing).** CIC-IoT2023 and CICIoMT2024 use a
   packet-window extractor and ToN_IoT uses Zeek connection logs; none shares a CICFlowMeter feature space, so
   "best model of Levels 1–3 applied unchanged" may have no defensible common features. Before the first Level 4
   run the protocol must fix either a canonical cross-extractor feature set (as for UNSW-NB15) or a
   retrain-in-domain design with the frozen recipe, and say which Level 2 number Δ4 is measured against.
2. **H3 non-inferiority margin for disruption cost**, before the first response run.
3. **Prior-exposure details** in [prior_exposure.md](prior_exposure.md) that only the author can fill in.
4. **Schema confirmation.** Column and label lists in the config come from the dataset documentation; the
   profiling step (`scripts/preprocess.py --profile`) checks them against the files in Drive, and any difference is
   logged as an amendment before preprocessing runs.

## 10. Deviations

Any departure from this document is logged in [protocol_amendments.md](protocol_amendments.md) with date, reason
and whether results had been seen. Exploratory analyses are labelled exploratory in the thesis and paper.
