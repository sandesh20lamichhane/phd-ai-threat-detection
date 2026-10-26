# Preregistered evaluation protocol (v1.0, frozen 2026-10-25)

**Author:** Sandesh Lamichhane, Assistant Professor, Yenepoya (Deemed to be University)

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
   flow-level model can classify, and dropping them would inflate test scores. Rows whose feature vector carries
   both a benign and an attack label are also flagged `_conflict_bin = 1`. Sensitivity S4 drops the conflicts of
   the task being scored: `_conflict_bin` for the binary task, `_conflict` for the multiclass task (amendment 1.2).
7. **Near-duplicates** are defined once, here: two rows are near-duplicates if every feature is equal after
   rounding to 3 significant figures. They are not removed; their within-partition and cross-partition rates are
   reported (§5), and sensitivity S7 (§8.1) scores every model again without the evaluation rows that have one in
   the data the model was fitted or tuned on.
8. **Types**: numeric features stored as float32; low-cardinality categoricals as strings; free-text fields
   (URIs, user agents, DNS queries, certificate subjects) excluded as identifier-like.

## 5. Splits and leakage controls

| Dataset | Split | Grouping unit for split and bootstrap |
|---|---|---|
| CICIDS2017 (primary) | within-day chronological block split: each day cut into 300 s blocks by flow start; each block takes the raw attack label most frequent in it (benign if none); for each (day, label) the earliest 70 % of blocks → train, next 15 % → val, last 15 % → test; a label with fewer than 3 blocks on a day stays in train for that day | 300 s block |
| CICIDS2017 (secondary S1) | by day: Mon–Wed train, Thu val, Fri test (unseen-attack analysis) | 300 s block |
| CSE-CIC-IDS2018, UNSW-NB15 | none: whole dataset is the external test set | 300 s block |
| LANL | chronological by day (§7) | host |
| CICIoMT2024 | official train/test captures; val = the last 15 % of rows (of 85 %) of each train capture, in file order | 1 000 consecutive rows within a file |
| CIC-IoT2023 | per capture file, in file order: first 70 % → train, next 15 % → val, last 15 % → test (no time or host fields) | 1 000 consecutive rows within a file |
| ToN_IoT network | same chronological block rule as CICIDS2017 | 300 s block |
| NSL-KDD | official KDDTrain+ / KDDTest+ | none |

**Purging.** Where two neighbouring blocks of a day fall in different partitions, flows of the earlier block whose
end time (start + duration) passes the boundary are removed. Counted.

**Frozen splits.** Split membership is written to `data/splits/<dataset>/<split>.parquet` (row key → partition)
with its SHA-256 recorded in the repo, before any model is trained.

**Integrity checks** (`src/data/leakage.py`, must pass before results are reported): no identifier column in any
feature matrix; zero exact feature+label overlap across partitions; partitions disjoint by row key and by block;
near-duplicate cross-partition rate reported; exact and near-duplicate feature overlap between CICIDS2017 train
and each external target reported. The cross-dataset check covers identical (or, at 3 and 2 significant figures,
nearly identical) feature vectors only; it does not rule out semantically similar traffic and is reported that way.

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

## 6a. Model development and the proposed detector (amendment 1.4)

**Candidates** (fixed in [`config/models.yaml`](../../config/models.yaml) before any model was trained):
logistic regression, random forest, XGBoost, a multilayer perceptron and a CNN-LSTM, each with a small fixed
hyperparameter grid. Inputs are median-imputed with statistics of the training rows; the linear model is
standardised, neural models see sign(x)·log1p(|x|) standardised; no resampling (class imbalance only through the
`class_weight` / `scale_pos_weight` options in the grids).

**Two feature spaces**, both trained on CICIDS2017: `cic`, the 82 frozen shared CICFlowMeter features (H1a, H1b),
and `unsw_canonical`, the canonical mapped features (H1c).

**Procedure** (`scripts/develop.py`), per candidate and feature space: (1) every grid entry is trained with the
grid seed (11) on train and scored on val at its val-optimal threshold; (2) the best entry, ties within 0.0005
going to the earlier entry, is trained with all five seeds; (3) per seed, the decision threshold maximises
macro-F1 on val. Early stopping (XGBoost, neural models) uses val. Development never reads test or target data.

**Proposed detector** (the author's decision, ADR-0007): the candidate family with the highest seed-mean val
macro-F1 in the `cic` space; for H1c the same family in the `unsw_canonical` space. All other candidates are
baselines for the secondary model comparisons. `scripts/develop.py --freeze` applies the rule once and writes
`results/models/freeze.json`, with the SHA-256 of every development record and model file.

**Sealed test.** Test partitions and external targets are scored once, after the freeze record is committed, by
code that checks it (`src/models/seal.py`). The prior-exposure record is completed before that scoring.

**Sensitivity analyses** S1–S6 retrain the frozen recipe (family, selected hyperparameters, threshold rule) on
their changed data; they do not re-tune. S7 scores the frozen models unchanged.

**Level 3 (H2).** Development rules for the LANL predictive model are fixed in a separate amendment before any
LANL model is trained.

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

### 7a. Level 3 model development (amendment 1.5)

Rules in [`config/models_lanl.yaml`](../../config/models_lanl.yaml), fixed before any LANL model was trained.

- **Periods** (contingency rule of §7): train days 1–10, val days 11–13, test days 14–58. Development reads days
  1–13 only; the test period is sealed until the Level 3 freeze (ADR-0007).
- **Features**: the 31 host-hour counts, plus per host the sum of each over the current and previous 23 hours and
  the number of active hours among them (causal: only events before the alert time).
- **Labels with censoring**: inside a period ending at E, a host's compromise time t is known only if t <= E.
  A host-hour is positive if its alert time (end of the hour) is 0–24 h before the known t; the host's later rows
  leave the population. A compromise after E leaves the host's rows negative in that period.
- **False-alarm threshold**: per seed, the lowest score threshold at which at most 1 % of eligible val host-days
  alert. Eligible host-days are host-days with at least one host-hour row that neither overlap the 24 h before a
  known compromise nor come after it. This operationalises §7's "threshold fixed on the training period": the
  pre-test period is used, on its held-out days, so the threshold is not set on rows the model was fitted on.
- **Selection**: candidates logistic regression, random forest, XGBoost and MLP with the Level 1–2 grids (the
  CNN-LSTM is not a candidate: host-hour counts have no order along the feature axis). Grid entries are scored by
  the val early-alert rate (share of hosts first compromised in val with an alert in their 24 h window) at the
  1 % threshold; ties by host-hour PR-AUC on val, then grid order. The selected entry is trained with five seeds.
- **The predictive model of H2** is the candidate with the best seed-mean val early-alert rate (ties: val
  PR-AUC, then candidate order), the same rule the author chose for H1 (ADR-0007), applied once by
  `scripts/develop.py --lanl --freeze`.
- **Test scoring** (after the freeze): every test-period host-hour is scored; the 24 h window of a host first
  compromised early on day 14 may include day-13 hours, which the frozen model scores like any other hour.

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
| S7 | evaluation without near-duplicates of fitting/tuning data (§8.1), at 3 and at 2 significant figures |

S4 is task-specific (amendment 1.2): with the binary task it removes rows flagged `_conflict_bin` (identical
features labelled both benign and attack); with the multiclass task (S6) it removes rows flagged `_conflict`
(identical features with different attack families).

### 8.1 Sensitivity S7: near-duplicate evaluation (amendment 1.2)

*Why.* After exact deduplication and the block split, many test rows still have a near-duplicate (§4.7) in
training: 7.8 % in CICIDS2017, 47 % in CIC-IoT2023, 57 % in CICIoMT2024 and 72 % in ToN_IoT (preprocessing
reports). High scores could then come partly from recurring, nearly identical traffic rather than generalisation.

*What changes.* Only the evaluation set. Models, training data, validation-based choices (including the decision
threshold) and splits are those of the primary analysis. Evaluation rows are removed when their 3-significant-
figure representation also occurs in data the model was fitted or tuned on:

| Evaluated rows | compared with |
|---|---|
| test (within a dataset) | train ∪ val of the same dataset |
| val (reported only, never used for a choice) | train |
| external target (CSE-CIC-IDS2018, UNSW-NB15) | CICIDS2017 train ∪ val, on the shared or canonical UNSW features (amendment 1.3; val sets the decision threshold) |
| Level 4 domain test sets | train ∪ val of the model being scored (amendment 1.3) |

The share of target rows seen in CICIDS2017 train alone, exact and near, is kept as a descriptive figure only.

The same is repeated at 2 significant figures, a stricter definition, to show how much the result depends on
the 3-figure choice. Rows are never reassigned between partitions: re-grouping near-duplicates across the
chronological boundary would break the time order of the primary split and could merge unrelated flows that
happen to round alike.

*Computation.* `scripts/preprocess.py --near-duplicates` and `--cross-overlap` write, per dataset, a record of
each row's near-duplicate hashes and whether it is "seen" (`data/splits/<dataset>/near*.parquet`, SHA-256 in
`results/data_audit/<dataset>/near_duplicates.json` and `cross_dataset_overlap.json`). The 3-figure hashes are
recomputed from the stored features and must reproduce the preprocessing report's counts exactly, or the step stops.

*Reporting.* For each confirmatory claim (H1a–c, H4): the S7 estimate with its 95 % block-bootstrap CI and its
three-outcome verdict beside the primary one; the share of evaluation rows removed; benign and attack rows and
blocks left; rows per family before and after. An S7 estimate is "not estimable" when fewer than 100 rows or
30 bootstrap blocks of either class remain, and is then reported as such, not replaced.

*Claims that compare two evaluation sets* (H1b, H1c: Δ = source test − target; H4 as fixed in §9): the S7 Δ
uses the S7 subsets of both sides, so it is estimable only if both are. When only the target side is estimable,
its S7 macro-F1 and CI are reported descriptively beside the primary target macro-F1, with no robustness verdict
(amendment 1.3). The minimum of 100 rows and 30 blocks per class is fixed and is not changed after the S7
subsets were counted: CICIDS2017 (17 attack blocks left at 3 significant figures) and NSL-KDD (23 blocks per
class) are reported as not estimable.

*Integrity (amendment 1.3).* Every S7 record states the S7 definition it was built under, the processed outputs it
was built from (dataset rules and output/split SHA-256) and the SHA-256 of its sidecar; the JSON records are
hashed in `results/data_audit/auxiliary_manifest.json`. The loader refuses an S7 filter whose sidecar, record or
provenance does not match, and `--verify` checks all of them, reported separately from the primary data.

*Interpretation.* S7 is a robustness check, not a new hypothesis: it is outside the Holm family and never changes
the primary verdict. A claim is called robust to near-duplicates if its S7 verdict (3 figures) is the same as the
primary one; if not, the paper's abstract and discussion say the claim depends on near-duplicate traffic.

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

## 10. No-shortcut rule

Every confirmatory and sensitivity result uses the full processed datasets, all five seeds, B = 10 000 bootstrap
replicates and the frozen splits. No subsampling, early stopping of the analysis, reduced seeds or partial data
are allowed for a reported number. `scripts/preprocess.py --max-rows` exists only for smoke tests: its report is
marked as a pilot, and `src/data/datasets.py` refuses to load pilot outputs for any experiment.

## 10a. Output verification (amendment 1.2)

Before any model run, `scripts/preprocess.py --verify` must pass for every dataset the run uses. It checks that
the dataset's settings and label rules (a SHA-256 fingerprint of its merged configuration) are identical to those
under which its outputs were built, that the outputs are not a pilot, and that every processed and split file
matches the SHA-256 in its report. `src/data/datasets.py` repeats the rules check at every load and refuses
outputs built under other rules. Since amendment 1.3 it also checks every S7 artifact (§8.1, *Integrity*) and
reports primary data (processed and split files) and auxiliary sensitivity-analysis artifacts in separate
tables. Results: `results/data_audit/verification.{json,md}`.

## 11. Records

Every report, data card, frozen feature list, split checksum and run log is written into the repository clone in
Drive (`MyDrive/PhD-AI-Cyber/code/phd-ai-threat-detection`) and pushed to GitHub from there, so the two copies are
identical. Data files themselves (raw, processed parquet, split membership) stay in Drive only, as agreed for raw
data; their SHA-256 values are in the reports on GitHub, so anyone can verify that the Drive files are the ones
the results were computed from.

## 12. Deviations

Any departure from this document is logged in [protocol_amendments.md](protocol_amendments.md) with date, reason
and whether results had been seen. Exploratory analyses are labelled exploratory in the thesis and paper.
