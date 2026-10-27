# Prior exposure to the datasets

**Author:** Sandesh Lamichhane, Assistant Professor, Yenepoya (Deemed to be University)

Reviewers should be able to tell which analyses are prospective. This file records what the author had already
done with each dataset before the protocol was frozen on 2026-10-25.

| Dataset | Earlier work by the author | What was seen |
|---|---|---|
| CICIDS2017 | `ids-adapt` project; `when-should-ids-adapt`; `cicids-label-correction-audit` | within-dataset models and their scores; label-correction audit of the corrected vs original release. **Author's declaration (2026-10-27):** M.Sc. thesis (Yenepoya, June 2026): Engelen et al. (2021) corrected release, 82 features, identifiers removed, Attempted flows labelled as attack. By-day splits in calendar order only (warm-up Monday-Tuesday, then Wednesday-Friday, also 50k-flow replay windows); no random or within-day block split. Models: random forest (100 trees, depth 12), XGBoost, MLP (256-128-64). Headline numbers seen: the frozen detector had recall 0.981 at precision 1.000 on Tuesday and recall 0.000 on Wednesday-Friday (FPR 0.000); detectors retrained on post-drift days reached recall about 0.99 (mean 0.994) at FPR near 0. Metrics were per-day recall, precision, FPR, F1 and PR-AUC; macro-F1 was not used. |
| CSE-CIC-IDS2018 | `ids-adapt` (processed parquet, probably from the original uncorrected release) | **Author's declaration (2026-10-27):** M.Sc. thesis (June 2026): improved (corrected-label) release, each day capped at 1.5M flows (stratified, seed 42), 82 features identical to CICIDS2017. Within-dataset only: a detector trained on the first day (14 Feb) had recall 1.000 on that day and at most 0.055 on later novel-family days (FPR below 3e-5), plus leave-one-family-out drift localization. No model was trained on CICIDS2017 and tested on CSE-CIC-IDS2018 or the reverse, so no transfer result was computed or seen. |
| UNSW-NB15 | `ids-adapt` (processed parquet) | **Author's declaration (2026-10-27):** M.Sc. thesis (June 2026): official train+test partition (257,673 records, 42 features, 9 attack families). Within-dataset only, with leave-one-family-out drift: with a broad warm-up the frozen detector had recall 0.957, precision 0.875 and FPR 0.131 on held-out families, and 0.783 / 0.989 / 0.009 after triggered retraining. No model was trained on CICIDS2017 and tested on UNSW-NB15, and no CIC-to-UNSW feature mapping was used, so no transfer result was computed or seen. |
| Edge-IIoTset | `ids-adapt` | not used in this study. |
| LANL, ToN_IoT, CIC-IoT2023, CICIoMT2024, NSL-KDD | none before this study | first access during data collection (2026-10-19 to 2026-10-25); no models trained. |

Consequences for the claims:

- Any CICIDS2017 → CSE-CIC-IDS2018 or CICIDS2017 → UNSW-NB15 result seen in earlier projects is treated as an
  exploratory prior. H1b and H1c are confirmatory only for this study's frozen pipeline (corrected labels,
  deduplication, block splits, frozen transforms), which differs from the earlier pipelines; the paper states
  this.
- The X margin (0.05) was set without reference to any earlier transfer result. If an earlier result is close
  to 0.05, the paper says so.
- No dataset or split in this study was chosen because of an earlier model score.

During this study, data profiling (row counts, label counts, schemas, timestamp ranges) happens before the
freeze is applied to the code; no model is trained until the splits are frozen and checksummed.

**Declaration.** The three items above were completed by Sandesh Lamichhane on 2026-10-27 (UTC) in notebook 06, step 2, after model development on validation data and before the models were frozen; no test or target score existed. The SHA-256 of this file is recorded in `results/models/freeze.json` and `results/models/freeze_lanl.json`.
