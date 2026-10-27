# Prior exposure to the datasets

**Author:** Sandesh Lamichhane, Assistant Professor, Yenepoya (Deemed to be University)

Reviewers should be able to tell which analyses are prospective. This file records what the author had already
done with each dataset before the protocol was frozen on 2026-10-25.

| Dataset | Earlier work by the author | What was seen |
|---|---|---|
| CICIDS2017 | M.Sc. thesis (June 2026, see below), via `ids-adapt` / `when-should-ids-adapt`; `cicids-label-correction-audit` | within-dataset models and their scores; label-correction audit of the corrected vs original release. **Author to complete:** which splits, models and headline numbers. |
| CSE-CIC-IDS2018 | M.Sc. thesis (June 2026), via `ids-adapt`: improved (corrected-label) release, each day capped at 1.5 M flows (stratified, seed 42) | **Author to complete:** whether any CICIDS2017 → CSE-CIC-IDS2018 transfer result was observed, and its size. |
| UNSW-NB15 | M.Sc. thesis (June 2026), via `ids-adapt`: official train + test partition (257,673 records, 42 features) | **Author to complete:** as above. |
| Edge-IIoTset | `ids-adapt` | not used in this study. |
| NSL-KDD | M.Sc. thesis (June 2026) | pipeline sanity check only (thesis §2.9); no model result from it informs this study. |
| LANL, ToN_IoT, CIC-IoT2023, CICIoMT2024 | none before this study | first access during data collection (2026-10-19 to 2026-10-25); no models trained. |

Earlier work: S. Lamichhane, *Conditions of Validity, Evaluation Protocols, and Regime-Aware Adaptation for
Explanation-Space Drift Localization in Intrusion Detection Systems*, M.Sc. thesis, Yenepoya (Deemed to be
University), June 2026. It studied drift localization and label-efficient retraining on each dataset separately,
with by-day (CICIDS2017, CSE-CIC-IDS2018) and leave-one-family-out (UNSW-NB15, CSE-CIC-IDS2018) drift streams;
the table says what it reported, and the items below what follows for this study.

Consequences for the claims:

- Any CICIDS2017 → CSE-CIC-IDS2018 or CICIDS2017 → UNSW-NB15 result seen in earlier projects is treated as an
  exploratory prior. H1b and H1c are confirmatory only for this study's frozen pipeline (corrected labels,
  deduplication, block splits, frozen transforms), which differs from the earlier pipelines; the paper states
  this.
- The X margin (0.05) was set without reference to any earlier transfer result. If an earlier result is close
  to 0.05, the paper says so.
- H1a: the thesis had shown that detectors trained on CICIDS2017 traffic of the attack families they are tested on
  reach recall of about 0.99 at a false-positive rate near 0 (by-day evaluation, Attempted flows labelled as
  attack). The 0.95 threshold of H1a was therefore set knowing that within-dataset detection on CICIDS2017 is near
  ceiling; the paper reports H1a as a check that this study's stricter pipeline reproduces it, not as a discovery.
- The thesis also showed that a detector trained on early days misses attack families it has not seen (CICIDS2017
  Wednesday-Friday recall 0.000; later CSE-CIC-IDS2018 days at most 0.055). This is within-dataset novel-family
  drift, not cross-dataset transfer; the paper cites it as prior work. It shaped the expectation that H1b and H1c
  may fail, but no transfer number was seen.
- No dataset or split in this study was chosen because of an earlier model score.

During this study, data profiling (row counts, label counts, schemas, timestamp ranges) happens before the
freeze is applied to the code; no model is trained until the splits are frozen and checksummed.
