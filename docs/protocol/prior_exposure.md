# Prior exposure to the datasets

Reviewers should be able to tell which analyses are prospective. This file records what the author had already
done with each dataset before the protocol was frozen on 2026-10-25.

| Dataset | Earlier work by the author | What was seen |
|---|---|---|
| CICIDS2017 | `ids-adapt` project; `when-should-ids-adapt`; `cicids-label-correction-audit` | within-dataset models and their scores; label-correction audit of the corrected vs original release. **Author to complete:** which splits, models and headline numbers. |
| CSE-CIC-IDS2018 | `ids-adapt` (processed parquet, probably from the original uncorrected release) | **Author to complete:** whether any CICIDS2017 → CSE-CIC-IDS2018 transfer result was observed, and its size. |
| UNSW-NB15 | `ids-adapt` (processed parquet) | **Author to complete:** as above. |
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
