# ADR-0007: Sealed test and a preregistered rule for the proposed detector

Status: accepted by the author (2026-10-26), amendment 1.4.

The confirmatory hypotheses H1a–H1c are about "the proposed detector", which the protocol did not define. If it
were defined after test or target scores had been seen, the model could be shaped by those scores. Two rules close
that gap:

1. **Sealed test.** Model development reads only the CICIDS2017 train and val partitions
   (`src/models/develop.py` refuses anything else). Test partitions and external targets are scored only by code
   that first calls `src/models/seal.py: require_frozen`, which needs `results/models/freeze.json` committed to
   git unchanged and every model file matching its SHA-256.
2. **Proposed detector by rule.** The author chose (decision card, 2026-10-26) that the proposed detector is the
   candidate family with the highest seed-mean validation macro-F1 in the `cic` feature space, from the fixed
   candidate list and grids in `config/models.yaml`; ties within 0.0005 go to the earlier candidate. For H1c the
   same family, re-tuned on its own grid in the `unsw_canonical` space, is the proposed detector. The freeze runs
   once and applies this rule mechanically.

The alternative, a new architecture designed by the author, was considered and not chosen; the study's
contribution is the leakage-resistant evaluation and the predictive and response components, not a new
flow classifier.

**Additions.** Amendment 1.5 applies the same two rules to Level 3 (`results/models/freeze_lanl.json`, selection by
val early-alert rate). Clarification 1.5a (2026-10-26): the freeze also refuses to run until
`docs/protocol/prior_exposure.md` is complete and records its SHA-256, and the author decided to freeze Level 3 as
preregistered although its best val early-alert rate (0.349) is below Y = 0.50, without another development round.
`results/models/freeze_summary.md` states both freezes in words, to six decimals.

**Amendment 1.6 (2026-10-27).** The Level 3 labels at the end of a period were corrected and Level 3 is developed
and frozen again: `results/models/freeze_lanl_1.6.json` (track `lanl`, the one H2 is scored with) with records in
`results/models/development/lanl_1.6/`. The 1.5 freeze `freeze_lanl.json` (track `lanl_1.5`) and its records and
models stay where they were written, unchanged, and `require_frozen(track="lanl_1.5")` still checks them; the 1.6
freeze names it in `supersedes`. The scoring code (`src/models/score.py`) calls `require_frozen` for the level it
scores, takes the candidates from the freeze record, loads only model files whose SHA-256 the record lists, and
for Level 3 refuses a config other than the one frozen. Scoring and the decision rules are in preregistration §6b.
