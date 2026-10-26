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
