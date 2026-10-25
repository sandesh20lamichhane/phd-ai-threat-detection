# ADR-0004: Within-day chronological block splits

Status: accepted by the author (2026-10-25), replacing the by-day split for Level 1.

CICIDS2017 runs different attacks on different days, so a by-day split tests only on attack types absent from
training. A random row split puts adjacent flows of one attack on both sides. Each day is cut into 300 s blocks;
per (day, family) the earliest 70 % of blocks train, next 15 % validate, last 15 % test; flows crossing a
partition boundary are purged. The same blocks are the resampling unit of the bootstrap. The by-day split is kept
as sensitivity S1.
