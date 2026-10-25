"""Checks that must pass before any result is reported."""
import pandas as pd

from .clean import identifier_columns


def row_overlap(train, test, columns=None):
    """Number of test rows that also appear (exactly) in train."""
    cols = columns or [c for c in train.columns if c in test.columns]
    h_train = set(pd.util.hash_pandas_object(train[cols], index=False))
    h_test = pd.util.hash_pandas_object(test[cols], index=False)
    return int(h_test.isin(h_train).sum())


def assert_no_leakage(train, test, label_col="Label"):
    feats = [c for c in train.columns if c != label_col]
    ids = identifier_columns(feats)
    if ids:
        raise AssertionError(f"identifier columns still present: {ids}")
    n = row_overlap(train, test, feats + [label_col])
    if n:
        raise AssertionError(f"{n} test rows also appear in training data")
    return True


# ---- split integrity checks on the hash arrays of the streaming pipeline -----------------------------------
import numpy as np

from .split import PARTITION_NAMES, TEST, TRAIN, VAL


def _share_in(a, b):
    """Fraction of entries of a whose value also occurs in b."""
    if len(a) == 0:
        return 0.0
    return float(np.isin(a, np.unique(b)).mean())


def split_integrity(part, fl_hash, f_hash, near_hash, block=None, feature_columns=(), block_split=False):
    """Checks of preregistration §5. Raises on a hard failure, returns the numbers for the report."""
    part = np.asarray(part)
    ids = identifier_columns([c for c in feature_columns])
    if ids:
        raise AssertionError(f"identifier columns in the feature matrix: {ids}")
    out = {"identifier_columns_in_features": 0, "pairs": {}}
    live = [p for p in (TRAIN, VAL, TEST) if (part == p).any()]
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            ma, mb = part == a, part == b
            exact = int(np.isin(fl_hash[mb], np.unique(fl_hash[ma])).sum())
            if exact:
                raise AssertionError(f"{exact} {PARTITION_NAMES[b]} rows repeat a {PARTITION_NAMES[a]} row exactly")
            out["pairs"][f"{PARTITION_NAMES[a]}->{PARTITION_NAMES[b]}"] = {
                "exact_feature_and_label_overlap": exact,
                "same_features_other_label_share": _share_in(f_hash[mb], f_hash[ma]),
                "near_duplicate_share": _share_in(near_hash[mb], near_hash[ma]),
            }
    if block_split and block is not None:
        bp = {}
        for p in live:
            for b_ in np.unique(block[part == p]):
                bp.setdefault(int(b_), set()).add(p)
        shared = [b_ for b_, s in bp.items() if len(s) > 1]
        if shared:
            raise AssertionError(f"{len(shared)} blocks appear in more than one partition")
        out["blocks_shared_between_partitions"] = 0
    return out


def cross_dataset_overlap(source_hash, target_hash):
    """Share of target rows whose (shared-feature) hash also occurs in the source training data."""
    return {"target_rows": int(len(target_hash)), "share_seen_in_source_train": _share_in(target_hash, source_hash)}
