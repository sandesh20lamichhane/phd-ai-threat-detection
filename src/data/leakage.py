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
