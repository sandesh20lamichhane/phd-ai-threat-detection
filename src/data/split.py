"""Leakage-safe splits for the four-level protocol.

Level 1: by capture day (CICIDS2017 70/15/15 by day) plus stratified 5-fold CV inside train.
Level 3: strictly by time (LANL first 40 days train, last 18 days test).
Balancing happens after splitting and only on training folds (src/data/balance.py).
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def split_by_group(df, group_col, train, val, test):
    """Assign whole groups (e.g. days) to train/val/test. Lists of group values must not overlap."""
    groups = [set(train), set(val), set(test)]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        if groups[a] & groups[b]:
            raise ValueError(f"groups overlap between splits: {groups[a] & groups[b]}")
    return tuple(df[df[group_col].isin(g)].reset_index(drop=True) for g in groups)


def split_by_time(df, time_col, cutoff):
    """Everything strictly before `cutoff` trains, everything at or after it tests."""
    return (df[df[time_col] < cutoff].reset_index(drop=True),
            df[df[time_col] >= cutoff].reset_index(drop=True))


def cv_folds(y, n_splits=5, seed=0):
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
