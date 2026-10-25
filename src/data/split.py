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


# ---- frozen split rules of protocol v1.0 (docs/protocol/preregistration.md §5) ------------------------------
import pandas as pd

TRAIN, VAL, TEST, EXTERNAL = 0, 1, 2, 3
EXCLUDED, PURGED = -1, -2
PARTITION_NAMES = {TRAIN: "train", VAL: "val", TEST: "test", EXTERNAL: "external",
                   EXCLUDED: "excluded", PURGED: "purged"}


def blocks(ts, block_seconds):
    """Block index per row: floor(start time / block length). Rows without a time get -1."""
    ts = np.asarray(ts, dtype="float64")
    out = np.full(len(ts), -1, dtype=np.int64)
    ok = np.isfinite(ts)
    out[ok] = np.floor(ts[ok] / block_seconds).astype(np.int64)
    return out


def _counts(n, fractions):
    n_val = max(1, int(round(fractions["val"] * n)))
    n_test = max(1, int(round(fractions["test"] * n)))
    return n - n_val - n_test, n_val, n_test


def block_strata(block, family, y, benign="BENIGN"):
    """Stratum of each block: its most frequent attack family, or benign if it holds no attack flow."""
    df = pd.DataFrame({"block": block, "family": family, "y": y})
    att = df[(df.y == 1) & (df.block >= 0)]
    top = att.groupby(["block", "family"]).size().reset_index(name="n")
    top = top.sort_values(["block", "n", "family"], ascending=[True, False, True]).drop_duplicates("block")
    strata = pd.Series(benign, index=pd.Index(np.unique(block[block >= 0]), name="block"))
    strata.loc[top.block.to_numpy()] = top.family.to_numpy()
    return strata


def chronological_block_split(block, day, family, y, fractions, min_blocks=3):
    """Per (day, stratum): earliest blocks -> train, next -> val, last -> test (ADR-0004).

    `family` is the label that defines the strata; the pipeline passes the raw attack label (e.g. DoS Hulk,
    DoS GoldenEye), so every attack variant, not just every family, is split in time.

    Returns (partition per row, diagnostics). Rows without a block are EXCLUDED. A stratum with fewer than
    `min_blocks` blocks on a day stays in train for that day and is listed in the diagnostics.
    """
    block, day = np.asarray(block), np.asarray(day, dtype=object)
    strata = block_strata(block, family, y)
    first_day = pd.Series(day).groupby(block).first()
    tab = pd.DataFrame({"stratum": strata, "day": first_day.reindex(strata.index).to_numpy()})
    assign, too_small = {}, []
    for (d, s), g in tab.groupby(["day", "stratum"], sort=True):
        b = np.sort(g.index.to_numpy())
        if len(b) < min_blocks:
            too_small.append({"day": d, "stratum": s, "blocks": int(len(b))})
            assign.update({int(x): TRAIN for x in b})
            continue
        n_tr, n_va, _ = _counts(len(b), fractions)
        for i, x in enumerate(b):
            assign[int(x)] = TRAIN if i < n_tr else VAL if i < n_tr + n_va else TEST
    part = pd.Series(block).map(assign).fillna(EXCLUDED).astype(np.int8).to_numpy()
    return part, {"strata_kept_in_train": too_small,
                  "blocks_per_partition": pd.Series(assign).map(PARTITION_NAMES).value_counts().to_dict()}


def purge(part, block, day, ts, dur, block_seconds):
    """Drop flows that run past the start of the next block of the same day when that block is in another
    partition. Returns (partition with PURGED rows, number purged)."""
    part = part.copy()
    df = pd.DataFrame({"block": block, "day": day, "part": part})
    bp = df[df.block >= 0].groupby("block").agg(day=("day", "first"), part=("part", "first")).sort_index()
    bp["next_block"] = pd.Series(bp.index, index=bp.index).groupby(bp["day"]).shift(-1)
    bp["next_part"] = bp["next_block"].map(bp["part"])
    edge = bp[bp.next_block.notna() & (bp.next_part != bp.part)]
    if edge.empty:
        return part, 0
    boundary = pd.Series(edge.next_block.to_numpy() * block_seconds, index=edge.index)
    end = np.asarray(ts, dtype="float64") + np.nan_to_num(np.asarray(dur, dtype="float64"), nan=0.0)
    lim = pd.Series(block).map(boundary).to_numpy(dtype="float64")
    hit = np.isfinite(lim) & (end > lim) & (part >= 0)
    part[hit] = PURGED
    return part, int(hit.sum())


def by_day_split(day, train, val, test):
    day = np.asarray(day, dtype=object)
    part = np.full(len(day), EXCLUDED, dtype=np.int8)
    for code, names in ((TRAIN, train), (VAL, val), (TEST, test)):
        part[np.isin(day, names)] = code
    return part


def _stratified_pick(idx, labels, frac, seed):
    """Indices (subset of idx) drawn per label with the given fraction; labels with one row are never drawn."""
    rng = np.random.default_rng(seed)
    pick = []
    lab = pd.Series(labels[idx], index=idx)
    for _, g in lab.groupby(lab, sort=True):
        k = int(round(frac * len(g))) if len(g) > 1 else 0
        if k:
            pick.append(rng.choice(g.index.to_numpy(), size=k, replace=False))
    return np.concatenate(pick) if pick else np.array([], dtype=np.int64)


def stratified_random_split(labels, fractions, seed):
    labels = np.asarray(labels, dtype=object)
    part = np.full(len(labels), TRAIN, dtype=np.int8)
    idx = np.arange(len(labels))
    test = _stratified_pick(idx, labels, fractions["test"], seed)
    part[test] = TEST
    rest = idx[part == TRAIN]
    val = _stratified_pick(rest, labels, fractions["val"] / (1 - fractions["test"]), seed + 1)
    part[val] = VAL
    return part


def official_split(official, labels, val_frac, seed):
    """Keep the provider's train/test; carve a stratified validation set out of train."""
    official = np.asarray(official, dtype=object)
    part = np.where(official == "test", TEST, TRAIN).astype(np.int8)
    tr = np.flatnonzero(part == TRAIN)
    part[_stratified_pick(tr, np.asarray(labels, dtype=object), val_frac, seed)] = VAL
    return part


def file_order_split(file, row, fractions, which=("train", "val", "test")):
    """Within each source file, rows in file order: first share -> train, next -> val, last -> test.

    For datasets without timestamps whose files are captures written in time order (CIC-IoT2023, CICIoMT2024).
    `which` lists the partitions to fill, e.g. ("train", "val") to carve validation out of official train files.
    """
    file, row = np.asarray(file), np.asarray(row)
    part = np.full(len(file), EXCLUDED, dtype=np.int8)
    codes = {"train": TRAIN, "val": VAL, "test": TEST}
    total = sum(fractions[w] for w in which)
    for f in np.unique(file):
        idx = np.flatnonzero(file == f)
        idx = idx[np.argsort(row[idx], kind="stable")]
        cuts = np.cumsum([fractions[w] / total for w in which])
        bounds = np.round(cuts * len(idx)).astype(int)
        start = 0
        for w, end in zip(which, bounds):
            part[idx[start:end]] = codes[w]
            start = end
    return part


def file_row_blocks(file, row, block_rows):
    """Bootstrap groups for datasets without time: consecutive runs of `block_rows` rows within a file."""
    return np.asarray(file, dtype=np.int64) * 10**9 + np.asarray(row, dtype=np.int64) // block_rows
