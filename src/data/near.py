"""Near-duplicate records for sensitivity S7 (amendment 1.2, preregistration §8.1).

For every processed row: its near-duplicate hash at 3 significant figures (the preregistered definition, §4.7)
and at 2 (a stricter check of that definition), and whether an evaluation row is "seen", i.e. its hash also
occurs in data the model was fitted or tuned on:

    within a dataset   val rows vs train; test rows vs train + val
    external targets   target rows vs CICIDS2017 train, on the shared (or canonical UNSW) features

S7 then scores the unchanged model on the unseen evaluation rows only. Models, splits and training data do
not change. The 3-figure hashes are recomputed from the stored features, so `build` first checks that they
reproduce the counts in the preprocessing report exactly.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..utils import paths
from ..utils.io import write_json, write_parquet
from .clean import round_sig, row_hash
from .split import PARTITION_NAMES, TEST, TRAIN, VAL

SIG_FIGS = (3, 2)
MIN_ROWS_PER_CLASS = 100        # below this, or below MIN_BLOCKS, an S7 estimate is reported as not estimable
MIN_BLOCKS_PER_CLASS = 30
WITHIN = ["cicids2017_improved", "ciciomt2024", "ton_iot", "ciciot2023", "nsl_kdd"]


def sidecar_path(ds_id, against=None):
    return paths.SPLITS / ds_id / (f"near_vs_{against}.parquet" if against else "near.parquet")


def _hashes(X, chunk=2_000_000):
    """Row hashes after rounding, in chunks so a 45 M-row frame never needs a float64 copy in memory."""
    out = {s: [] for s in SIG_FIGS}
    for i in range(0, max(len(X), 1), chunk):
        part = X.iloc[i:i + chunk]
        for s in SIG_FIGS:
            out[s].append(row_hash(round_sig(part, s)))
    return {s: np.concatenate(v) for s, v in out.items()}


def _seen_within(h, part):
    seen = np.zeros(len(h), bool)
    tr = np.unique(h[part == TRAIN])
    trv = np.unique(h[(part == TRAIN) | (part == VAL)])
    seen[part == VAL] = np.isin(h[part == VAL], tr)
    seen[part == TEST] = np.isin(h[part == TEST], trv)
    return seen


def subset_summary(seen, y, fam, block, mask):
    """Composition of the unseen subset of the rows in mask (reported with every S7 number)."""
    keep = mask & ~seen & (y >= 0)
    out = {"rows": int((mask & (y >= 0)).sum()), "unseen_rows": int(keep.sum())}
    out["unseen_share"] = out["unseen_rows"] / out["rows"] if out["rows"] else None
    est = True
    for cls, name in ((0, "benign"), (1, "attack")):
        k = keep & (y == cls)
        nb = int(len(np.unique(block[k])))
        out[f"unseen_{name}_rows"], out[f"unseen_{name}_blocks"] = int(k.sum()), nb
        est &= k.sum() >= MIN_ROWS_PER_CLASS and nb >= MIN_BLOCKS_PER_CLASS
    out["estimable"] = bool(est)
    out["unseen_by_family"] = {f: int(v) for f, v in pd.Series(fam[keep]).value_counts().items()}
    out["all_by_family"] = {f: int(v) for f, v in pd.Series(fam[mask & (y >= 0)]).value_counts().items()}
    return out


def build(ds_id, split="primary", log=print):
    rep = json.loads((paths.RESULTS / "data_audit" / ds_id / "report.json").read_text())
    files = sorted((paths.PROCESSED / ds_id / rep.get("output_version", "v1")).glob("part-*.parquet"))
    feats = rep["schema"]["feature_columns"]
    keys, y, fam, H = [], [], [], {s: [] for s in SIG_FIGS}
    for f in files:
        df = pd.read_parquet(f, columns=[*feats, "_key", "_y", "_label"])
        for s, h in _hashes(df[feats]).items():
            H[s].append(h)
        keys.append(df["_key"].to_numpy())
        y.append(df["_y"].to_numpy())
        fam.append(df["_label"].astype(str).to_numpy())
        log(f"[{ds_id}] near-duplicate hashes: {f.name}")
    keys, y, fam = np.concatenate(keys), np.concatenate(y), np.concatenate(fam)
    H = {s: np.concatenate(v) for s, v in H.items()}
    sp = pd.read_parquet(paths.SPLITS / ds_id / f"{split}.parquet")
    if not np.array_equal(keys, sp["_key"].to_numpy()):
        raise AssertionError("processed data and split file are out of step; re-run preprocessing")
    part, block = sp["partition"].to_numpy(), sp["_block"].to_numpy()

    # the recomputed 3-figure hashes must reproduce the preprocessing report exactly
    h3 = H[3]
    _, inv, cnt = np.unique(h3, return_inverse=True, return_counts=True)
    got = int((cnt[inv] > 1).sum())
    want = rep["counts"]["rows_in_near_duplicate_groups"]
    pairs = rep["splits"][split]["integrity"]["pairs"]
    check = {"rows_in_near_duplicate_groups": {"report": want, "recomputed": got}}
    for a, b in ((TRAIN, VAL), (TRAIN, TEST), (VAL, TEST)):
        k = f"{PARTITION_NAMES[a]}->{PARTITION_NAMES[b]}"
        if k in pairs:
            mb = part == b
            share = float(np.isin(h3[mb], np.unique(h3[part == a])).mean()) if mb.any() else 0.0
            check[k] = {"report": pairs[k]["near_duplicate_share"], "recomputed": share}
    for k, v in check.items():
        if not np.isclose(v["report"], v["recomputed"], rtol=0, atol=1e-12):
            raise AssertionError(f"{ds_id}: recomputed near-duplicates differ from the report ({k}: {v}); "
                                 "the stored features do not match the preprocessing run")

    side = pd.DataFrame({"_key": keys})
    summary = {"dataset": ds_id, "split": split, "check_against_report": check, "subsets": {}}
    for s in SIG_FIGS:
        seen = _seen_within(H[s], part)
        side[f"near_sig{s}"] = H[s]
        side[f"seen_sig{s}"] = seen
        summary["subsets"][f"sig{s}"] = {PARTITION_NAMES[p]: subset_summary(seen, y, fam, block, part == p)
                                         for p in (VAL, TEST) if (part == p).any()}
    summary["sidecar_sha256"] = write_parquet(side, sidecar_path(ds_id))
    write_json(paths.RESULTS / "data_audit" / ds_id / "near_duplicates.json", summary)
    for s in SIG_FIGS:
        t = summary["subsets"][f"sig{s}"].get("test", {})
        log(f"[{ds_id}] S7 test subset at {s} s.f.: {t.get('unseen_rows', 0):,} of {t.get('rows', 0):,} rows unseen; "
            f"estimable={t.get('estimable')}")
    return summary


def external(target, source_frame, target_frame, target_meta, block, against="cicids2017_improved"):
    """Target rows seen in the source train (same features, same order); writes the sidecar, returns the summary."""
    side = pd.DataFrame({"_key": target_meta["_key"].to_numpy()})
    y, fam = target_meta["_y"].to_numpy(), target_meta["_label"].astype(str).to_numpy()
    out = {}
    hs, ht = _hashes(source_frame), _hashes(target_frame)
    for s in SIG_FIGS:
        seen = np.isin(ht[s], np.unique(hs[s]))
        side[f"seen_sig{s}"] = seen
        out[f"sig{s}"] = subset_summary(seen, y, fam, block, np.ones(len(y), bool))
    out["sidecar_sha256"] = write_parquet(side, sidecar_path(target, against))
    return out
