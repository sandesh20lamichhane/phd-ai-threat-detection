"""Load processed data with a frozen split for experiments. Metadata columns ("_" prefix) never reach models."""
import json

import numpy as np
import pandas as pd

from ..utils import paths
from .near import require_sidecar, sidecar_path
from .split import PARTITION_NAMES
from .verify import require_current

META_PREFIX = "_"


def feature_columns(df):
    return [c for c in df.columns if not c.startswith(META_PREFIX)]


SENSITIVITIES = {"S2_ports", "S3_attempted_benign", "S4_drop_conflicts", "S7_unseen_eval", "S7_unseen_eval_sig2"}


def load(ds_id, split="primary", partitions=("train", "val", "test"), columns=None, sensitivity=(),
         version="v1", task="binary", near_against=None):
    """Returns (X, meta) for the requested partitions.

    sensitivity: subset of SENSITIVITIES (preregistration §8, amendment 1.2).
    Primary analysis: Attempted rows (_y = -1) dropped, conflicts kept, ports excluded.
    task: "binary" or "multiclass"; S4 drops rows whose feature vector carries conflicting labels for that task
        (binary: benign vs attack, `_conflict_bin`; multiclass: attack family, `_conflict`).
    S7_unseen_eval[_sig2]: evaluation rows (val, test, external) whose near-duplicate representation at 3 (or 2)
        significant figures occurs in the data the model was fitted or tuned on are removed; train is unchanged.
        near_against: for external targets, the source dataset (default cicids2017_improved), compared on its
        train + val (amendment 1.3). The sidecar is checked against its recorded SHA-256 before use.
    """
    unknown = set(sensitivity) - SENSITIVITIES
    if unknown:
        raise ValueError(f"unknown sensitivity {sorted(unknown)}")
    if task not in ("binary", "multiclass"):
        raise ValueError(f"task must be binary or multiclass, not {task}")
    rep = paths.RESULTS / "data_audit" / ds_id / "report.json"
    if not rep.exists():
        raise FileNotFoundError(f"no preprocessing report for {ds_id}; run scripts/preprocess.py --dataset {ds_id}")
    report = json.loads(rep.read_text())
    require_current(ds_id, report)      # built under the current rules for this dataset (amendment 1.2)
    if report.get("max_rows"):
        raise RuntimeError(f"{ds_id} was preprocessed as a pilot (--max-rows); re-run on the full data "
                           "before any experiment (preregistration §10)")
    files = sorted((paths.PROCESSED / ds_id / version).glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no processed data for {ds_id}; run scripts/preprocess.py --dataset {ds_id}")
    df = pd.concat((pd.read_parquet(f, columns=columns) for f in files), ignore_index=True)
    sp = pd.read_parquet(paths.SPLITS / ds_id / f"{split}.parquet")
    if not np.array_equal(df["_key"].to_numpy(), sp["_key"].to_numpy()):
        raise AssertionError("processed data and split file are out of step; re-run preprocessing")
    df["_block"] = sp["_block"].to_numpy()
    df["_partition"] = sp["partition"].map(PARTITION_NAMES).to_numpy()
    df = df[df["_partition"].isin(partitions)]
    if "S3_attempted_benign" in sensitivity:
        df.loc[df["_y"] < 0, ["_y", "_label"]] = [0, "BENIGN"]
    df = df[df["_y"] >= 0]
    if "S4_drop_conflicts" in sensitivity:
        df = df[df["_conflict_bin" if task == "binary" else "_conflict"] == 0]
    s7 = [s for s in sensitivity if s.startswith("S7")]
    if s7:
        if "S3_attempted_benign" in sensitivity:
            raise ValueError("S7 is defined on the primary label set and is not combined with S3")
        sig = 2 if s7[0].endswith("sig2") else 3
        ext = split == "external"
        against = (near_against or "cicids2017_improved") if ext else None
        require_sidecar(ds_id, against)     # recorded SHA-256, definition and provenance (amendment 1.3)
        side = pd.read_parquet(sidecar_path(ds_id, against), columns=["_key", f"seen_sig{sig}"])
        seen = pd.Series(side[f"seen_sig{sig}"].to_numpy(), index=side["_key"].to_numpy())
        if not df["_key"].isin(seen.index).all():
            raise AssertionError("near-duplicate sidecar does not cover these rows; run --near-duplicates again")
        df = df[~seen.loc[df["_key"]].to_numpy()]
    X = df[feature_columns(df)].copy()
    if "S2_ports" in sensitivity and "_dst_port" in df:
        X["dst_port"] = df["_dst_port"].astype("float32")
    meta = df[[c for c in df.columns if c.startswith(META_PREFIX)]]
    return X.reset_index(drop=True), meta.reset_index(drop=True)
