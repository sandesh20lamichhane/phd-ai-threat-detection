"""Load processed data with a frozen split for experiments. Metadata columns ("_" prefix) never reach models."""
import json

import numpy as np
import pandas as pd

from ..utils import paths
from .split import PARTITION_NAMES

META_PREFIX = "_"


def feature_columns(df):
    return [c for c in df.columns if not c.startswith(META_PREFIX)]


def load(ds_id, split="primary", partitions=("train", "val", "test"), columns=None, sensitivity=(),
         version="v1"):
    """Returns (X, meta) for the requested partitions.

    sensitivity: subset of {"S2_ports", "S3_attempted_benign", "S4_drop_conflicts"} (preregistration §8).
    Primary analysis: Attempted rows (_y = -1) dropped, conflicts kept, ports excluded.
    """
    rep = paths.RESULTS / "data_audit" / ds_id / "report.json"
    if not rep.exists():
        raise FileNotFoundError(f"no preprocessing report for {ds_id}; run scripts/preprocess.py --dataset {ds_id}")
    if json.loads(rep.read_text()).get("max_rows"):
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
        df = df[df["_conflict"] == 0]
    X = df[feature_columns(df)].copy()
    if "S2_ports" in sensitivity and "_dst_port" in df:
        X["dst_port"] = df["_dst_port"].astype("float32")
    meta = df[[c for c in df.columns if c.startswith(META_PREFIX)]]
    return X.reset_index(drop=True), meta.reset_index(drop=True)
