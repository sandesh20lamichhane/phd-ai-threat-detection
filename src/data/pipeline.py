"""Raw files -> deduplicated, identifier-free parquet + frozen splits + audit report, for one dataset.

Streaming, so that the 2018 corrected release and CIC-IoT2023 fit in a Colab session:

  pass 1  read chunks, standardise (src/data/load.py), hash rows, write stage parquet, keep small arrays
  decide  all-NaN rows, exact duplicates, conflicting labels, near-duplicates (on the arrays, in memory)
  pass 2  re-read stage parquet, keep the surviving rows, add flags, write data/processed/<id>/v1/
  split   frozen split rules on the arrays, purging, integrity checks, write data/splits/<id>/
  report  results/data_audit/<id>/report.json and docs/data_cards/<id>.md

Rules: docs/protocol/preregistration.md §3-5. Configuration: config/preprocessing.yaml.
"""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import paths
from ..utils.io import sha256_file, write_json, write_parquet
from ..utils.logging import run_info
from . import split as S
from .clean import combine, round_sig, row_hash
from .labels import ATTEMPTED, BENIGN
from .leakage import split_integrity
from .load import CONFIG_PATH, Standardiser, discover, load_config, read_chunks

OUTPUT_VERSION = "v1"


class Codes:
    """Small string -> int16 code book, so per-row metadata stays compact in memory."""

    def __init__(self):
        self.values, self.index = [], {}

    def encode(self, arr):
        arr = pd.Series(arr, dtype=object).fillna("")
        for v in pd.unique(arr):
            if v not in self.index:
                self.index[v] = len(self.values)
                self.values.append(v)
        return arr.map(self.index).to_numpy(dtype=np.int16)

    def decode(self, codes):
        return np.asarray(self.values, dtype=object)[codes]


def scratch_dir(ds_id):
    base = os.environ.get("PHD_SCRATCH") or ("/content/scratch" if Path("/content").exists() else None)
    root = Path(base) if base else paths.PROCESSED / ds_id
    return root / f"_stage_{ds_id}"


def _manifest(ds_id):
    m = paths.RAW / ds_id / "MANIFEST.json"
    return json.loads(m.read_text()).get("files", {}) if m.exists() else {}


def _all_nan(X):
    return X.isna().all(axis=1).to_numpy() if X.shape[1] else np.ones(len(X), bool)


def preprocess(ds_id, config_path=CONFIG_PATH, max_rows=None, keep_stage=False, log=print):
    cfg = load_config(config_path)
    ds = cfg["datasets"][ds_id]
    if ds.get("role") == "reference_only" or ds.get("reader") == "lanl":
        raise ValueError(f"{ds_id} is not processed by this pipeline (role {ds.get('role')}, reader {ds.get('reader')})")
    sources = discover(ds)
    if not sources:
        raise FileNotFoundError(f"no input files for {ds_id} under {paths.RAW / ds_id} or {paths.INTERIM / ds_id}")
    log(f"[{ds_id}] {len(sources)} input files, {sum(s.size for s in sources) / 1e9:.2f} GB")

    stage = scratch_dir(ds_id)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    out_dir = paths.PROCESSED / ds_id / OUTPUT_VERSION
    std = Standardiser(ds, cfg["label_rules"])
    fam_codes, day_codes, raw_codes, off_codes = Codes(), Codes(), Codes(), Codes()
    arrays = {k: [] for k in ("f", "fl", "near", "fam", "raw", "y", "ts", "dur", "day", "off", "allnan",
                              "file", "row")}
    parts, n_read = [], 0

    # ---- pass 1 ------------------------------------------------------------------------------------------
    for X, meta, src in read_chunks(ds, sources, std):
        if max_rows is not None and n_read >= max_rows:
            break
        fh = row_hash(X)
        arrays["f"].append(fh)
        arrays["fl"].append(combine(fh, meta["_label"].to_numpy()))
        arrays["near"].append(row_hash(round_sig(X, ds["near_duplicate_sig_figs"])))
        arrays["fam"].append(fam_codes.encode(meta["_label"]))
        arrays["raw"].append(raw_codes.encode(meta["_label_raw"]))
        arrays["y"].append(meta["_y"].to_numpy(np.int8))
        arrays["ts"].append(meta["_ts"].to_numpy("float64") if "_ts" in meta else np.full(len(X), np.nan))
        arrays["dur"].append(meta["_dur"].to_numpy("float64") if "_dur" in meta else np.zeros(len(X)))
        arrays["day"].append(day_codes.encode(meta["_day"] if "_day" in meta else [""] * len(X)))
        arrays["off"].append(off_codes.encode(meta["_partition_official"] if "_partition_official" in meta
                                              else [""] * len(X)))
        arrays["allnan"].append(_all_nan(X))
        arrays["file"].append(meta["_file"].to_numpy(np.int16))
        arrays["row"].append(meta["_row"].to_numpy(np.int64))
        p = stage / f"part-{len(parts):05d}.parquet"
        pd.concat([X, meta], axis=1).to_parquet(p, index=False)
        parts.append((p, len(X)))
        n_read += len(X)
        log(f"[{ds_id}] pass 1: {n_read:,} rows ({src.rel})")

    A = {k: np.concatenate(v) for k, v in arrays.items()}
    n = len(A["f"])
    A["key"] = np.arange(n, dtype=np.int64)
    fam = fam_codes.decode(A["fam"])

    # ---- decide ------------------------------------------------------------------------------------------
    keep = ~A["allnan"]
    _, first = np.unique(A["fl"][keep], return_index=True)
    first_mask = np.zeros(keep.sum(), bool)
    first_mask[first] = True
    is_first = np.zeros(n, bool)
    is_first[np.flatnonzero(keep)] = first_mask
    dup = keep & ~is_first
    keep &= is_first

    def _conflicts(codes):
        live = keep & (A["y"] >= 0)
        pairs = pd.DataFrame({"f": A["f"][live], "c": codes[live]}).drop_duplicates()
        bad = pairs.groupby("f").size()
        bad = bad[bad > 1].index.to_numpy(dtype=np.uint64)
        return live & np.isin(A["f"], bad), len(bad)

    conflict, n_conf_vectors = _conflicts(A["fam"])
    conflict_bin, n_conf_bin_vectors = _conflicts(A["y"])
    att = keep & (fam == ATTEMPTED)
    benign_f = np.unique(A["f"][keep & (fam == BENIGN)])
    att_collide = int(np.isin(A["f"][att], benign_f).sum())
    near_kept = A["near"][keep]
    _, inv, cnt = np.unique(near_kept, return_inverse=True, return_counts=True)
    near_within = int((cnt[inv] > 1).sum())

    counts = {"rows_read": n, "all_nan_rows_dropped": int(A["allnan"].sum()),
              "exact_duplicates_removed": int(dup.sum()), "rows_out": int(keep.sum()),
              "attempted_rows": int(att.sum()), "attempted_rows_identical_to_a_benign_flow": att_collide,
              "conflicting_label_feature_vectors": int(n_conf_vectors), "conflict_rows": int(conflict.sum()),
              "conflicting_binary_feature_vectors": int(n_conf_bin_vectors),
              "binary_conflict_rows": int(conflict_bin.sum()),
              "rows_in_near_duplicate_groups": near_within,
              "hash_note": "64-bit row hashes; expected collisions below 1 per 10^3 datasets of 10^8 rows"}
    classes = {"raw_in": dict(Counter(raw_codes.decode(A["raw"]))),
               "family_in": dict(Counter(fam)),
               "family_out": dict(Counter(fam[keep])),
               "duplicates_removed_by_family": dict(Counter(fam[dup]))}
    log(f"[{ds_id}] kept {keep.sum():,} of {n:,}; duplicates {dup.sum():,}; conflicts {conflict.sum():,}")

    # ---- pass 2 ------------------------------------------------------------------------------------------
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    outputs, nan_counts, offset = {}, Counter(), 0
    for i, (p, m) in enumerate(parts):
        sl = slice(offset, offset + m)
        offset += m
        k = keep[sl]
        df = pd.read_parquet(p)[k].reset_index(drop=True)
        df.insert(0, "_key", A["key"][sl][k])
        df["_conflict"] = conflict[sl][k].astype(np.int8)
        df["_conflict_bin"] = conflict_bin[sl][k].astype(np.int8)
        nan_counts.update({c: int(v) for c, v in df[std.feature_columns].isna().sum().items() if v})
        name = f"part-{i:05d}.parquet"
        outputs[name] = write_parquet(df, out_dir / name)
        if not keep_stage:
            p.unlink()
    if not keep_stage:
        shutil.rmtree(stage, ignore_errors=True)

    # ---- split -------------------------------------------------------------------------------------------
    K = {k: v[keep] for k, v in A.items()}
    famk, dayk = fam[keep], day_codes.decode(K["day"])
    rawk = raw_codes.decode(K["raw"])
    splits_report = {}
    for name, spec in (ds.get("splits") or {}).items():
        splits_report[name] = make_split(ds, ds_id, name, spec, K, famk, dayk, std.feature_columns,
                                         off_codes.decode(K["off"]), rawk)
        log(f"[{ds_id}] split {name}: {splits_report[name]['rows_per_partition']}")

    # ---- report ------------------------------------------------------------------------------------------
    man = _manifest(ds_id)
    inputs = []
    for s in sources:
        e = {"file": s.rel, "size": s.size, **s.stats}
        ref = man.get(Path(s.rel).name) or man.get(s.rel)
        if ref and "sha256" in s.stats:
            e["manifest_sha256_match"] = ref.get("sha256") == s.stats["sha256"]
        elif s.member is not None:
            e["zip_sha256_in_manifest"] = man.get(s.path.name, {}).get("sha256")
        else:
            e["manifest_note"] = ("extracted from a zip in data/raw (zip checksummed in the manifest)"
                                  if paths.INTERIM in s.path.parents else "not in the manifest")
        inputs.append(e)
    fam_in_multiclass = multiclass_eligible(K, famk, ds)
    report = {"dataset": ds_id, "role": ds.get("role"), "protocol_version": cfg["version"],
              "output_version": OUTPUT_VERSION, "run": run_info(Path(config_path)),
              "max_rows": max_rows, "inputs": inputs, "schema": std.report(), "counts": counts,
              "classes": classes, "nan_counts_after_cleaning": dict(nan_counts),
              "multiclass_families": fam_in_multiclass, "splits": splits_report,
              "outputs": {"dir": str(out_dir), "parquet_sha256": outputs}}
    dest = paths.RESULTS / "data_audit" / ds_id
    write_json(dest / "report.json", report)
    from .audit import write_data_card
    write_data_card(report, ds)
    log(f"[{ds_id}] report: {dest / 'report.json'}")
    return report


def multiclass_eligible(K, famk, ds):
    blk = S.blocks(K["ts"], ds["block_seconds"])
    out = {}
    for f in sorted(set(famk) - {ATTEMPTED}):
        m = famk == f
        nb = int(len(np.unique(blk[m & (blk >= 0)]))) if np.isfinite(K["ts"]).any() else None
        ok = m.sum() >= ds["multiclass_min_flows"] and (nb is None or nb >= ds["multiclass_min_blocks"])
        out[f] = {"flows": int(m.sum()), "blocks": nb, "in_multiclass_task": bool(ok)}
    return out


def make_split(ds, ds_id, name, spec, K, famk, dayk, feature_columns, official, rawk=None):
    bs, fr, seed = ds["block_seconds"], ds["split_fractions"], ds["split_seed"]
    has_time = np.isfinite(K["ts"]).any()
    blk = S.blocks(K["ts"], bs) if has_time else S.file_row_blocks(K["file"], K["row"], ds["block_rows"])
    t = spec["type"]
    diag, purged = {}, 0
    strata = rawk if (rawk is not None and ds.get("stratum", "raw_label") == "raw_label") else famk
    if t == "chronological_blocks":
        part, diag = S.chronological_block_split(blk, dayk, strata, K["y"], fr, ds["min_blocks_per_stratum"])
        part, purged = S.purge(part, blk, dayk, K["ts"], K["dur"], bs)
    elif t == "by_day":
        part = S.by_day_split(dayk, spec["train"], spec["val"], spec["test"])
        part, purged = S.purge(part, blk, dayk, K["ts"], K["dur"], bs)
    elif t == "external":
        part = np.full(len(blk), S.EXTERNAL, dtype=np.int8)
    elif t == "file_order":
        part = S.file_order_split(K["file"], K["row"], fr)
    elif t == "stratified_random":
        part = S.stratified_random_split(famk, fr, seed)
    elif t == "official":
        part = np.where(np.asarray(official, dtype=object) == "test", S.TEST, S.TRAIN).astype(np.int8)
        tr = part == S.TRAIN
        if spec.get("val_mode") == "file_order":
            sub = S.file_order_split(K["file"][tr], K["row"][tr], fr, which=("train", "val"))
            part[np.flatnonzero(tr)] = sub
        else:
            part = S.official_split(official, famk, spec.get("val_from_train", 0.15), seed)
    else:
        raise ValueError(f"unknown split type {t}")
    if not has_time:   # keep every row block inside one partition
        blk = blk * 8 + (part.astype(np.int64) + 2)
    diag["bootstrap_groups"] = (f"{bs} s time blocks" if has_time else
                                f"runs of {ds['block_rows']} consecutive rows within a file")
    # "Attempted" rows (_y = -1) keep a partition so sensitivity S3 can use them; primary loaders drop them.
    integrity = split_integrity(part, K["fl"], K["f"], K["near"], blk, feature_columns,
                                block_split=t in ("chronological_blocks", "by_day", "file_order"))
    tab = pd.crosstab(pd.Series(famk, name="family"), pd.Series(part, name="partition").map(S.PARTITION_NAMES))
    df = pd.DataFrame({"_key": K["key"], "_block": blk, "partition": part})
    path = paths.SPLITS / ds_id / f"{name}.parquet"
    sha = write_parquet(df, path)
    return {"type": t, "spec": spec, "seed": seed, "block_seconds": bs,
            "rows_per_partition": pd.Series(part).map(S.PARTITION_NAMES).value_counts().to_dict(),
            "family_by_partition": {c: {k: int(v) for k, v in tab[c].items()} for c in tab.columns},
            "purged_rows": purged, "diagnostics": diag, "integrity": integrity,
            "file": str(path), "sha256": sha}
